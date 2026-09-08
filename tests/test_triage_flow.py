"""Kiểm tra các invariant của luồng condition-based triage phiên bản mới."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.artifact import find_latest_release, verify_release_bundle
from src.contracts import MODEL_FEATURE_CONTRACT, RUNTIME_METADATA_FIELDS
from src.data import _is_valid_split_registry
from src.features import build_canonical_features, canonicalize_raw_dataframe
from src.inference import RiskInferenceService
from src.monitoring import assess_distribution_guardrails
from src.policy import build_maintenance_queue, failure_capture_at_k
from src.storage import SQLiteRiskEventStore
from src.utils import normalize_event_time


def test_runtime_metadata_is_not_a_model_feature() -> None:
    """Metadata dùng cho vận hành nhưng không được lọt vào feature vector."""
    assert not set(RUNTIME_METADATA_FIELDS) & set(MODEL_FEATURE_CONTRACT)


def test_missing_reference_distribution_is_unavailable() -> None:
    """Thiếu reference phải fail-closed thay vì bị coi là nominal."""
    features = pd.DataFrame({"torque_nm": [40.0]})
    status, warnings = assess_distribution_guardrails(features, {})
    assert status == "UNAVAILABLE"
    assert warnings


def test_incomplete_reference_distribution_is_unavailable() -> None:
    """Reference thiếu một feature hoặc một percentile phải fail-closed."""
    features = pd.DataFrame({"torque_nm": [40.0], "tool_wear_min": [50.0]})
    reference_ranges = {
        "torque_nm": {"p0_5": 10.0, "p99_5": 80.0},
        "tool_wear_min": {"p0_5": 0.0},
    }
    status, warnings = assess_distribution_guardrails(features, reference_ranges)
    assert status == "UNAVAILABLE"
    assert warnings


def test_inference_rejects_target_derived_payload() -> None:
    """Payload online không được mang target hoặc failure-mode metadata."""
    service = RiskInferenceService.get_instance()
    with pytest.raises(ValueError, match="bị cấm"):
        service.predict(
            {
                "product_quality_type": "M",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 50.0,
                "machine_failure": 1,
            }
        )


def test_queue_keeps_latest_event_and_priority_override() -> None:
    """Queue loại event cũ theo asset và chỉ vượt capacity khi là priority override."""
    events = [
        {
            "event_id": "a-old",
            "asset_id": "A",
            "event_time": "2026-09-07T10:00:00Z",
            "risk_score": 0.99,
            "action": "PRIORITY_REVIEW",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
        {
            "event_id": "a-new",
            "asset_id": "A",
            "event_time": "2026-09-07T11:00:00Z",
            "risk_score": 0.10,
            "action": "NO_ALERT",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
        {
            "event_id": "b-priority",
            "asset_id": "B",
            "event_time": "2026-09-07T11:00:00Z",
            "risk_score": 0.80,
            "action": "PRIORITY_REVIEW",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
        {
            "event_id": "c-review",
            "asset_id": "C",
            "event_time": "2026-09-07T11:00:00Z",
            "risk_score": 0.70,
            "action": "REVIEW_REQUIRED",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
        {
            "event_id": "d-review",
            "asset_id": "D",
            "event_time": "2026-09-07T11:00:00Z",
            "risk_score": 0.60,
            "action": "REVIEW_REQUIRED",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
    ]
    queue = build_maintenance_queue(events, capacity=1)
    assert [item["asset_id"] for item in queue] == ["B", "C"]


def test_failure_capture_at_k_ranks_by_risk() -> None:
    """Failure Capture@K phải lấy đúng nhóm risk cao nhất."""
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.2, 0.9, 0.8, 0.1])
    assert failure_capture_at_k(labels, probabilities, 0.5) == 0.5


def test_engineered_features_are_recomputed_from_raw_sensors() -> None:
    """Caller không được ghi đè engineered feature bằng giá trị không nhất quán."""
    features = build_canonical_features(
        pd.DataFrame(
            {
                "product_quality_type": ["M"],
                "air_temperature_k": [300.0],
                "process_temperature_k": [310.0],
                "rotational_speed_rpm": [1500.0],
                "torque_nm": [40.0],
                "tool_wear_min": [20.0],
                "temperature_delta_k": [999.0],
                "mechanical_power_w": [999.0],
                "wear_load_interaction": [999.0],
            }
        )
    )
    assert features.loc[0, "temperature_delta_k"] == 10.0
    assert features.loc[0, "wear_load_interaction"] == 800.0


def test_conflicting_quality_aliases_are_rejected() -> None:
    """Hai alias cùng trường nhưng khác giá trị không được âm thầm chọn một giá trị."""
    with pytest.raises(ValueError, match="mâu thuẫn"):
        canonicalize_raw_dataframe(
            pd.DataFrame(
                {
                    "Type": ["L"],
                    "machine_type": ["H"],
                }
            )
        )


def test_non_persistent_prediction_does_not_create_runtime_event() -> None:
    """Readiness và dashboard có thể chấm điểm mà không làm bẩn event store."""
    service = RiskInferenceService.get_instance()
    before = len(service.risk_events)
    service.predict(
        {
            "product_quality_type": "M",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 50.0,
        },
        persist_event=False,
    )
    assert len(service.risk_events) == before


def test_split_registry_requires_disjoint_complete_indices() -> None:
    """Split manifest phải bao phủ đủ dòng và không có index chồng lấn."""
    registry = {
        "seed": 42,
        "split_fractions": {
            "development": 0.70,
            "policy_validation": 0.15,
            "locked_test": 0.15,
        },
        "split_counts": {"development": 7, "policy_validation": 1, "test": 2},
        "development_indices": list(range(7)),
        "policy_indices": [7],
        "test_indices": [8, 9],
    }
    assert _is_valid_split_registry(registry, total_rows=10, seed=42)
    registry["test_indices"] = [7, 9]
    assert not _is_valid_split_registry(registry, total_rows=10, seed=42)


def test_event_time_is_normalized_to_utc() -> None:
    """Các timezone khác nhau phải được quy về một mốc UTC để queue xếp đúng."""
    assert normalize_event_time("2026-09-07T17:15:00+07:00") == "2026-09-07T10:15:00Z"
    with pytest.raises(ValueError, match="timezone"):
        normalize_event_time("2026-09-07T10:15:00")


def test_queue_compares_timezone_offsets_and_skips_invalid_time() -> None:
    """Queue so sánh theo instant UTC, không theo chuỗi offset và không nhận timestamp lỗi."""
    events = [
        {
            "event_id": "old",
            "asset_id": "A",
            "event_time": "2026-09-07T10:00:00Z",
            "risk_score": 0.2,
            "action": "REVIEW_REQUIRED",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
        {
            "event_id": "new",
            "asset_id": "A",
            "event_time": "2026-09-07T18:00:00+07:00",
            "risk_score": 0.8,
            "action": "PRIORITY_REVIEW",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
        {
            "event_id": "invalid",
            "asset_id": "B",
            "event_time": "not-a-time",
            "risk_score": 1.0,
            "action": "PRIORITY_REVIEW",
            "reliability_status": "NOMINAL",
            "queue_eligible": True,
        },
    ]
    queue = build_maintenance_queue(events, capacity=1)
    assert [item["event_id"] for item in queue] == ["new"]


def test_storage_upsert_preserves_existing_review(tmp_path: Path) -> None:
    """Cập nhật event không được xóa review đã liên kết."""
    store = SQLiteRiskEventStore(tmp_path / "events.sqlite3")
    event = {
        "event_id": "evt-1",
        "asset_id": "A",
        "event_time": "2026-09-07T10:00:00Z",
        "shift": "day",
        "model_version": "v1",
        "policy_version": "p1",
        "risk_score": 0.8,
        "reliability_status": "NOMINAL",
        "action": "PRIORITY_REVIEW",
        "queue_eligible": True,
        "observed_conditions": [],
    }
    store.record_event(event, {"torque_nm": 40.0})
    store.record_review("evt-1", "A", "inspected", True)
    event["risk_score"] = 0.9
    store.record_event(event, {"torque_nm": 45.0})
    store.record_review("evt-1", "A", "closed", True)


def test_latest_release_has_valid_manifest() -> None:
    """Release đang được serving phải tự kiểm tra được checksum."""
    release = find_latest_release()
    assert release is not None
    checks = verify_release_bundle(release)
    assert checks["hashes_match"] is True
