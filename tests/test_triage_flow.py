"""Kiểm tra các invariant của luồng condition-based triage phiên bản mới."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.artifact import find_latest_release, verify_release_bundle
from src.contracts import MODEL_FEATURE_CONTRACT, RUNTIME_METADATA_FIELDS
from src.inference import RiskInferenceService
from src.monitoring import assess_distribution_guardrails
from src.policy import build_maintenance_queue, failure_capture_at_k


def test_runtime_metadata_is_not_a_model_feature() -> None:
    """Metadata dùng cho vận hành nhưng không được lọt vào feature vector."""
    assert not set(RUNTIME_METADATA_FIELDS) & set(MODEL_FEATURE_CONTRACT)


def test_missing_reference_distribution_is_unavailable() -> None:
    """Thiếu reference phải fail-closed thay vì bị coi là nominal."""
    features = pd.DataFrame({"torque_nm": [40.0]})
    status, warnings = assess_distribution_guardrails(features, {})
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


def test_latest_release_has_valid_manifest() -> None:
    """Release đang được serving phải tự kiểm tra được checksum."""
    release = find_latest_release()
    assert release is not None
    checks = verify_release_bundle(release)
    assert checks["hashes_match"] is True
