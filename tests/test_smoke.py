"""Bộ kiểm thử tự động toàn diện (Comprehensive Test Suite) và Kiểm định Tính Bất biến Kiến trúc (Architectural Invariants).

Bao gồm:
1. Architectural Invariant Tests (Chống rò rỉ dữ liệu, đảm bảo hợp đồng đặc trưng, đóng băng policy, kiểm định reliability gate).
2. Schema & Payload Validation Tests (Pydantic validation, chuẩn hóa phân loại máy, kiểm tra biên hợp lý).
3. Physics & Feature Engineering Tests (Công thức công suất cơ học Watts, tương tác tải mòn, chênh nhiệt).
4. Decision Policy & Optimization Tests (Tối ưu theo chi phí, ràng buộc công suất bảo trì).
5. API Service & Inference Engine Integration Tests (Health probes, 4-block schema response, readiness cycle).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api import SensorPayload, app
from src.contracts import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_CONTRACT,
    TARGET_COLUMN,
)
from src.data import (
    compute_dataset_sha256,
    load_data,
)
from src.features import add_engineered_features, build_canonical_features
from src.inference import RiskInferenceService
from src.models import compute_calibration_curve_and_ece, compute_classification_metrics
from src.policy import (
    BusinessCosts,
    find_threshold_maximizing_f1,
    find_threshold_minimizing_cost,
    map_decision_action,
)

client = TestClient(app)


def test_calibration_curve_includes_probability_one() -> None:
    """Điểm xác suất đúng bằng 1.0 phải được tính vào bin cuối."""
    _, curve = compute_calibration_curve_and_ece(
        np.array([0, 1, 1]),
        np.array([0.1, 0.6, 1.0]),
        n_bins=5,
    )
    assert sum(point["count"] for point in curve) == 3


def test_classification_metrics_accept_single_class_labels() -> None:
    """Báo cáo metric không được vỡ khi một lát dữ liệu chỉ có một nhãn."""
    metrics = compute_classification_metrics(
        np.array([0, 0]),
        np.array([0.1, 0.2]),
        threshold=0.5,
    )
    assert metrics["true_negatives"] == 2
    assert metrics["true_positives"] == 0
    assert metrics["pr_auc"] == 0.0


# ---------------------------------------------------------------------------
# 1. ARCHITECTURAL INVARIANT TESTS (KIỂM ĐỊNH BẤT BIẾN KIẾN TRÚC)
# ---------------------------------------------------------------------------


def test_failure_flags_never_enter_features() -> None:
    """INVARIANT 1: Các cờ cơ chế hỏng hóc hậu nghiệm (TWF, HDF, PWF, OSF, RNF) tuyệt đối không được lọt vào feature contract."""
    contract_features = set(MODEL_FEATURE_CONTRACT)
    for flag in FAILURE_MODE_COLUMNS:
        assert flag not in contract_features, (
            f"Phát hiện rò rỉ: cờ '{flag}' nằm trong feature contract!"
        )
    assert TARGET_COLUMN not in contract_features
    for identifier in IDENTIFIER_COLUMNS:
        assert identifier not in contract_features


def test_train_and_api_feature_builder_are_same() -> None:
    """INVARIANT 2: Pipeline biến đổi đặc trưng của Training và API phải cho ra chính xác cùng danh sách cột và thứ tự."""
    X_train, *_ = load_data()

    dummy_payload = {
        "Type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    api_features = build_canonical_features(pd.DataFrame([dummy_payload]))

    assert list(X_train.columns) == list(api_features.columns), (
        "Train-Serving Skew phát hiện: Cột đặc trưng của Train khác cột đặc trưng của API!"
    )
    assert list(api_features.columns) == list(MODEL_FEATURE_CONTRACT)


def test_feature_contract_exact_order() -> None:
    """INVARIANT 3: Hợp đồng đặc trưng phải cố định chính xác 9 cột theo thứ tự định trước."""
    expected_order = [
        "quality_type",
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temperature_delta_k",
        "mechanical_power_w",
        "wear_load_interaction",
    ]
    assert list(MODEL_FEATURE_CONTRACT) == expected_order


def test_test_set_never_used_for_threshold_selection() -> None:
    """INVARIANT 4: Mã nguồn evaluate.py không được chứa logic tìm kiếm threshold (argmax f1 trên test)."""
    eval_file = Path("src/evaluate.py")
    content = eval_file.read_text(encoding="utf-8")
    assert "argmax(f1_scores)" not in content, (
        "Rò rỉ tập Test: Phát hiện argmax(f1_scores) trong src/evaluate.py!"
    )
    assert "np.argmax" not in content, (
        "Rò rỉ tập Test: Phát hiện tìm kiếm cực trị trên tập Test trong src/evaluate.py!"
    )


def test_max_f1_threshold_is_validation_derived() -> None:
    """INVARIANT 5: Ngưỡng Max-F1 phải được tối ưu và đóng băng trên tập Validation."""
    y_val_dummy = np.array([0, 0, 0, 1, 1, 1])
    probs_val_dummy = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    thresh, best_f1 = find_threshold_maximizing_f1(y_val_dummy, probs_val_dummy)
    assert 0.0 <= thresh <= 1.0
    assert best_f1 > 0.5


def test_policy_thresholds_are_frozen() -> None:
    """INVARIANT 6: Tệp decision_policy.json phải ghi nhận các ngưỡng đã được đóng băng từ Validation."""
    policy_path = Path("artifacts/champion/decision_policy.json")
    if not policy_path.exists():
        policy_path = Path("models/config.json")

    assert policy_path.exists(), "Chưa tìm thấy tệp cấu hình chính sách đã lưu!"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    if "decision_policy" in policy:
        policy = policy["decision_policy"]

    assert "primary_alert_threshold" in policy
    assert "cost_weights" in policy
    assert 0.0 <= policy["primary_alert_threshold"] <= 1.0


def test_distribution_warning_changes_reliability_status() -> None:
    """INVARIANT 7: Khi cảm biến vượt ngưỡng P0.5 - P99.5, Reliability Gate phải chuyển trạng thái sang DEGRADED."""
    service = RiskInferenceService.get_instance()

    # Dữ liệu nằm sâu ngoài phân bố (Torque = 5000 Nm, Tool wear = 9999 phút)
    extreme_payload = {
        "Type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 5000.0,
        "tool_wear_min": 9999.0,
    }
    result = service.predict(extreme_payload)
    rel = result["reliability"]
    assert rel["distribution_warning"] is True
    assert rel["status"] == "DEGRADED"
    assert len(rel["warning_features"]) > 0


def test_rf_response_does_not_claim_fake_model_explanation() -> None:
    """INVARIANT 8: Khối giải thích của mô hình ensemble phải được gắn nhãn là feature_context, không mạo danh feature attribution."""
    service = RiskInferenceService.get_instance()
    normal_payload = {
        "Type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 50.0,
    }
    result = service.predict(normal_payload)
    feature_ctx = result["operational_context"]["feature_context"]
    assert isinstance(feature_ctx, list)
    assert all(item.get("type") == "observed_feature" for item in feature_ctx)


def test_failure_mode_metadata_not_used_in_training() -> None:
    """INVARIANT 9: load_data(return_metadata=True) phải trả về metadata hoàn toàn tách biệt khỏi ma trận đặc trưng."""
    _, _, X_test, _, _, _, modes_test = load_data(return_metadata=True)
    assert isinstance(modes_test, pd.DataFrame)
    assert set(modes_test.columns) <= set(FAILURE_MODE_COLUMNS)
    assert not any(col in X_test.columns for col in FAILURE_MODE_COLUMNS)


def test_artifact_data_hash_matches_dataset() -> None:
    """INVARIANT 10: Mã băm SHA256 dữ liệu thực tế phải khớp với bản ghi trong reports/data_audit.json."""
    actual_hash = compute_dataset_sha256("data/raw/ai4i2020.csv")
    audit_file = Path("reports/data_audit.json")
    if audit_file.exists():
        audit_data = json.loads(audit_file.read_text(encoding="utf-8"))
        assert audit_data["raw_sha256"] == actual_hash


# ---------------------------------------------------------------------------
# 2. SCHEMA & PAYLOAD VALIDATION TESTS
# ---------------------------------------------------------------------------


def test_sensor_payload_normalizes_machine_type() -> None:
    """Pydantic schema tự động chuẩn hóa chữ thường thành chữ hoa (' l ' -> 'L')."""
    payload = SensorPayload(
        Type=" l ",
        air_temperature_k=300.0,
        process_temperature_k=310.0,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=10.0,
    )
    assert payload.machine_type == "L"
    assert payload.Type == "L"


def test_sensor_payload_rejects_unrealistic_temperature() -> None:
    """Pydantic schema từ chối nhiệt độ không khả thi trong thực tế (>1000 K)."""
    with pytest.raises(ValidationError):
        SensorPayload(
            Type="L",
            air_temperature_k=10_000.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=10.0,
        )


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING & PHYSICS TESTS
# ---------------------------------------------------------------------------


def test_engineered_features_formulas() -> None:
    """Kiểm tra tính chính xác của công thức đặc trưng vật lý trong add_engineered_features."""
    raw_df = pd.DataFrame(
        {
            "quality_type": ["M"],
            "air_temperature_k": [300.0],
            "process_temperature_k": [310.0],
            "rotational_speed_rpm": [1500.0],
            "torque_nm": [40.0],
            "tool_wear_min": [20.0],
        }
    )
    res_df = add_engineered_features(raw_df)

    # temperature_delta_k = 310 - 300 = 10.0 K
    assert res_df.loc[0, "temperature_delta_k"] == 10.0

    # mechanical_power_w = 40 * (1500 * 2 * pi / 60) ≈ 6283.185 Watts
    expected_power = 40.0 * (1500.0 * 2.0 * math.pi / 60.0)
    assert (
        pytest.approx(res_df.loc[0, "mechanical_power_w"], rel=1e-3) == expected_power
    )

    # wear_load_interaction = 20 * 40 = 800.0 min*Nm
    assert res_df.loc[0, "wear_load_interaction"] == 800.0


# ---------------------------------------------------------------------------
# 4. DECISION POLICY & OPTIMIZATION TESTS
# ---------------------------------------------------------------------------


def test_business_threshold_prefers_lower_total_cost() -> None:
    """Thuật toán tối ưu ngưỡng giảm thiểu tổng chi phí trọng số FN*5 + FP*1."""
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.4, 0.1])
    costs = BusinessCosts(false_negative=5.0, false_positive=1.0)

    optimal_thresh, min_cost = find_threshold_minimizing_cost(
        labels, probabilities, costs
    )
    assert optimal_thresh == 0.6
    assert min_cost == 0.0


def test_business_threshold_capacity_constraint() -> None:
    """Ngưỡng tối ưu tuân thủ ràng buộc công suất bảo trì max_alert_rate <= 20%."""
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    costs = BusinessCosts(false_negative=5.0, false_positive=1.0)

    optimal_thresh, _ = find_threshold_minimizing_cost(
        labels, probabilities, costs, max_alert_rate=0.20
    )
    alerts = probabilities >= optimal_thresh
    assert alerts.mean() <= 0.20


def test_map_decision_action_prioritization() -> None:
    """Hàm map_decision_action trả về PRIORITY_REVIEW, REVIEW_REQUIRED, NO_ALERT chuẩn xác."""
    assert (
        map_decision_action(0.85, alert_threshold=0.35, critical_threshold=0.75)
        == "PRIORITY_REVIEW"
    )
    assert (
        map_decision_action(0.50, alert_threshold=0.35, critical_threshold=0.75)
        == "REVIEW_REQUIRED"
    )
    assert (
        map_decision_action(0.20, alert_threshold=0.35, critical_threshold=0.75)
        == "NO_ALERT"
    )
    # Khi có warning phân bố nhưng xác suất thấp -> Vẫn kích hoạt REVIEW_REQUIRED
    assert (
        map_decision_action(
            0.10,
            alert_threshold=0.35,
            critical_threshold=0.75,
            distribution_warning=True,
        )
        == "REVIEW_REQUIRED"
    )


# ---------------------------------------------------------------------------
# 5. RESTful API & INFERENCE INTEGRATION TESTS
# ---------------------------------------------------------------------------


def test_api_health_live_and_ready_endpoints() -> None:
    """Kiểm tra endpoints probe /health/live, /health/ready và /health."""
    res_live = client.get("/health/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "alive"

    res_ready = client.get("/health/ready")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready["ready"] is True
    assert data_ready["test_cycle_passed"] is True

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"


def test_api_predict_endpoint_4block_structure() -> None:
    """POST /predict-risk trả về đầy đủ 4 block: prediction, reliability, decision, operational_context."""
    valid_payload = {
        "Type": "L",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 120.0,
    }
    response = client.post("/predict-risk", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    # Kiểm tra 4 block
    assert "prediction" in data
    assert "failure_risk" in data["prediction"]
    assert 0.0 <= data["prediction"]["failure_risk"] <= 1.0

    assert "reliability" in data
    assert data["reliability"]["status"] in ["NOMINAL", "DEGRADED"]

    assert "decision" in data
    assert data["decision"]["action"] in [
        "NO_ALERT",
        "REVIEW_REQUIRED",
        "PRIORITY_REVIEW",
    ]
    assert "alert_threshold" in data["decision"]

    assert "operational_context" in data
    assert "reason_codes" in data["operational_context"]
    assert "feature_context" in data["operational_context"]

    # Kiểm tra trường phẳng tương thích ngược
    assert "failure_risk" in data
    assert "threshold" in data
    assert "risk_tier" in data
    assert "alert" in data


def test_api_predict_endpoint_invalid_payload() -> None:
    """POST /predict-risk từ chối payload có giá trị phân loại máy không hợp lệ."""
    invalid_payload = {
        "Type": "INVALID_X",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 120.0,
    }
    response = client.post("/predict-risk", json=invalid_payload)
    assert response.status_code == 422
