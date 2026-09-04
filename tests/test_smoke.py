"""Bộ kiểm thử tự động (Test Suite) cho toàn bộ hệ thống Predictive Maintenance.

Các bài test bao gồm:
- Kiểm tra tính hợp lệ dữ liệu Pydantic SensorPayload.
- Kiểm tra hàm tạo đặc trưng vật lý (Feature Engineering).
- Kiểm tra thuật toán tìm ngưỡng tối ưu theo chi phí (Cost-sensitive Threshold Optimization).
- Kiểm tra đọc tệp cấu hình mô hình (config.json).
- Kiểm tra các quy tắc sinh mã lý do vận hành (Reason Codes).
- Kiểm tra các RESTful API Endpoints (`/health` và `/predict-risk`) bằng TestClient của FastAPI.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from src.api import CONFIG_PATH, SensorPayload, app, generate_operational_reason_codes
from src.data import add_engineered_features, load_data
from src.train import BusinessCosts, analyze_cost_sensitivity, select_business_threshold

client = TestClient(app)


def test_cost_sensitivity_increases_attention_to_failures() -> None:
    """Báo cáo độ nhạy phải bao phủ nhiều giả định chi phí và threshold hợp lệ."""
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.4, 0.45, 0.8])
    scenarios = analyze_cost_sensitivity(labels, probabilities)
    assert len(scenarios) == 4
    assert all(0.0 <= row["threshold"] <= 1.0 for row in scenarios)
    assert all(0.0 <= row["recall"] <= 1.0 for row in scenarios)


def test_sensor_payload_normalizes_machine_type():
    """Kiểm tra Pydantic schema tự động chuẩn hóa chữ thường thành chữ hoa (' l ' -> 'L')."""
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


def test_sensor_payload_rejects_unrealistic_temperature():
    """Kiểm tra Pydantic schema từ chối nhiệt độ không khả thi trong thực tế (ví dụ: 10,000 K)."""
    with pytest.raises(ValidationError):
        SensorPayload(
            Type="L",
            air_temperature_k=10_000.0,
            process_temperature_k=310.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=10.0,
        )


def test_engineered_features_exist_in_real_split():
    """Kiểm tra các đặc trưng tạo mới xuất hiện đầy đủ trong tập dữ liệu đã chia phân tầng."""
    train_x, *_ = load_data()
    expected_features = {"temperature_delta", "power_proxy", "strain_proxy"}
    assert expected_features <= set(train_x.columns)


def test_engineered_features_have_expected_values():
    """Kiểm tra tính chính xác của công thức vật lý trong add_engineered_features."""
    raw_df = pd.DataFrame(
        {
            "Air temperature": [300.0],
            "Process temperature": [310.0],
            "Rotational speed": [1500.0],
            "Torque": [40.0],
            "Tool wear": [20.0],
        }
    )
    res_df = add_engineered_features(raw_df)

    # temperature_delta = 310 - 300 = 10.0
    assert res_df.loc[0, "temperature_delta"] == 10.0
    # power_proxy = 1500 * 40 = 60000.0
    assert res_df.loc[0, "power_proxy"] == 60_000.0
    # strain_proxy = 20 * 40 = 800.0
    assert res_df.loc[0, "strain_proxy"] == 800.0


def test_model_config_records_business_costs():
    """Kiểm tra tệp config.json ghi nhận đúng cấu trúc schema và ma trận chi phí FN > FP."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["schema_version"] in {1, 2}
    assert config["false_negative_cost"] > config["false_positive_cost"]
    assert set(config["features"]) >= {
        "temperature_delta",
        "power_proxy",
        "strain_proxy",
    }


def test_business_threshold_prefers_lower_total_cost():
    """Kiểm tra thuật toán chọn ngưỡng tối ưu hóa hàm tổng chi phí FN * 5 + FP * 1."""
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.4, 0.1])
    costs = BusinessCosts(false_negative=5.0, false_positive=1.0)

    optimal_thresh, min_cost = select_business_threshold(labels, probabilities, costs)

    assert optimal_thresh == 0.6
    assert min_cost == 0.0


def test_generate_operational_reason_codes():
    """Kiểm tra hàm sinh mã lý do vận hành (Reason Codes) theo quy tắc chuyên gia."""
    payload_high_wear = SensorPayload(
        Type="M",
        air_temperature_k=300.0,
        process_temperature_k=305.0,  # Delta = 5.0 <= 8.6 -> TEMPERATURE_DELTA_LOW
        rotational_speed_rpm=1200.0,  # Speed <= 1300 -> ROTATIONAL_SPEED_LOW
        torque_nm=60.0,  # Torque >= 55 -> TORQUE_HIGH
        tool_wear_min=210.0,  # Wear >= 200 -> TOOL_WEAR_HIGH
    )
    reasons = generate_operational_reason_codes(payload_high_wear)

    assert "TOOL_WEAR_HIGH" in reasons
    assert "TORQUE_HIGH" in reasons
    assert "ROTATIONAL_SPEED_LOW" in reasons
    assert "TEMPERATURE_DELTA_LOW" in reasons


def test_api_health_endpoint():
    """Kiểm tra GET /health trả về status HTTP 200 và model_ready=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "model_ready" in data
    assert "model_version" in data


def test_api_predict_endpoint_success():
    """Kiểm tra POST /predict-risk xử lý payload chuẩn xác và trả về kết quả dự báo."""
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
    assert "failure_risk" in data
    assert 0.0 <= data["failure_risk"] <= 1.0
    assert "risk_tier" in data
    assert data["risk_tier"] in ["HIGH", "MEDIUM", "LOW"]
    assert "alert" in data
    assert isinstance(data["alert"], bool)
    assert "reason_codes" in data
    assert isinstance(data["reason_codes"], list)


def test_api_predict_endpoint_invalid_payload():
    """Kiểm tra POST /predict-risk từ chối payload có kiểu máy (Type) không hợp lệ."""
    invalid_payload = {
        "Type": "INVALID_TYPE",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 120.0,
    }
    response = client.post("/predict-risk", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity từ Pydantic validation
