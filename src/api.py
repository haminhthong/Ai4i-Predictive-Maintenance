"""FastAPI RESTful Service dự báo rủi ro hỏng hóc thiết bị máy móc sản xuất.

Dịch vụ này cung cấp 2 endpoints chính:
1. `GET /health`: Kiểm tra sức khỏe hệ thống và trạng thái sẵn sàng của mô hình ML.
2. `POST /predict-risk`: Nhận dữ liệu cảm biến thời gian thực, tính toán xác suất hỏng hóc,
   phân cấp mức độ rủi ro, kiểm tra ngưỡng phát cảnh báo và sinh các lý do vận hành (Reason Codes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# Xác định đường dẫn tuyệt đối tới tệp mô hình và tệp cấu hình
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models/config.json"

# Khởi tạo ứng dụng FastAPI với đầy đủ metadata OpenAPI
app = FastAPI(
    title="Predictive Maintenance Risk Service",
    description="Dịch vụ AI dự báo rủi ro sự cố máy móc và đề xuất hành động bảo trì theo chi phí nghiệp vụ.",
    version="2.0.0",
)

# Biến toàn cục dùng lưu cache mô hình và cấu hình (Lazy Loading Pattern)
_model: Any | None = None
_config: dict[str, Any] | None = None


class SensorPayload(BaseModel):
    """Schema Pydantic kiểm định dữ liệu cảm biến đầu vào từ thiết bị sản xuất."""

    machine_type: str = Field(
        default="M",
        alias="Type",
        description="Loại chất lượng máy móc: L (Low), M (Medium), H (High)",
    )
    air_temperature_k: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Nhiệt độ môi trường/không khí xung quanh (Đơn vị: Kelvin K)",
    )
    process_temperature_k: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Nhiệt độ quá trình vận hành của máy (Đơn vị: Kelvin K)",
    )
    rotational_speed_rpm: float = Field(
        ...,
        gt=0,
        le=100_000,
        description="Tốc độ quay của trục máy (Đơn vị: Vòng/phút RPM)",
    )
    torque_nm: float = Field(
        ...,
        ge=0,
        le=10_000,
        description="Mô-men xoắn hoạt động (Đơn vị: Newton mét Nm)",
    )
    tool_wear_min: float = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Thời gian độ mòn công cụ tích lũy (Đơn vị: Phút min)",
    )

    @field_validator("machine_type")
    @classmethod
    def validate_machine_type(cls, value: str) -> str:
        """Chuẩn hóa và kiểm tra loại máy hợp lệ (L, M, H)."""
        normalized = value.upper().strip()
        if normalized not in {"L", "M", "H"}:
            raise ValueError(
                "Phân loại máy (Type) phải thuộc một trong các loại: 'L', 'M', hoặc 'H'"
            )
        return normalized

    @property
    def Type(self) -> str:
        """Thuộc tính hỗ trợ tính tương thích ngược với API cũ."""
        return self.machine_type


class HealthResponse(BaseModel):
    """Schema dữ liệu phản hồi của endpoint /health."""

    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_ready: bool = Field(..., json_schema_extra={"example": True})
    model_version: str = Field(..., json_schema_extra={"example": "ai4i-calibrated-v2"})


class PredictResponse(BaseModel):
    """Schema dữ liệu phản hồi của endpoint /predict-risk."""

    failure_risk: float = Field(
        ..., description="Xác suất dự báo máy bị hỏng [0.0 - 1.0]"
    )
    threshold: float = Field(..., description="Ngưỡng quyết định phát cảnh báo tối ưu")
    risk_tier: str = Field(..., description="Mức độ rủi ro: HIGH, MEDIUM, LOW")
    alert: bool = Field(..., description="Cờ cảnh báo (True nếu rủi ro >= threshold)")
    reason_codes: list[str] = Field(
        ..., description="Danh sách mã lý do vận hành minh bạch"
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang sử dụng")


def get_model_and_config() -> tuple[Any, dict[str, Any]]:
    """Tải và lưu cache mô hình ML kèm tệp cấu hình (Thread-safe singleton/lazy load)."""
    global _model, _config
    if _model is None or _config is None:
        if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
            raise FileNotFoundError(
                "Mô hình chưa được huấn luyện. Thiếu model.joblib hoặc config.json."
            )

        _model = joblib.load(MODEL_PATH)
        _config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    return _model, _config


def build_feature_row(payload: SensorPayload, feature_names: list[str]) -> pd.DataFrame:
    """Ánh xạ các trường từ SensorPayload sang định dạng Dataframe tương thích với mô hình đã huấn luyện.

    Tự động tính toán 3 đặc trưng vật lý bổ sung:
    - `temperature_delta = process_temperature - air_temperature`
    - `power_proxy = rotational_speed * torque`
    - `strain_proxy = tool_wear * torque`
    """
    feature_map = {
        "type": payload.machine_type,
        "air temperature": payload.air_temperature_k,
        "process temperature": payload.process_temperature_k,
        "rotational speed": payload.rotational_speed_rpm,
        "torque": payload.torque_nm,
        "tool wear": payload.tool_wear_min,
        "temperature_delta": payload.process_temperature_k - payload.air_temperature_k,
        "power_proxy": payload.rotational_speed_rpm * payload.torque_nm,
        "strain_proxy": payload.tool_wear_min * payload.torque_nm,
    }

    row_data: dict[str, float | str] = {}
    for feature in feature_names:
        matched_key = next(
            (k for k in feature_map if feature.lower().startswith(k)),
            None,
        )
        if matched_key is None:
            raise KeyError(f"Đặc trưng chưa được ánh xạ trong hệ thống API: {feature}")
        row_data[feature] = feature_map[matched_key]

    return pd.DataFrame([row_data], columns=feature_names)


def generate_operational_reason_codes(payload: SensorPayload) -> list[str]:
    """Sinh danh sách các mã lý do vận hành (Reason Codes) dựa trên quy tắc chuyên gia minh bạch.

    Lưu ý: Các mã lý do này hoạt động độc lập với dự báo của mô hình ML nhằm cung cấp
    thông tin rõ ràng cho kỹ sư vận hành tại nhà máy.
    """
    reasons: list[str] = []

    # 1. Cảnh báo độ mòn công cụ cao (>= 200 phút)
    if payload.tool_wear_min >= 200:
        reasons.append("TOOL_WEAR_HIGH")

    # 2. Cảnh báo mô-men xoắn quá tải (>= 55 Nm)
    if payload.torque_nm >= 55:
        reasons.append("TORQUE_HIGH")

    # 3. Cảnh báo tốc độ quay bất thường thấp (<= 1300 RPM)
    if payload.rotational_speed_rpm <= 1300:
        reasons.append("ROTATIONAL_SPEED_LOW")

    # 4. Cảnh báo chênh lệch nhiệt độ thấp (<= 8.6 K)
    temperature_delta = payload.process_temperature_k - payload.air_temperature_k
    if temperature_delta <= 8.6:
        reasons.append("TEMPERATURE_DELTA_LOW")

    return reasons


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> dict[str, Any]:
    """Endpoint kiểm tra trạng thái hoạt động của hệ thống (Health Check)."""
    is_ready = MODEL_PATH.exists() and CONFIG_PATH.exists()
    version_str = "not_trained"

    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            version_str = cfg.get("version", "unknown")
        except (OSError, json.JSONDecodeError):
            is_ready = False

    return {
        "status": "ok" if is_ready else "degraded",
        "model_ready": is_ready,
        "model_version": version_str,
    }


@app.post("/predict-risk", response_model=PredictResponse, tags=["Prediction"])
def predict_failure_risk(payload: SensorPayload) -> dict[str, Any]:
    """Endpoint tính toán xác suất rủi ro hỏng máy và đưa ra khuyến nghị bảo trì."""
    try:
        model, config = get_model_and_config()
        feature_df = build_feature_row(payload, config["features"])
        failure_prob = float(model.predict_proba(feature_df)[0, 1])
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống mô hình chưa sẵn sàng. Vui lòng thực thi huấn luyện mô hình trước.",
        ) from err
    except (ValueError, KeyError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dữ liệu đầu vào không hợp lệ: {err!s}",
        ) from err

    threshold = float(config.get("threshold", 0.5))

    # Phân cấp rủi ro (Risk Tiering): HIGH, MEDIUM, LOW
    if failure_prob >= max(threshold, 0.7):
        risk_tier = "HIGH"
    elif failure_prob >= threshold:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    return {
        "failure_risk": failure_prob,
        "threshold": threshold,
        "risk_tier": risk_tier,
        "alert": failure_prob >= threshold,
        "reason_codes": generate_operational_reason_codes(payload),
        "model_version": config.get("version", "unknown"),
    }
