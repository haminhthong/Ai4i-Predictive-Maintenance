"""FastAPI RESTful Service dự báo rủi ro hỏng hóc thiết bị máy móc sản xuất (Machine Failure Risk System).

Dịch vụ này cung cấp các endpoints chính:
1. `GET /health/live`: Liveness probe kiểm tra tiến trình đang chạy.
2. `GET /health/ready`: Readiness probe kiểm tra mô hình, tệp cấu hình và tính hợp lệ của artifact.
3. `GET /health`: Endpoint tổng hợp tính tương thích ngược.
4. `POST /predict-risk`: Nhận thông số cảm biến thời gian thực, tính toán xác suất hỏng máy,
   kiểm tra Out-Of-Distribution (OOD), phân cấp rủi ro (Risk Tiering), đưa ra quyết định cảnh báo bảo trì
   và trả về tách biệt mã lý do vận hành (Reason Codes) cùng giải thích mô hình (Model Explanations).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# Xác định đường dẫn tuyệt đối tới tệp mô hình, tệp cấu hình và khoảng phân bố đặc trưng
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models/config.json"
RANGES_PATH = PROJECT_ROOT / "models/feature_ranges.json"

# Khởi tạo ứng dụng FastAPI với đầy đủ metadata OpenAPI
app = FastAPI(
    title="Machine Failure Risk & Maintenance Decision Service",
    description="Dịch vụ AI dự báo xác suất rủi ro sự cố máy móc và đưa ra quyết định bảo trì tối ưu chi phí.",
    version="2.0.0",
)

# Biến toàn cục lưu cache mô hình, cấu hình và feature ranges (Lazy Loading Singleton Pattern)
_model: Any | None = None
_config: dict[str, Any] | None = None
_feature_ranges: dict[str, dict[str, float]] | None = None


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
    model_version: str = Field(..., json_schema_extra={"example": "ai4i-20260905-1a2b3c4"})


class ReadinessResponse(BaseModel):
    """Schema chi tiết của endpoint /health/ready."""

    ready: bool
    checks: dict[str, bool]
    model_version: str
    feature_contract_version: str


class PredictionDetails(BaseModel):
    """Chi tiết khối dự báo xác suất và OOD."""

    failure_probability: float
    model_version: str
    ood_warning: bool
    ood_features: list[str]


class DecisionDetails(BaseModel):
    """Chi tiết khối quyết định bảo trì và phân cấp rủi ro."""

    maintenance_alert: bool
    threshold: float
    risk_tier: str
    cost_scenario: str


class PredictResponse(BaseModel):
    """Schema dữ liệu phản hồi đầy đủ của endpoint /predict-risk."""

    failure_risk: float = Field(..., description="Xác suất dự báo máy bị hỏng [0.0 - 1.0]")
    threshold: float = Field(..., description="Ngưỡng quyết định phát cảnh báo tối ưu")
    risk_tier: str = Field(..., description="Mức độ rủi ro: HIGH, MEDIUM, LOW")
    alert: bool = Field(..., description="Cờ cảnh báo (True nếu rủi ro >= threshold)")
    reason_codes: list[str] = Field(..., description="Danh sách mã lý do vận hành chuyên gia")
    model_version: str = Field(..., description="Phiên bản mô hình đang sử dụng")
    prediction: PredictionDetails = Field(..., description="Chi tiết xác suất dự báo & OOD warning")
    decision: DecisionDetails = Field(..., description="Chi tiết quyết định bảo trì & phân cấp rủi ro")
    model_explanation: list[dict[str, Any]] = Field(..., description="Giải thích đặc trưng mô hình ML")


def get_model_config_and_ranges() -> tuple[Any, dict[str, Any], dict[str, dict[str, float]]]:
    """Tải và lưu cache mô hình ML, tệp cấu hình và feature ranges (Singleton lazy load)."""
    global _model, _config, _feature_ranges
    if _model is None or _config is None or _feature_ranges is None:
        if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
            raise FileNotFoundError(
                "Mô hình chưa được huấn luyện. Thiếu model.joblib hoặc config.json."
            )

        _model = joblib.load(MODEL_PATH)
        _config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

        if RANGES_PATH.exists():
            _feature_ranges = json.loads(RANGES_PATH.read_text(encoding="utf-8"))
        else:
            _feature_ranges = {}

    return _model, _config, _feature_ranges


def build_feature_row(payload: SensorPayload, feature_names: list[str]) -> pd.DataFrame:
    """Ánh xạ các trường từ SensorPayload sang định dạng Dataframe tương thích với mô hình đã huấn luyện.

    Tự động tính toán các đặc trưng vật lý bổ sung:
    - `temperature_delta = process_temperature - air_temperature`
    - `mechanical_power = torque * (rotational_speed * 2 * pi / 60)` (Watts)
    - `wear_load_interaction = tool_wear * torque`
    """
    angular_velocity = payload.rotational_speed_rpm * (2.0 * math.pi / 60.0)
    mech_power = payload.torque_nm * angular_velocity
    wear_load = payload.tool_wear_min * payload.torque_nm

    feature_map = {
        "type": payload.machine_type,
        "air temperature": payload.air_temperature_k,
        "process temperature": payload.process_temperature_k,
        "rotational speed": payload.rotational_speed_rpm,
        "torque": payload.torque_nm,
        "tool wear": payload.tool_wear_min,
        "temperature_delta": payload.process_temperature_k - payload.air_temperature_k,
        "mechanical_power": mech_power,
        "power_proxy": payload.rotational_speed_rpm * payload.torque_nm,
        "wear_load_interaction": wear_load,
        "strain_proxy": wear_load,
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


def check_out_of_distribution(
    feature_df: pd.DataFrame, feature_ranges: dict[str, dict[str, float]]
) -> tuple[bool, list[str]]:
    """Kiểm tra xem các giá trị đặc trưng có nằm ngoài khoảng phân bố P0.5 - P99.5 của tập Train hay không."""
    ood_features: list[str] = []
    if not feature_ranges:
        return False, []

    for col in feature_df.columns:
        if col in feature_ranges:
            val = float(feature_df[col].iloc[0])
            p0_5 = feature_ranges[col].get("p0_5", -float("inf"))
            p99_5 = feature_ranges[col].get("p99_5", float("inf"))
            if val < p0_5 or val > p99_5:
                ood_features.append(f"{col}={val:.2f} (phạm vi train: [{p0_5:.2f}, {p99_5:.2f}])")

    return len(ood_features) > 0, ood_features


def generate_operational_reason_codes(payload: SensorPayload) -> list[str]:
    """Sinh danh sách các mã lý do vận hành (Reason Codes) dựa trên quy tắc chuyên gia minh bạch.

    Lưu ý: Các mã lý do này là quy tắc vận hành độc lập với mô hình ML, dùng cho kỹ sư nhà máy.
    """
    reasons: list[str] = []

    if payload.tool_wear_min >= 200:
        reasons.append("TOOL_WEAR_HIGH")

    if payload.torque_nm >= 55:
        reasons.append("TORQUE_HIGH")

    if payload.rotational_speed_rpm <= 1300:
        reasons.append("ROTATIONAL_SPEED_LOW")

    temperature_delta = payload.process_temperature_k - payload.air_temperature_k
    if temperature_delta <= 8.6:
        reasons.append("TEMPERATURE_DELTA_LOW")

    return reasons


def generate_model_explanations(model: Any, feature_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Giải thích đóng góp của các đặc trưng tới dự báo xác suất hỏng máy."""
    explanations: list[dict[str, Any]] = []
    try:
        # Kiểm tra nếu mô hình có pipeline với logistic regression
        clf = getattr(model, "estimator", model)
        if hasattr(clf, "named_steps") and "classifier" in clf.named_steps:
            classifier = clf.named_steps["classifier"]
            preprocessor = clf.named_steps["preprocessor"]
            if hasattr(classifier, "coef_"):
                # Mô hình Logistic Regression: Tính contribution = scaled_value * coef
                X_transformed = preprocessor.transform(feature_df)
                coefs = classifier.coef_[0]
                feature_names = feature_df.columns
                for idx, col in enumerate(feature_names[: len(coefs)]):
                    val = float(X_transformed[0, idx]) if idx < X_transformed.shape[1] else 0.0
                    contrib = float(val * coefs[idx])
                    explanations.append(
                        {
                            "feature": col,
                            "contribution": round(contrib, 4),
                            "direction": "positive_risk" if contrib > 0 else "negative_risk",
                        }
                    )
                explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                return explanations

        # Trường hợp mô hình cây (RandomForest, HistGradientBoosting)
        for col in feature_df.columns:
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                explanations.append(
                    {
                        "feature": col,
                        "value": float(feature_df[col].iloc[0]),
                        "type": "observed_feature",
                    }
                )
    except Exception:
        pass

    return explanations


@app.get("/health/live", tags=["System"])
def liveness_check() -> dict[str, str]:
    """Endpoint liveness probe kiểm tra tiến trình service đang chạy."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadinessResponse, tags=["System"])
def readiness_check() -> dict[str, Any]:
    """Endpoint readiness probe kiểm tra tính sẵn sàng của mô hình, tệp cấu hình và hợp đồng đặc trưng."""
    model_exists = MODEL_PATH.exists()
    config_exists = CONFIG_PATH.exists()
    ranges_exists = RANGES_PATH.exists()

    model_version = "not_trained"
    contract_version = "unknown"

    if config_exists:
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            model_version = cfg.get("version", "unknown")
            contract_version = cfg.get("feature_contract_version", "unknown")
        except Exception:
            config_exists = False

    ready = model_exists and config_exists

    return {
        "ready": ready,
        "checks": {
            "model_file": model_exists,
            "config_file": config_exists,
            "feature_ranges_file": ranges_exists,
        },
        "model_version": model_version,
        "feature_contract_version": contract_version,
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> dict[str, Any]:
    """Endpoint kiểm tra sức khỏe hệ thống (Tính tương thích ngược)."""
    ready_data = readiness_check()
    is_ready = ready_data["ready"]
    return {
        "status": "ok" if is_ready else "degraded",
        "model_ready": is_ready,
        "model_version": ready_data["model_version"],
    }


@app.post("/predict-risk", response_model=PredictResponse, tags=["Prediction"])
def predict_failure_risk(payload: SensorPayload) -> dict[str, Any]:
    """Endpoint dự báo xác suất rủi ro hỏng máy và đưa ra khuyến nghị quyết định bảo trì."""
    try:
        model, config, feature_ranges = get_model_config_and_ranges()
        feature_df = build_feature_row(payload, config["features"])
        failure_prob = float(model.predict_proba(feature_df)[0, 1])
        is_ood, ood_features = check_out_of_distribution(feature_df, feature_ranges)
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
    fn_cost = float(config.get("false_negative_cost", 5.0))
    fp_cost = float(config.get("false_positive_cost", 1.0))
    model_ver = config.get("version", "unknown")

    # Phân cấp rủi ro (Risk Tiering): HIGH, MEDIUM, LOW
    if failure_prob >= max(threshold, 0.7):
        risk_tier = "HIGH"
    elif failure_prob >= threshold:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    is_alert = failure_prob >= threshold
    reason_codes = generate_operational_reason_codes(payload)
    model_explanations = generate_model_explanations(model, feature_df)

    return {
        "failure_risk": failure_prob,
        "threshold": threshold,
        "risk_tier": risk_tier,
        "alert": is_alert,
        "reason_codes": reason_codes,
        "model_version": model_ver,
        "prediction": {
            "failure_probability": failure_prob,
            "model_version": model_ver,
            "ood_warning": is_ood,
            "ood_features": ood_features,
        },
        "decision": {
            "maintenance_alert": is_alert,
            "threshold": threshold,
            "risk_tier": risk_tier,
            "cost_scenario": f"FN{fn_cost:.0f}_FP{fp_cost:.0f}",
        },
        "model_explanation": model_explanations,
    }

