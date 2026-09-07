"""FastAPI RESTful Service cho Hệ thống Phân loại Rủi ro & Quyết định Bảo trì Máy móc (AI4I Risk Decision System).

Service cung cấp:
1. `GET /health/live`: Liveness probe.
2. `GET /health/ready`: Readiness probe kiểm tra model, contract, policy và chạy chu trình suy luận mẫu.
3. `GET /health`: Kiểm tra tổng quát (Tính tương thích ngược).
4. `POST /predict-risk`: Nhận dữ liệu cảm biến, chuyển tiếp tới Inference Engine và trả về cấu trúc 4 khối rõ ràng:
   - Prediction Block (Ước lượng rủi ro ML)
   - Reliability Block (Cảnh báo phân bố / Gate)
   - Decision Block (Hành động bảo trì)
   - Operational Context Block (Mã lý do vận hành & Ngữ cảnh đặc trưng)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import VALID_QUALITY_TYPES
from .inference import RiskInferenceService

LOGGER = logging.getLogger("ai_predictive_maintenance.api")

app = FastAPI(
    title="AI4I Machine Failure Risk Decision System",
    description=(
        "Production-oriented machine failure risk classification and maintenance decision service. "
        "Built with calibrated ensemble models, cost-sensitive frozen policies, distribution guardrails, "
        "and operational reason codes."
    ),
    version="2.1.0",
)


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------


class SensorPayload(BaseModel):
    """Schema dữ liệu đầu vào từ cảm biến thiết bị sản xuất."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    machine_type: str = Field(
        default="M",
        alias="Type",
        description="Chất lượng sản phẩm/máy móc: L (Low), M (Medium), H (High)",
    )
    air_temperature_k: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Nhiệt độ không khí xung quanh buồng máy (Kelvin K)",
    )
    process_temperature_k: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Nhiệt độ quá trình gia công vận hành (Kelvin K)",
    )
    rotational_speed_rpm: float = Field(
        ...,
        gt=0,
        le=100_000,
        description="Tốc độ quay của trục chính (Vòng/phút RPM)",
    )
    torque_nm: float = Field(
        ...,
        ge=0,
        le=10_000,
        description="Mô-men xoắn hoạt động (Newton mét Nm)",
    )
    tool_wear_min: float = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Thời gian độ mòn dụng cụ tích lũy (Phút min)",
    )

    @field_validator("machine_type")
    @classmethod
    def validate_machine_type(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in VALID_QUALITY_TYPES:
            raise ValueError(
                f"Phân loại máy (Type) phải thuộc một trong các giá trị: {sorted(VALID_QUALITY_TYPES)}"
            )
        return normalized

    @property
    def Type(self) -> str:
        """Hỗ trợ tương thích ngược với API cũ."""
        return self.machine_type


class PredictionBlock(BaseModel):
    """Khối dự báo rủi ro xác suất của mô hình ML."""

    failure_risk: float = Field(..., description="Xác suất dự báo máy bị hỏng [0.0 - 1.0]")
    model_version: str = Field(..., description="Phiên bản mô hình đang phục vụ")
    model_type: str = Field(..., description="Thuật toán mô hình")


class ReliabilityBlock(BaseModel):
    """Khối kiểm tra độ tin cậy và phân bố cảm biến (Reliability Gate)."""

    status: str = Field(..., description="Trạng thái tin cậy: 'NOMINAL' hoặc 'DEGRADED'")
    distribution_warning: bool = Field(
        ..., description="True nếu có cảm biến nằm ngoài dải huấn luyện P0.5 - P99.5"
    )
    warning_features: list[str] = Field(
        ..., description="Danh sách chi tiết các đặc trưng vi phạm dải phân bố"
    )


class DecisionBlock(BaseModel):
    """Khối quyết định bảo trì và khuyến nghị vận hành."""

    action: str = Field(
        ..., description="Hành động: 'NO_ALERT', 'REVIEW_REQUIRED', hoặc 'PRIORITY_REVIEW'"
    )
    alert_threshold: float = Field(..., description="Ngưỡng kích hoạt kiểm tra bảo trì")
    critical_threshold: float = Field(..., description="Ngưỡng kích hoạt bảo trì khẩn cấp")
    maintenance_alert: bool = Field(..., description="Cờ cảnh báo (True nếu action != NO_ALERT)")
    policy_version: str = Field(..., description="Phiên bản chính sách quyết định đang áp dụng")
    cost_scenario: str = Field(..., description="Kịch bản chi phí nghiệp vụ đã tối ưu")


class OperationalContextBlock(BaseModel):
    """Khối thông tin ngữ cảnh vận hành và giải thích đặc trưng."""

    reason_codes: list[str] = Field(
        ..., description="Mã lý do vận hành chuyên gia (Heuristic Rules độc lập)"
    )
    feature_context: list[dict[str, Any]] = Field(
        ..., description="Bảng ngữ cảnh giá trị các đặc trưng quan sát"
    )


class PredictRiskResponse(BaseModel):
    """Schema dữ liệu phản hồi cấu trúc 4 khối chuẩn mực của endpoint POST /predict-risk."""

    prediction: PredictionBlock
    reliability: ReliabilityBlock
    decision: DecisionBlock
    operational_context: OperationalContextBlock

    # Các trường phẳng ở cấp root phục vụ tương thích ngược (Backward Compatibility)
    failure_risk: float
    threshold: float
    risk_tier: str
    alert: bool
    reason_codes: list[str]
    model_version: str
    model_explanation: list[dict[str, Any]]


class HealthCheckResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str


class ReadinessCheckResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
    model_version: str
    policy_version: str
    test_cycle_passed: bool


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------


@app.get("/health/live", tags=["System"])
def liveness_check() -> dict[str, str]:
    """Endpoint liveness probe kiểm tra tiến trình service đang chạy."""
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadinessCheckResponse, tags=["System"])
def readiness_check() -> dict[str, Any]:
    """Endpoint readiness probe kiểm tra mô hình, chính sách và thực thi một chu trình test inference."""
    service = RiskInferenceService.get_instance()
    is_ready = service.is_loaded

    test_passed = False
    if is_ready:
        try:
            # Thực thi một test inference nhẹ
            test_sample = {
                "Type": "M",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 10.0,
            }
            res = service.predict(test_sample)
            test_passed = "prediction" in res
        except Exception as exc:
            LOGGER.error(f"Readiness test cycle failed: {exc}")
            test_passed = False

    overall_ready = is_ready and test_passed

    return {
        "ready": overall_ready,
        "checks": {
            "artifacts_loaded": is_ready,
            "inference_engine_ready": test_passed,
        },
        "model_version": service.manifest.get("model_version", "not_trained"),
        "policy_version": service.policy.get("policy_version", "not_configured"),
        "test_cycle_passed": test_passed,
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check() -> dict[str, Any]:
    """Endpoint tổng quát kiểm tra sức khỏe hệ thống."""
    service = RiskInferenceService.get_instance()
    is_ready = service.is_loaded
    return {
        "status": "ok" if is_ready else "degraded",
        "model_ready": is_ready,
        "model_version": service.manifest.get("model_version", "not_trained"),
    }


@app.post("/predict-risk", response_model=PredictRiskResponse, tags=["Prediction"])
def predict_machine_failure_risk(payload: SensorPayload) -> dict[str, Any]:
    """Dự báo xác suất rủi ro hỏng máy, kiểm tra gate phân bố và đưa ra quyết định bảo trì."""
    service = RiskInferenceService.get_instance()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống mô hình chưa sẵn sàng. Vui lòng hoàn tất huấn luyện mô hình.",
        )

    try:
        raw_dict = payload.model_dump(by_alias=True)
        return service.predict(raw_dict)
    except Exception as exc:
        LOGGER.error(f"Lỗi trong quá trình suy luận: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi xử lý dữ liệu cảm biến: {exc!s}",
        ) from exc


# ---------------------------------------------------------------------------
# HÀM BỔ TRỢ TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBILITY HELPERS)
# ---------------------------------------------------------------------------


def generate_operational_reason_codes(payload: SensorPayload) -> list[str]:
    """Hàm helper phục vụ tương thích ngược với các bài test và scripts cũ."""
    service = RiskInferenceService.get_instance()
    return service.extract_operational_reason_codes(payload.model_dump(by_alias=True))


def get_model_config_and_ranges() -> tuple[Any, dict[str, Any], dict[str, dict[str, float]]]:
    """Hàm helper phục vụ tương thích ngược với scripts cũ."""
    service = RiskInferenceService.get_instance()
    return service.model, service.manifest, service.reference_distribution
