"""FastAPI cho hệ thống Condition-Based Maintenance Risk Triage AI4I.

Service cung cấp:
1. `GET /health/live`: Liveness probe.
2. `GET /health/ready`: Readiness probe kiểm tra model, contract, policy và chạy chu trình suy luận mẫu.
3. `GET /health`: Kiểm tra tổng quát (Tính tương thích ngược).
4. `POST /score`: Chấm điểm snapshot hiện tại và trả về risk/triage/reliability.
5. `POST /maintenance/queue/build`: Dựng queue top-K theo capacity.
6. `POST /maintenance/reviews`: Lưu feedback kỹ thuật viên cho offline QA.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .contracts import VALID_QUALITY_TYPES
from .inference import RiskInferenceService

LOGGER = logging.getLogger("ai_predictive_maintenance.api")

app = FastAPI(
    title="AI4I Condition-Based Maintenance Risk Triage",
    description=(
        "Chấm điểm rủi ro của operating snapshot hiện tại và xếp hàng bảo trì theo capacity. "
        "Điểm risk không hàm ý dự báo thời điểm hỏng trong tương lai."
    ),
    version="3.0.0",
)


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------


class SensorPayload(BaseModel):
    """Event contract: metadata runtime tách khỏi sensor features."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    event_id: str | None = Field(default=None, min_length=1)
    asset_id: str | None = Field(default=None, min_length=1)
    event_time: str | None = Field(default=None, description="ISO-8601 UTC")
    line_id: str | None = Field(default=None)
    sensor_source: str | None = Field(default=None)
    shift: str | None = Field(default=None, min_length=1)
    product_quality_type: str = Field(
        default="M",
        validation_alias=AliasChoices(
            "product_quality_type", "product_type", "machine_type", "Type"
        ),
        description="Chất lượng sản phẩm: L (Low), M (Medium), H (High); không phải machine identity",
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

    @field_validator("product_quality_type")
    @classmethod
    def validate_product_quality_type(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in VALID_QUALITY_TYPES:
            raise ValueError(
                f"product_quality_type phải thuộc một trong các giá trị: {sorted(VALID_QUALITY_TYPES)}"
            )
        return normalized

    @field_validator("event_time")
    @classmethod
    def validate_event_time(cls, value: str | None) -> str | None:
        """Chỉ nhận event time dạng ISO-8601 nếu metadata được gửi lên."""
        if value is None:
            return value
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("event_time phải là chuỗi ISO-8601 hợp lệ.") from exc
        return value

    @property
    def machine_type(self) -> str:
        """Alias đọc tương thích ngược; code mới phải dùng product_quality_type."""
        return self.product_quality_type

    @property
    def Type(self) -> str:
        """Alias đọc tương thích ngược với payload AI4I cũ."""
        return self.product_quality_type


class PredictionBlock(BaseModel):
    """Điểm rủi ro của snapshot hiện tại, không có future horizon."""

    snapshot_failure_risk: float = Field(..., ge=0.0, le=1.0)
    failure_risk: float = Field(
        ..., description="Điểm rủi ro gắn với operating snapshot hiện tại [0.0 - 1.0]"
    )
    model_version: str = Field(..., description="Phiên bản mô hình đang phục vụ")
    model_type: str = Field(..., description="Thuật toán mô hình")


class ReliabilityBlock(BaseModel):
    """Khối kiểm tra độ tin cậy và phân bố cảm biến (Reliability Gate)."""

    status: str = Field(..., description="NOMINAL, DEGRADED hoặc UNAVAILABLE")
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
    critical_threshold: float = Field(..., description="Ngưỡng đưa event vào priority review")
    maintenance_alert: bool = Field(..., description="Cờ cảnh báo (True nếu action != NO_ALERT)")
    policy_version: str = Field(..., description="Phiên bản chính sách quyết định đang áp dụng")
    cost_scenario: str = Field(..., description="Kịch bản chi phí nghiệp vụ đã tối ưu")


class RiskBlock(BaseModel):
    """Tên contract mới cho điểm risk snapshot."""

    snapshot_failure_risk: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class TriageBlock(BaseModel):
    """Kết quả triage và khả năng đưa vào queue."""

    priority: str
    queue_eligible: bool
    policy_version: str


class OperationalContextBlock(BaseModel):
    """Khối thông tin ngữ cảnh vận hành và giải thích đặc trưng."""

    reason_codes: list[str] = Field(
        ..., description="Điều kiện heuristic quan sát được, không phải model attribution"
    )
    feature_context: list[dict[str, Any]] = Field(
        ..., description="Bảng ngữ cảnh giá trị các đặc trưng quan sát"
    )


class PredictRiskResponse(BaseModel):
    """Schema response canonical của endpoint chấm điểm risk snapshot."""

    event_id: str
    asset_id: str
    event_time: str
    risk: RiskBlock
    prediction: PredictionBlock
    reliability: ReliabilityBlock
    triage: TriageBlock
    decision: DecisionBlock
    operational_context: OperationalContextBlock
    observed_conditions: list[str]

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
    checks: dict[str, Any]
    model_version: str
    policy_version: str
    test_cycle_passed: bool


class QueueBuildRequest(BaseModel):
    """Yêu cầu dựng queue trong một scheduling window."""

    shift: str = Field(..., min_length=1)
    capacity: int = Field(..., ge=0, le=100_000)


class QueueItem(BaseModel):
    rank: int
    asset_id: str
    event_id: str
    event_time: str
    risk_score: float
    action: str
    observed_conditions: list[str]
    model_version: str


class QueueBuildResponse(BaseModel):
    shift: str
    capacity: int
    queue_size: int
    priority_override_count: int
    items: list[QueueItem]


class MaintenanceReviewRequest(BaseModel):
    """Kết quả technician review được lưu để QA/retrain offline."""

    event_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    technician_action: str = Field(..., min_length=1)
    confirmed_issue: bool
    failure_mode: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    reviewed_at: str | None = Field(default=None)

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str | None) -> str | None:
        """Kiểm tra thời điểm review nếu client gửi lên."""
        if value is None:
            return value
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at phải là chuỗi ISO-8601 hợp lệ.") from exc
        return value


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
    is_ready = service.is_ready

    test_passed = False
    if is_ready:
        try:
            # Thực thi một test inference nhẹ
            test_sample = {
                "asset_id": "READINESS_SAMPLE",
                "product_quality_type": "M",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 10.0,
            }
            res = service.predict(test_sample)
            test_passed = "risk" in res and res["reliability"]["status"] != "UNAVAILABLE"
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            LOGGER.error(f"Readiness test cycle failed: {exc}")
            test_passed = False

    overall_ready = is_ready and test_passed

    return {
        "ready": overall_ready,
        "checks": {
            "artifacts_loaded": service.is_loaded,
            **service.artifact_checks,
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
    is_ready = service.is_ready
    return {
        "status": "ok" if is_ready else "degraded",
        "model_ready": is_ready,
        "model_version": service.manifest.get("model_version", "not_trained"),
    }


@app.post("/predict-risk", response_model=PredictRiskResponse, tags=["Prediction"])
def predict_machine_failure_risk(payload: SensorPayload) -> dict[str, Any]:
    """Chấm điểm risk của snapshot hiện tại và trả về triage."""
    service = RiskInferenceService.get_instance()
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống mô hình chưa sẵn sàng. Vui lòng hoàn tất huấn luyện mô hình.",
        )

    try:
        raw_dict = payload.model_dump(by_alias=True, exclude_none=True)
        return service.predict(raw_dict)
    except Exception as exc:
        LOGGER.error(f"Lỗi trong quá trình suy luận: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi xử lý dữ liệu cảm biến: {exc!s}",
        ) from exc


@app.post("/score", response_model=PredictRiskResponse, tags=["Prediction"])
def score_snapshot(payload: SensorPayload) -> dict[str, Any]:
    """Endpoint canonical: sensor snapshot -> condition failure risk score."""
    return predict_machine_failure_risk(payload)


@app.post("/maintenance/queue/build", response_model=QueueBuildResponse, tags=["Maintenance"])
def build_maintenance_queue_endpoint(request: QueueBuildRequest) -> dict[str, Any]:
    """Xếp event mới nhất theo asset và lấy top-K theo capacity."""
    service = RiskInferenceService.get_instance()
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Release model chưa sẵn sàng.",
        )
    items = service.build_queue(request.capacity, shift=request.shift)
    response_items = [
        {
            "rank": item["rank"],
            "asset_id": item["asset_id"],
            "event_id": item["event_id"],
            "event_time": item["event_time"],
            "risk_score": item["risk_score"],
            "action": item["action"],
            "observed_conditions": item.get("observed_conditions", []),
            "model_version": item.get("model_version", "unknown"),
        }
        for item in items
    ]
    return {
        "shift": request.shift,
        "capacity": request.capacity,
        "queue_size": len(response_items),
        "priority_override_count": sum(
            item["action"] == "PRIORITY_REVIEW" for item in response_items
        ),
        "items": response_items,
    }


@app.post("/maintenance/reviews", tags=["Maintenance"])
def record_maintenance_review(review: MaintenanceReviewRequest) -> dict[str, str]:
    """Lưu outcome của kỹ thuật viên; không retrain model trực tiếp."""
    service = RiskInferenceService.get_instance()
    try:
        payload = review.model_dump(exclude_none=True)
        payload.setdefault("reviewed_at", datetime.now(UTC).isoformat().replace("+00:00", "Z"))
        service.record_review(payload)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return {"status": "stored", "event_id": review.event_id}


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
