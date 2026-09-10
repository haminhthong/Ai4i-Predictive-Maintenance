"""API tối giản cho scoring snapshot và ranking batch."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import VALID_QUALITY_TYPES
from .inference import RiskInferenceService

LOGGER = logging.getLogger("ai4i.api")

app = FastAPI(
    title="AI4I Maintenance Risk Triage",
    description=(
        "Chấm điểm failure risk của operating snapshot hiện tại. "
        "API không dự báo RUL hoặc failure trong một horizon tương lai."
    ),
    version="1.0.0",
)


class SensorPayload(BaseModel):
    """Sáu biến sensor của một snapshot; `record_id` chỉ để truy vết dòng."""

    model_config = ConfigDict(extra="forbid")

    record_id: str | None = Field(default=None, min_length=1)
    product_quality_type: str = Field(default="M", description="L, M hoặc H")
    air_temperature_k: float = Field(..., gt=0, le=1000)
    process_temperature_k: float = Field(..., gt=0, le=1000)
    rotational_speed_rpm: float = Field(..., gt=0, le=100_000)
    torque_nm: float = Field(..., ge=0, le=10_000)
    tool_wear_min: float = Field(..., ge=0, le=1_000_000)

    @field_validator("product_quality_type")
    @classmethod
    def validate_quality_type(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in VALID_QUALITY_TYPES:
            raise ValueError("product_quality_type phải là L, M hoặc H.")
        return normalized


Decision = Literal["NO_ALERT", "REVIEW_REQUIRED"]


class ScoreResponse(BaseModel):
    record_id: str | None
    failure_risk: float = Field(..., ge=0, le=1)
    decision: Decision
    threshold: float = Field(..., ge=0, le=1)
    warnings: list[str]
    model: str


class RankRequest(BaseModel):
    snapshots: list[SensorPayload] = Field(..., min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=10_000)


class RankedScore(ScoreResponse):
    rank: int = Field(..., ge=1)


class RankResponse(BaseModel):
    items: list[RankedScore]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_ready: bool
    model: str | None


def _service_or_503() -> RiskInferenceService:
    service = RiskInferenceService.get_instance()
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Artifact model chưa sẵn sàng. Hãy chạy `python -m src.train`.",
        )
    return service


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> dict[str, Any]:
    """Kiểm tra API và artifact model."""
    service = RiskInferenceService.get_instance()
    return {
        "status": "ok" if service.is_ready else "degraded",
        "model_ready": service.is_ready,
        "model": service.metadata.get("model") if service.is_ready else None,
    }


@app.post("/score", response_model=ScoreResponse, tags=["Prediction"])
def score(payload: SensorPayload) -> dict[str, Any]:
    """Snapshot cảm biến -> risk đã hiệu chỉnh -> quyết định review."""
    service = _service_or_503()
    try:
        return service.predict(payload.model_dump(exclude_none=True))
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Snapshot không hợp lệ: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@app.post("/rank", response_model=RankResponse, tags=["Prediction"])
def rank(payload: RankRequest) -> dict[str, Any]:
    """Score batch snapshot và trả về các dòng theo risk giảm dần."""
    service = _service_or_503()
    try:
        items = service.rank(
            [snapshot.model_dump(exclude_none=True) for snapshot in payload.snapshots],
            top_k=payload.top_k,
        )
        return {"items": items}
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Batch snapshot không hợp lệ: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
