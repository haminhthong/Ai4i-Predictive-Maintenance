"""Suy luận risk snapshot và tạo risk event cho maintenance queue."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifact import find_latest_release, verify_release_bundle
from .contracts import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_CONTRACT,
    TARGET_COLUMN,
)
from .features import build_canonical_features, canonicalize_raw_dataframe
from .monitoring import assess_distribution_guardrails
from .policy import build_maintenance_queue, map_decision_action
from .storage import SQLiteRiskEventStore
from .utils import normalize_event_time

LOGGER = logging.getLogger("ai_condition_risk.inference")
ARTIFACTS_DIR = Path("artifacts/champion")
MODELS_DIR = Path("models")


class RiskInferenceService:
    """Tải một release và chấm điểm trạng thái vận hành hiện tại."""

    _instance: RiskInferenceService | None = None

    def __init__(self) -> None:
        self.model: Any = None
        self.manifest: dict[str, Any] = {}
        self.policy: dict[str, Any] = {}
        self.contract: dict[str, Any] = {}
        self.reference_distribution: dict[str, dict[str, float]] = {}
        self.bundle_dir: Path | None = None
        self.artifact_checks: dict[str, Any] = {}
        self.is_loaded = False
        self.risk_events: list[dict[str, Any]] = []
        self.event_store = SQLiteRiskEventStore()
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> RiskInferenceService:
        """Lấy service dùng chung trong tiến trình API."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        """Readiness chỉ đúng khi model đã tải và release integrity hợp lệ."""
        return self.is_loaded and bool(self.artifact_checks.get("ready", False))

    def _load_release(self, bundle_dir: Path) -> None:
        checks = verify_release_bundle(bundle_dir)
        self.artifact_checks = {
            "bundle_exists": bool(checks.get("bundle_exists")),
            "hashes_match": bool(checks.get("hashes_match")),
            "required_files": not bool(checks.get("missing_files")),
            "feature_contract_exact": False,
            "reference_distribution_exists": False,
            "ready": False,
        }
        if not checks.get("hashes_match"):
            self.artifact_checks["error"] = checks.get("error", "Hash release không khớp.")
            return

        self.bundle_dir = bundle_dir
        self.manifest = checks["manifest"]
        self.model = joblib.load(bundle_dir / "model.joblib")
        self.policy = json.loads((bundle_dir / "decision_policy.json").read_text(encoding="utf-8"))
        self.contract = json.loads((bundle_dir / "feature_contract.json").read_text(encoding="utf-8"))
        self.reference_distribution = json.loads(
            (bundle_dir / "reference_distribution.json").read_text(encoding="utf-8")
        )
        self.artifact_checks["feature_contract_exact"] = self.contract.get("features") == list(
            MODEL_FEATURE_CONTRACT
        )
        self.artifact_checks["reference_distribution_exists"] = bool(self.reference_distribution)
        self.artifact_checks["ready"] = all(
            self.artifact_checks[key]
            for key in (
                "bundle_exists",
                "hashes_match",
                "required_files",
                "feature_contract_exact",
                "reference_distribution_exists",
            )
        )

    def _load_legacy_artifacts(self) -> None:
        """Đọc artifact cũ trong giai đoạn chuyển đổi, không dùng làm release chuẩn."""
        model_file = ARTIFACTS_DIR / "model.joblib"
        if not model_file.exists():
            model_file = MODELS_DIR / "model.joblib"
        if not model_file.exists():
            return

        self.model = joblib.load(model_file)
        manifest_file = ARTIFACTS_DIR / "model_manifest.json"
        config_file = MODELS_DIR / "config.json"
        if manifest_file.exists():
            self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        elif config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            self.manifest = {
                "model_version": config.get("version", "unknown"),
                "model_type": config.get("selected_model", "unknown"),
            }

        policy_file = ARTIFACTS_DIR / "decision_policy.json"
        if policy_file.exists():
            self.policy = json.loads(policy_file.read_text(encoding="utf-8"))
        elif config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            self.policy = config.get("decision_policy", {
                "policy_version": "maintenance-policy-v2",
                "primary_alert_threshold": config.get("threshold", 0.5),
                "critical_threshold": config.get("critical_threshold", 0.75),
            })

        contract_file = ARTIFACTS_DIR / "feature_contract.json"
        self.contract = (
            json.loads(contract_file.read_text(encoding="utf-8"))
            if contract_file.exists()
            else {"features": list(MODEL_FEATURE_CONTRACT)}
        )
        distribution_file = ARTIFACTS_DIR / "reference_distribution.json"
        if not distribution_file.exists():
            distribution_file = MODELS_DIR / "feature_ranges.json"
        if distribution_file.exists():
            self.reference_distribution = json.loads(distribution_file.read_text(encoding="utf-8"))

        # Legacy artifact được phép chạy để migration, nhưng không được coi là ready.
        self.artifact_checks = {
            "bundle_exists": False,
            "hashes_match": False,
            "required_files": False,
            "feature_contract_exact": self.contract.get("features") == list(MODEL_FEATURE_CONTRACT),
            "reference_distribution_exists": bool(self.reference_distribution),
            "ready": False,
            "warning": "Đang dùng artifact legacy; hãy tạo release bundle mới.",
        }

    def load_artifacts(self) -> None:
        """Ưu tiên release bundle; chỉ fallback legacy để hỗ trợ migration."""
        self.is_loaded = False
        latest_release = find_latest_release()
        try:
            if latest_release is not None:
                self._load_release(latest_release)
            else:
                self._load_legacy_artifacts()
            self.is_loaded = self.model is not None
        except Exception as exc:
            self.is_loaded = False
            self.artifact_checks = {"ready": False, "error": str(exc)}
            LOGGER.exception("Không thể nạp artifact: %s", exc)

        if self.is_loaded:
            LOGGER.info(
                "Đã nạp model=%s version=%s ready=%s",
                self.manifest.get("model_type", "unknown"),
                self.manifest.get("model_version", "unknown"),
                self.is_ready,
            )

    def extract_operational_reason_codes(self, raw_input: dict[str, Any]) -> list[str]:
        """Sinh điều kiện quan sát theo heuristic, không phải model attribution."""
        def number(*keys: str, default: float) -> float:
            for key in keys:
                value = raw_input.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        break
            return default

        tool_wear = number("tool_wear_min", "Tool wear", default=0.0)
        torque = number("torque_nm", "Torque", default=0.0)
        speed = number("rotational_speed_rpm", "Rotational speed", default=1500.0)
        process_temperature = number(
            "process_temperature_k", "Process temperature", default=310.0
        )
        air_temperature = number("air_temperature_k", "Air temperature", default=300.0)

        observed: list[str] = []
        if tool_wear >= 200:
            observed.append("TOOL_WEAR_HIGH")
        if torque >= 55:
            observed.append("TORQUE_HIGH")
        if speed <= 1300:
            observed.append("ROTATIONAL_SPEED_LOW")
        if process_temperature - air_temperature <= 8.6:
            observed.append("TEMPERATURE_DELTA_LOW")
        return observed

    def extract_feature_context(self, feature_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Trả về giá trị quan sát, không gắn nhãn là giải thích mô hình."""
        context: list[dict[str, Any]] = []
        for column in feature_df.columns:
            value = feature_df[column].iloc[0]
            context.append(
                {
                    "feature": column,
                    "value": round(float(value), 2)
                    if isinstance(value, (int, float, np.number))
                    else str(value),
                    "type": "observed_feature",
                }
            )
        return context

    def predict(self, raw_payload: dict[str, Any], persist_event: bool = True) -> dict[str, Any]:
        """Chấm điểm snapshot: contract -> features -> risk -> reliability -> triage."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Mô hình chưa sẵn sàng hoạt động.")

        canonical_payload = canonicalize_raw_dataframe(pd.DataFrame([raw_payload]))
        forbidden_columns = set(IDENTIFIER_COLUMNS) | {TARGET_COLUMN} | set(FAILURE_MODE_COLUMNS)
        leaked_columns = sorted(forbidden_columns & set(canonical_payload.columns))
        if leaked_columns:
            raise ValueError(
                f"Inference payload chứa cột bị cấm hoặc target-derived: {leaked_columns}"
            )

        features = build_canonical_features(
            canonical_payload, expected_features=MODEL_FEATURE_CONTRACT
        )
        risk_score = float(self.model.predict_proba(features)[0, 1])
        reliability_status, warning_features = assess_distribution_guardrails(
            features, self.reference_distribution
        )
        distribution_warning = reliability_status == "DEGRADED"

        alert_threshold = float(self.policy.get("primary_alert_threshold", 0.5))
        critical_threshold = float(self.policy.get("critical_threshold", 0.75))
        if reliability_status == "UNAVAILABLE":
            action = "REVIEW_REQUIRED"
            queue_eligible = False
        else:
            action = map_decision_action(
                risk_score,
                alert_threshold=alert_threshold,
                critical_threshold=critical_threshold,
                distribution_warning=distribution_warning,
            )
            queue_eligible = True

        observed_conditions = self.extract_operational_reason_codes(raw_payload)
        feature_context = self.extract_feature_context(features)
        event_id = str(raw_payload.get("event_id") or f"evt_{uuid.uuid4().hex[:12]}")
        asset_id = str(raw_payload.get("asset_id") or "UNKNOWN_ASSET")
        event_time = normalize_event_time(raw_payload.get("event_time")) or datetime.now(
            timezone.utc,  # noqa: UP017 - tương thích Python 3.10
        ).isoformat().replace("+00:00", "Z")
        model_version = str(self.manifest.get("model_version", "unknown"))
        policy_version = str(self.policy.get("policy_version", "unknown"))
        priority = {
            "PRIORITY_REVIEW": "HIGH",
            "REVIEW_REQUIRED": "MEDIUM",
            "NO_ALERT": "LOW",
        }[action]
        event = {
            "event_id": event_id,
            "asset_id": asset_id,
            "event_time": event_time,
            "shift": raw_payload.get("shift"),
            "risk_score": round(risk_score, 6),
            "action": action,
            "reliability_status": reliability_status,
            "queue_eligible": queue_eligible,
            "observed_conditions": observed_conditions,
            "model_version": model_version,
            "policy_version": policy_version,
        }
        if persist_event:
            self.risk_events.append(event)
            try:
                self.event_store.record_event(event, raw_payload)
            except Exception:
                # Lưu trữ không được làm thay đổi điểm risk; lỗi sẽ được monitoring ghi nhận.
                LOGGER.exception("Không thể ghi risk event vào SQLite.")

        return {
            "event_id": event_id,
            "asset_id": asset_id,
            "event_time": event_time,
            "risk": {
                "snapshot_failure_risk": round(risk_score, 4),
                "model_version": model_version,
            },
            "prediction": {
                "snapshot_failure_risk": round(risk_score, 4),
                "failure_risk": round(risk_score, 4),
                "model_version": model_version,
                "model_type": self.manifest.get("model_type", "unknown"),
            },
            "reliability": {
                "status": reliability_status,
                "distribution_warning": distribution_warning,
                "warning_features": warning_features,
            },
            "triage": {
                "priority": priority,
                "queue_eligible": queue_eligible,
                "policy_version": policy_version,
            },
            "decision": {
                "action": action,
                "alert_threshold": alert_threshold,
                "critical_threshold": critical_threshold,
                "maintenance_alert": action != "NO_ALERT",
                "policy_version": policy_version,
                "cost_scenario": self._cost_scenario(),
            },
            "observed_conditions": observed_conditions,
            "operational_context": {
                "observed_conditions": observed_conditions,
                "reason_codes": observed_conditions,
                "feature_context": feature_context,
                "note": "Điều kiện heuristic quan sát được, không phải model attribution.",
            },
            # Các trường phẳng giữ tương thích với client cũ.
            "failure_risk": round(risk_score, 4),
            "threshold": alert_threshold,
            "risk_tier": priority,
            "alert": action != "NO_ALERT",
            "reason_codes": observed_conditions,
            "model_version": model_version,
            "model_explanation": feature_context,
        }

    def _cost_scenario(self) -> str:
        costs = self.policy.get("cost_weights", {})
        return f"FN{float(costs.get('false_negative', 5.0)):.0f}_FP{float(costs.get('false_positive', 1.0)):.0f}"

    def build_queue(self, capacity: int, shift: str | None = None) -> list[dict[str, Any]]:
        """Xếp queue từ các risk event đã ghi nhận trong tiến trình hiện tại."""
        persisted_events = self.event_store.list_risk_events()
        if shift is not None:
            persisted_events = [
                event for event in persisted_events if event.get("shift") == shift
            ]
        return build_maintenance_queue(persisted_events, capacity=capacity)

    def record_review(self, review: dict[str, Any]) -> None:
        """Ghi review kỹ thuật viên mà không thay đổi model trong runtime."""
        self.event_store.record_review(**review)
