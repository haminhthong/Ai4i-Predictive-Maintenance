"""Động cơ suy luận (Inference Engine) cho hệ thống Machine Failure Risk Decision System.

Đảm bảo:
1. Tách biệt hoàn toàn tầng logic ML và tầng Web API (FastAPI).
2. Xử lý thống nhất qua Shared Feature Builder (triệt tiêu Train-Serving Skew).
3. Đánh giá Reliability Gate qua Distribution Range Guardrail.
4. Áp dụng Frozen Decision Policy xác định hành động (NO_ALERT, REVIEW_REQUIRED, PRIORITY_REVIEW).
5. Phân tách rõ ràng giữa Operational Reason Codes (mã lý do vận hành chuyên gia) và Feature Context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .contracts import MODEL_FEATURE_CONTRACT
from .features import build_canonical_features
from .monitoring import check_distribution_guardrails
from .policy import map_decision_action

LOGGER = logging.getLogger("ai_predictive_maintenance.inference")

ARTIFACTS_DIR = Path("artifacts/champion")
MODELS_DIR = Path("models")


class RiskInferenceService:
    """Service thực hiện suy luận rủi ro hỏng máy và ra quyết định bảo trì."""

    _instance: RiskInferenceService | None = None

    def __init__(self) -> None:
        self.model: Any = None
        self.manifest: dict[str, Any] = {}
        self.policy: dict[str, Any] = {}
        self.contract: dict[str, Any] = {}
        self.reference_distribution: dict[str, dict[str, float]] = {}
        self.is_loaded: bool = False
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> RiskInferenceService:
        """Lấy thể hiện duy nhất (Singleton) của service."""
        if cls._instance is None:
            cls._instance = RiskInferenceService()
        return cls._instance

    def load_artifacts(self) -> None:
        """Tải các artifact đã đóng băng từ artifacts/champion/ hoặc models/."""
        # 1. Tìm đường dẫn mô hình
        model_file = ARTIFACTS_DIR / "model.joblib"
        if not model_file.exists():
            model_file = MODELS_DIR / "model.joblib"

        if not model_file.exists():
            LOGGER.warning("Không tìm thấy model.joblib. Service ở trạng thái chưa sẵn sàng.")
            self.is_loaded = False
            return

        self.model = joblib.load(model_file)

        # 2. Tìm model manifest
        manifest_file = ARTIFACTS_DIR / "model_manifest.json"
        if manifest_file.exists():
            self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        else:
            config_file = MODELS_DIR / "config.json"
            if config_file.exists():
                cfg = json.loads(config_file.read_text(encoding="utf-8"))
                self.manifest = {
                    "model_version": cfg.get("version", "unknown"),
                    "model_type": cfg.get("selected_model", "unknown"),
                    "feature_contract_version": cfg.get("feature_contract_version", "unknown"),
                }

        # 3. Tìm decision policy
        policy_file = ARTIFACTS_DIR / "decision_policy.json"
        if policy_file.exists():
            self.policy = json.loads(policy_file.read_text(encoding="utf-8"))
        else:
            config_file = MODELS_DIR / "config.json"
            if config_file.exists():
                cfg = json.loads(config_file.read_text(encoding="utf-8"))
                self.policy = {
                    "policy_version": "maintenance-policy-v2",
                    "primary_alert_threshold": cfg.get("threshold", 0.3574),
                    "critical_threshold": cfg.get("critical_threshold", 0.75),
                    "cost_weights": {
                        "false_negative": cfg.get("false_negative_cost", 5.0),
                        "false_positive": cfg.get("false_positive_cost", 1.0),
                    },
                }

        # 4. Tìm feature contract
        contract_file = ARTIFACTS_DIR / "feature_contract.json"
        if contract_file.exists():
            self.contract = json.loads(contract_file.read_text(encoding="utf-8"))
        else:
            self.contract = {
                "contract_version": "ai4i-canonical-v2",
                "features": list(MODEL_FEATURE_CONTRACT),
            }

        # 5. Tìm reference distribution
        dist_file = ARTIFACTS_DIR / "reference_distribution.json"
        if not dist_file.exists():
            dist_file = MODELS_DIR / "feature_ranges.json"

        if dist_file.exists():
            self.reference_distribution = json.loads(dist_file.read_text(encoding="utf-8"))
        else:
            self.reference_distribution = {}

        self.is_loaded = True
        LOGGER.info(
            f"Đã nạp thành công artifacts: Model '{self.manifest.get('model_type')}' | "
            f"Version '{self.manifest.get('model_version')}' | "
            f"Alert Threshold {self.policy.get('primary_alert_threshold', 0.5):.4f}"
        )

    def extract_operational_reason_codes(self, raw_input: dict[str, Any]) -> list[str]:
        """Sinh mã lý do vận hành chuyên gia (Heuristic Rules độc lập với ML)."""
        reasons: list[str] = []

        def get_val(key_canonical: str, key_legacy: str, default: float) -> float:
            v = raw_input.get(key_canonical)
            if v is None:
                v = raw_input.get(key_legacy)
            try:
                return float(v) if v is not None else default
            except (ValueError, TypeError):
                return default

        tool_wear = get_val("tool_wear_min", "Tool wear", 0.0)
        torque = get_val("torque_nm", "Torque", 0.0)
        speed = get_val("rotational_speed_rpm", "Rotational speed", 1500.0)
        proc_temp = get_val("process_temperature_k", "Process temperature", 310.0)
        air_temp = get_val("air_temperature_k", "Air temperature", 300.0)

        if tool_wear >= 200.0:
            reasons.append("TOOL_WEAR_HIGH")
        if torque >= 55.0:
            reasons.append("TORQUE_HIGH")
        if speed <= 1300.0:
            reasons.append("ROTATIONAL_SPEED_LOW")
        if (proc_temp - air_temp) <= 8.6:
            reasons.append("TEMPERATURE_DELTA_LOW")

        return reasons

    def extract_feature_context(self, feature_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Trả về ngữ cảnh giá trị đặc trưng quan sát (Feature Context).

        LƯU Ý: Với mô hình Random Forest / Ensembles, đây là Input Feature Context quan sát được,
        không phải là Attribution / Explanation xấp xỉ giả lập.
        """
        context_items: list[dict[str, Any]] = []
        for col in feature_df.columns:
            val = feature_df[col].iloc[0]
            context_items.append(
                {
                    "feature": col,
                    "value": round(float(val), 2) if isinstance(val, (int, float, np.number)) else str(val),
                    "type": "observed_feature",
                }
            )
        return context_items

    def predict(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Thực hiện chu trình suy luận đầy đủ: Input -> Feature -> Model -> Gate -> Policy -> Output."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "Mô hình chưa sẵn sàng hoạt động. Vui lòng kiểm tra artifacts đã được huấn luyện."
            )

        # 1. Chuyển đổi qua Shared Feature Builder (chống Skew)
        input_df = pd.DataFrame([raw_payload])
        features_df = build_canonical_features(input_df, expected_features=MODEL_FEATURE_CONTRACT)

        # 2. Tính toán xác suất rủi ro hỏng máy (Calibrated Failure Risk)
        failure_prob = float(self.model.predict_proba(features_df)[0, 1])

        # 3. Đánh giá Reliability Gate (Distribution Range Guardrail)
        has_warning, warning_features = check_distribution_guardrails(
            features_df, self.reference_distribution
        )
        reliability_status = "DEGRADED" if has_warning else "NOMINAL"

        # 4. Áp dụng Frozen Decision Policy
        alert_thresh = float(self.policy.get("primary_alert_threshold", 0.3574))
        critical_thresh = float(self.policy.get("critical_threshold", 0.75))
        cost_weights = self.policy.get("cost_weights", {"false_negative": 5.0, "false_positive": 1.0})
        fn_w = float(cost_weights.get("false_negative", 5.0))
        fp_w = float(cost_weights.get("false_positive", 1.0))

        decision_action = map_decision_action(
            failure_risk=failure_prob,
            alert_threshold=alert_thresh,
            critical_threshold=critical_thresh,
            distribution_warning=has_warning,
        )
        is_alert = decision_action in ["REVIEW_REQUIRED", "PRIORITY_REVIEW"]

        # 5. Sinh mã lý do vận hành và ngữ cảnh đặc trưng
        reason_codes = self.extract_operational_reason_codes(raw_payload)
        feature_context = self.extract_feature_context(features_df)

        model_ver = self.manifest.get("model_version", "v2")
        policy_ver = self.policy.get("policy_version", "maintenance-policy-v2")

        # Ánh xạ risk tier tương thích ngược
        risk_tier_map = {
            "PRIORITY_REVIEW": "HIGH",
            "REVIEW_REQUIRED": "MEDIUM",
            "NO_ALERT": "LOW",
        }
        risk_tier = risk_tier_map.get(decision_action, "LOW")

        # 6. Đóng gói phản hồi theo cấu trúc 4 khối chuẩn mực
        return {
            # Khối 1: Dự báo ML thuần túy
            "prediction": {
                "failure_risk": round(failure_prob, 4),
                "model_version": model_ver,
                "model_type": self.manifest.get("model_type", "calibrated_ensemble"),
            },
            # Khối 2: Độ tin cậy & Cảnh báo phân bố
            "reliability": {
                "status": reliability_status,
                "distribution_warning": has_warning,
                "warning_features": warning_features,
            },
            # Khối 3: Quyết định vận hành
            "decision": {
                "action": decision_action,
                "alert_threshold": alert_thresh,
                "critical_threshold": critical_thresh,
                "maintenance_alert": is_alert,
                "policy_version": policy_ver,
                "cost_scenario": f"FN{fn_w:.0f}_FP{fp_w:.0f}",
            },
            # Khối 4: Ngữ cảnh vận hành & Giải thích
            "operational_context": {
                "reason_codes": reason_codes,
                "feature_context": feature_context,
            },
            # Các trường tương thích ngược ở cấp root
            "failure_risk": round(failure_prob, 4),
            "threshold": alert_thresh,
            "risk_tier": risk_tier,
            "alert": is_alert,
            "reason_codes": reason_codes,
            "model_version": model_ver,
            "model_explanation": feature_context,
        }
