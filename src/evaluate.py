"""Module đánh giá độc lập trên tập Locked Test (Hold-out Test Set).

NGUYÊN TẮC VÀNG TRONG EVALUATION (ZERO LEAKAGE ORACLE PROTOCOL):
1. Không thực hiện bất kỳ tối ưu hóa hay tìm kiếm ngưỡng (Grid-search/Argmax) nào trên tập Test.
2. Nạp toàn bộ các ngưỡng quyết định ĐÃ ĐƯỢC ĐÓNG BĂNG từ `decision_policy.json` (huấn luyện trên Validation).
3. Đánh giá toàn diện các chỉ số: PR-AUC, ROC-AUC, Precision, Recall, F1, Brier, ECE, Alert Rate.
4. Đánh giá đa kịch bản (Threshold Ablation Study & Capacity Constraints).
5. Phân tích lát cắt theo từng cơ chế hỏng hóc (Failure-Mode Slices: TWF, HDF, PWF, OSF, RNF) và giải phẫu FN/FP.
6. Tính toán Weighted Decision Cost (Chi phí trọng số) - tuyệt đối không gán đơn vị tiền tệ $ khi chưa có case study.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifact import find_latest_release, sha256_file
from .contracts import FAILURE_MODE_COLUMNS, FAILURE_MODE_DESCRIPTIONS
from .data import load_data
from .models import compute_calibration_curve_and_ece, compute_classification_metrics
from .policy import failure_capture_at_k, queue_precision_at_k
from .utils import LOGGER, save_json, setup_logging

ARTIFACTS_DIR = Path("artifacts/champion")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_frozen_artifacts() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Nạp model/policy từ release bundle đã đóng băng."""
    release_dir = find_latest_release()
    if release_dir is not None:
        model = joblib.load(release_dir / "model.joblib")
        policy = json.loads((release_dir / "decision_policy.json").read_text(encoding="utf-8"))
        config = json.loads((release_dir / "model_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "model_version": config.get("model_version", manifest.get("release_version")),
                "model_type": config.get("model_type", manifest.get("model_type", "unknown")),
            }
        )
        return model, policy, manifest

    model_path = ARTIFACTS_DIR / "model.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "model.joblib"

    policy_path = ARTIFACTS_DIR / "decision_policy.json"
    manifest_path = ARTIFACTS_DIR / "model_manifest.json"

    if not model_path.exists() or not policy_path.exists():
        # Fallback đọc từ models/config.json nếu artifacts chưa có
        config_path = MODELS_DIR / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy mô hình hoặc chính sách đã huấn luyện. "
                "Vui lòng chạy `python -m src.train` trước khi thực hiện đánh giá."
            )
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        policy = cfg.get("decision_policy", {})
        manifest = {
            "model_version": cfg.get("version", "v2"),
            "model_type": cfg.get("selected_model", "unknown"),
        }
        return model, policy, manifest

    model = joblib.load(model_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return model, policy, manifest


def evaluate_failure_mode_slices(
    modes_df: pd.DataFrame,
    y_test: np.ndarray,
    test_preds: np.ndarray,
) -> dict[str, Any]:
    """Phân tích lát cắt khả năng phát hiện lỗi của mô hình theo từng cơ chế hỏng hóc (Failure Modes).

    LƯU Ý: modes_df chứa các cờ hậu nghiệm (TWF, HDF, PWF, OSF, RNF) hoàn toàn được cô lập làm Evaluation Metadata,
    không tham gia vào bất kỳ khâu tính toán đặc trưng hay dự báo nào.
    """
    slices_report: dict[str, Any] = {}

    for mode_col in FAILURE_MODE_COLUMNS:
        if mode_col not in modes_df.columns:
            continue

        mode_mask = modes_df[mode_col].to_numpy().astype(int) == 1
        total_mode_failures = int(np.sum(mode_mask))

        if total_mode_failures > 0:
            detected = int(np.sum(test_preds[mode_mask]))
            recall_rate = float(detected / total_mode_failures)
        else:
            detected = 0
            recall_rate = 0.0

        slices_report[mode_col] = {
            "description": FAILURE_MODE_DESCRIPTIONS.get(mode_col, mode_col),
            "total_test_failures": total_mode_failures,
            "detected_by_policy": detected,
            "missed_by_policy": total_mode_failures - detected,
            "detection_recall": recall_rate,
            "detection_percentage": f"{recall_rate * 100:.2f}%",
        }

    return slices_report


def analyze_false_negatives_and_positives(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    test_preds: np.ndarray,
    modes_df: pd.DataFrame,
) -> dict[str, Any]:
    """Phân tích chi tiết các ca bỏ sót (False Negatives) và cảnh báo nhầm (False Positives)."""
    fn_mask = (y_test == 1) & (test_preds == 0)
    fp_mask = (y_test == 0) & (test_preds == 1)

    fn_count = int(np.sum(fn_mask))
    fp_count = int(np.sum(fp_mask))

    fn_modes: dict[str, int] = {}
    for col in FAILURE_MODE_COLUMNS:
        if col in modes_df.columns:
            fn_modes[col] = int(np.sum(modes_df.loc[fn_mask, col].to_numpy().astype(int) == 1))

    # Tóm tắt đặc trưng của các ca FN
    fn_features_summary = {}
    if fn_count > 0:
        fn_df = X_test[fn_mask]
        for col in ["torque_nm", "tool_wear_min", "rotational_speed_rpm", "temperature_delta_k"]:
            if col in fn_df.columns:
                fn_features_summary[col] = {
                    "mean": float(fn_df[col].mean()),
                    "min": float(fn_df[col].min()),
                    "max": float(fn_df[col].max()),
                }

    return {
        "false_negatives_count": fn_count,
        "false_positives_count": fp_count,
        "false_negatives_by_failure_mode": fn_modes,
        "false_negatives_feature_summary": fn_features_summary,
    }


def evaluate_twf_error_slice(
    X_test: pd.DataFrame,
    modes_test: pd.DataFrame,
    test_preds: np.ndarray,
) -> dict[str, Any]:
    """So sánh feature snapshot giữa TWF detected và TWF missed."""
    if "failure_twf" not in modes_test.columns:
        return {"status": "failure_twf metadata unavailable"}
    twf_mask = modes_test["failure_twf"].to_numpy().astype(int) == 1
    detected = twf_mask & (test_preds == 1)
    missed = twf_mask & (test_preds == 0)
    result: dict[str, Any] = {
        "total_twf": int(twf_mask.sum()),
        "detected": int(detected.sum()),
        "missed": int(missed.sum()),
    }
    for group_name, mask in (("detected", detected), ("missed", missed)):
        group = X_test.loc[mask]
        result[group_name] = {
            "count": len(group),
            "summary": {
                column: {
                    "mean": float(group[column].mean()),
                    "min": float(group[column].min()),
                    "max": float(group[column].max()),
                }
                for column in (
                    "tool_wear_min",
                    "torque_nm",
                    "wear_load_interaction",
                    "quality_type",
                )
                if column in group.columns and len(group) > 0 and column != "quality_type"
            },
            "product_quality_type_counts": (
                group["quality_type"].value_counts().to_dict()
                if "quality_type" in group.columns
                else {}
            ),
        }
    return result


def evaluate_model_on_locked_test() -> dict[str, Any]:
    """Đánh giá mô hình đã đóng băng trên tập Locked Hold-out Test hoàn toàn độc lập."""
    LOGGER.info("=== BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP LOCKED TEST (HOLD-OUT) ===")

    # 1. Nạp Locked Test kèm metadata phân tích lỗi (15% dữ liệu)
    _, _, X_test, _, _, y_test, modes_test = load_data(return_metadata=True)
    y_test_arr = y_test.to_numpy()
    LOGGER.info(f"Tập Test độc lập gồm {len(X_test)} mẫu ({int(np.sum(y_test_arr))} ca hỏng máy).")

    # 2. Nạp mô hình và chính sách quyết định đã đóng băng
    model, policy, manifest = load_frozen_artifacts()
    frozen_thresholds = policy.get("frozen_thresholds", {})
    cost_weights = policy.get("cost_weights", {"false_negative": 5.0, "false_positive": 1.0})
    fn_w = float(cost_weights.get("false_negative", 5.0))
    fp_w = float(cost_weights.get("false_positive", 1.0))

    primary_thresh = float(policy.get("primary_alert_threshold", 0.5))
    LOGGER.info(
        f"Áp dụng ngưỡng đã đóng băng từ Policy Validation: {primary_thresh:.4f}"
    )

    # 3. Tính toán xác suất dự báo rủi ro
    y_probs = model.predict_proba(X_test)[:, 1]

    # 4. Tính toán độ hiệu chuẩn xác suất (Calibration Curve & ECE)
    ece_val, calib_points = compute_calibration_curve_and_ece(y_test_arr, y_probs)

    # 5. Đánh giá đa chiến lược ngưỡng (Threshold Ablation Study) TRÊN TEST
    # LƯU Ý: Tất cả các ngưỡng đều được lấy từ frozen_thresholds, TUYỆT ĐỐI KHÔNG TỐI ƯU TRÊN TEST!
    threshold_candidates = {
        "fixed_0_50": frozen_thresholds.get("fixed_0_50", 0.50),
        "max_f1_validation_tuned": frozen_thresholds.get("max_f1_validation", 0.5),
        "cost_sensitive_validation_tuned": primary_thresh,
        "capacity_constrained_5pct": frozen_thresholds.get("capacity_constrained_5pct", primary_thresh),
        "capacity_constrained_3pct": frozen_thresholds.get("capacity_constrained_3pct", primary_thresh),
        "capacity_constrained_2pct": frozen_thresholds.get("capacity_constrained_2pct", primary_thresh),
    }

    ablation_study: dict[str, Any] = {}
    for name, thresh in threshold_candidates.items():
        metrics = compute_classification_metrics(
            y_test_arr, y_probs, threshold=thresh, fn_weight=fn_w, fp_weight=fp_w
        )
        ablation_study[name] = metrics

    # 6. Đánh giá chính sách chính (Cost-sensitive tuned)
    primary_metrics = ablation_study["cost_sensitive_validation_tuned"]
    test_preds = (y_probs >= primary_thresh).astype(int)

    # 7. Phân tích lát cắt theo từng cơ chế hỏng hóc (Failure-Mode Slices)
    failure_slices = evaluate_failure_mode_slices(modes_test, y_test_arr, test_preds)

    # 8. Phân tích lỗi các ca FN và FP
    error_analysis = analyze_false_negatives_and_positives(
        X_test, y_test_arr, y_probs, test_preds, modes_test
    )
    twf_error_analysis = evaluate_twf_error_slice(X_test, modes_test, test_preds)

    # 9. Tổng hợp kết quả báo cáo
    final_test_report = {
        "model_version": manifest.get("model_version", "v2"),
        "model_type": manifest.get("model_type", "calibrated_model"),
        "evaluation_protocol": "locked_holdout_test_zero_leakage",
        "total_test_samples": len(X_test),
        "total_test_failures": int(np.sum(y_test_arr)),
        "primary_policy": {
            "policy_version": policy.get("policy_version", "maintenance-policy-v2"),
            "alert_threshold": primary_thresh,
            "critical_threshold": policy.get("critical_threshold", 0.75),
            "cost_scenario": f"FN{fn_w:.0f}_FP{fp_w:.0f}",
            "cost_unit": "relative_cost_units",
        },
        "test_performance": {
            "pr_auc": primary_metrics["pr_auc"],
            "roc_auc": primary_metrics["roc_auc"],
            "precision": primary_metrics["precision"],
            "recall": primary_metrics["recall"],
            "f1_score": primary_metrics["f1"],
            "brier_score": primary_metrics["brier"],
            "ece": ece_val,
            "alert_rate": primary_metrics["alert_rate"],
            "weighted_decision_cost": primary_metrics["weighted_cost"],
            "cost_units_per_1000_observations": primary_metrics["cost_units_per_1000"],
            "confusion_matrix": primary_metrics["confusion_matrix"],
            "failure_capture_at_1pct": failure_capture_at_k(y_test_arr, y_probs, 0.01),
            "failure_capture_at_2pct": failure_capture_at_k(y_test_arr, y_probs, 0.02),
            "failure_capture_at_3pct": failure_capture_at_k(y_test_arr, y_probs, 0.03),
            "queue_precision_at_1pct": queue_precision_at_k(y_test_arr, y_probs, 0.01),
            "queue_precision_at_2pct": queue_precision_at_k(y_test_arr, y_probs, 0.02),
            "queue_precision_at_3pct": queue_precision_at_k(y_test_arr, y_probs, 0.03),
            "review_coverage": primary_metrics["alert_rate"],
            "priority_override_rate": float(
                np.sum(y_probs >= float(policy.get("critical_threshold", 0.75)))
                / max(np.sum(y_probs >= primary_thresh), 1)
            ),
        },
        "threshold_ablation_study": ablation_study,
        "failure_mode_analysis": failure_slices,
        "error_analysis_fn_fp": error_analysis,
        "twf_error_analysis": twf_error_analysis,
        "calibration_curve": calib_points,
    }

    # 10. Lưu các báo cáo chi tiết. Báo cáo legacy có thể bị khóa bởi dashboard;
    # release bundle vẫn là nguồn chuẩn và được ghi riêng ở bên dưới.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for report_path, report in (
        (REPORTS_DIR / "final_test_metrics.json", final_test_report),
        (REPORTS_DIR / "failure_mode_analysis.json", failure_slices),
        (REPORTS_DIR / "twf_error_analysis.json", twf_error_analysis),
    ):
        try:
            save_json(report_path, report)
        except PermissionError:
            LOGGER.warning("Bỏ qua báo cáo legacy bị khóa: %s", report_path)

    # Cập nhật locked_test_metrics trong release sau khi đã đánh giá; không thay đổi policy.
    release_dir = find_latest_release()
    if release_dir is not None:
        save_json(release_dir / "locked_test_metrics.json", final_test_report)
        manifest_path = release_dir / "manifest.json"
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        release_manifest["file_hashes"] = {
            path.name: sha256_file(path)
            for path in release_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        release_manifest["model_sha256"] = release_manifest["file_hashes"]["model.joblib"]
        release_manifest["feature_contract_sha256"] = release_manifest["file_hashes"]["feature_contract.json"]
        release_manifest["policy_sha256"] = release_manifest["file_hashes"]["decision_policy.json"]
        release_manifest["reference_distribution_sha256"] = release_manifest["file_hashes"]["reference_distribution.json"]
        save_json(manifest_path, release_manifest)

    LOGGER.info("=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP ===")
    LOGGER.info(
        f"PR-AUC: {primary_metrics['pr_auc']:.4f} | ROC-AUC: {primary_metrics['roc_auc']:.4f} | "
        f"Brier: {primary_metrics['brier']:.4f} | ECE: {ece_val:.4f}"
    )
    LOGGER.info(
        f"Precision: {primary_metrics['precision']:.4f} | Recall: {primary_metrics['recall']:.4f} | "
        f"F1: {primary_metrics['f1']:.4f} | Alert Rate: {primary_metrics['alert_rate']:.2%}"
    )
    LOGGER.info(
        f"Weighted Decision Cost: {primary_metrics['weighted_cost']:.2f} cost units "
        f"({primary_metrics['cost_units_per_1000']:.2f} cost units / 1,000 observations)"
    )
    LOGGER.info("Phân tích cơ chế hỏng hóc (Failure Mode Recall):")
    for mode, data in failure_slices.items():
        LOGGER.info(
            f"  - {mode:12s}: {data['detected_by_policy']}/{data['total_test_failures']} phát hiện "
            f"({data['detection_percentage']})"
        )

    return final_test_report


if __name__ == "__main__":
    setup_logging()
    evaluate_model_on_locked_test()
