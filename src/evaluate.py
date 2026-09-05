"""Module đánh giá độc lập mô hình trên tập Test (Hold-out Test Set).

Module thực hiện các bước:
1. Nạp dữ liệu tập Test hoàn toàn độc lập (20% dữ liệu chưa từng thấy).
2. Tải mô hình đã huấn luyện (model.joblib) và cấu hình ngưỡng quyết định (config.json).
3. Đánh giá toàn diện các chỉ số phân loại: PR-AUC, ROC-AUC, Precision, Recall, F1-Score, Brier Score, ECE, Alert Rate và Confusion Matrix.
4. Tính toán **Test Expected Business Cost** (FN*5 + FP*1) và quy đổi chi phí trên 1.000 thiết bị.
5. Thực hiện **Ablation Study** so sánh 3 chiến lược threshold (Mặc định 0.5 vs. Max F1 vs. Cost-Sensitive Tuned).
6. Xuất kết quả chi tiết ra tệp `reports/test_metrics.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import load_data
from .utils import LOGGER, save_json


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Tính toán Expected Calibration Error (ECE) và chi tiết các điểm dữ liệu Calibration Curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    curve_points: list[dict[str, float]] = []

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
            curve_points.append(
                {
                    "bin_lower": float(bin_edges[i]),
                    "bin_upper": float(bin_edges[i + 1]),
                    "count": int(bin_size),
                    "mean_predicted": float(bin_conf),
                    "fraction_of_positives": float(bin_acc),
                }
            )

    return float(ece), curve_points


def evaluate_threshold_strategy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict[str, float]:
    """Tính toán bộ metric đầy đủ cho một ngưỡng quyết định cụ thể."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    expected_cost = float(fn * fn_cost + fp * fp_cost)
    cost_per_1000 = float((expected_cost / len(y_true)) * 1000.0)

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "alert_rate": float(y_pred.mean()),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "expected_cost": expected_cost,
        "cost_per_1000_machines": cost_per_1000,
    }


def evaluate_model_on_test() -> dict[str, Any]:
    """Đánh giá mô hình đã lưu trên tập Test độc lập.

    Returns:
        dict[str, Any]: Từ điển chứa tất cả chỉ số đánh giá kỹ thuật, hiệu chỉnh xác suất và chi phí kinh doanh.
    """
    LOGGER.info("Bắt đầu quy trình đánh giá mô hình trên tập Test độc lập...")

    # 1. Nạp tập dữ liệu Test (Hold-out test split)
    _, _, X_test, _, _, y_test = load_data()
    y_test_arr = y_test.to_numpy()

    # 2. Kiểm tra tệp mô hình và tệp cấu hình
    model_path = Path("models/model.joblib")
    config_path = Path("models/config.json")

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy mô hình hoặc tệp cấu hình! "
            "Vui lòng chạy `python -m src.train` trước khi đánh giá."
        )

    model = joblib.load(model_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    tuned_threshold = float(config.get("threshold", 0.5))
    fn_cost = float(config.get("false_negative_cost", 5.0))
    fp_cost = float(config.get("false_positive_cost", 1.0))
    LOGGER.info(f"Áp dụng ngưỡng quyết định tối ưu từ config: {tuned_threshold:.4f}")

    # 3. Tính toán xác suất dự báo rủi ro hỏng máy
    y_probs = model.predict_proba(X_test)[:, 1]

    # 4. Tính toán các metric kỹ thuật cốt lõi
    pr_auc_val = float(average_precision_score(y_test_arr, y_probs))
    roc_auc_val = float(roc_auc_score(y_test_arr, y_probs))
    brier_val = float(brier_score_loss(y_test_arr, y_probs))
    ece_val, calibration_curve_points = compute_expected_calibration_error(y_test_arr, y_probs)

    # 5. Thực hiện Threshold Strategy Ablation Study trên Test Set
    # Tìm ngưỡng Max F1 trên tập test để làm đối chứng
    candidate_thresholds = np.unique(np.r_[0.0, y_probs, 1.0])
    f1_scores = [
        f1_score(y_test_arr, (y_probs >= t).astype(int), zero_division=0)
        for t in candidate_thresholds
    ]
    max_f1_thresh = float(candidate_thresholds[np.argmax(f1_scores)])

    ablation_study = {
        "fixed_0_50": evaluate_threshold_strategy(
            y_test_arr, y_probs, 0.50, fn_cost, fp_cost
        ),
        "max_f1_strategy": evaluate_threshold_strategy(
            y_test_arr, y_probs, max_f1_thresh, fn_cost, fp_cost
        ),
        "cost_sensitive_tuned": evaluate_threshold_strategy(
            y_test_arr, y_probs, tuned_threshold, fn_cost, fp_cost
        ),
    }

    primary_eval = ablation_study["cost_sensitive_tuned"]

    test_metrics = {
        "model_version": config.get("version", "v2"),
        "feature_contract_version": config.get("feature_contract_version", "v2"),
        "selected_model": config.get("selected_model", "unknown"),
        "threshold": tuned_threshold,
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "precision": primary_eval["precision"],
        "recall": primary_eval["recall"],
        "f1": primary_eval["f1"],
        "brier": brier_val,
        "ece": ece_val,
        "alert_rate": primary_eval["alert_rate"],
        "expected_business_cost": primary_eval["expected_cost"],
        "cost_per_1000_machines": primary_eval["cost_per_1000_machines"],
        "confusion_matrix": [
            [primary_eval["true_negatives"], primary_eval["false_positives"]],
            [primary_eval["false_negatives"], primary_eval["true_positives"]],
        ],
        "threshold_ablation_study": ablation_study,
        "calibration_curve": calibration_curve_points,
    }

    # 6. Lưu báo cáo đánh giá vào tập tin JSON
    reports_path = Path("reports/test_metrics.json")
    save_json(reports_path, test_metrics)

    LOGGER.info("=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP ===")
    LOGGER.info(f"Model: {config.get('selected_model')} | Version: {config.get('version')}")
    LOGGER.info(f"PR-AUC: {pr_auc_val:.4f} | ROC-AUC: {roc_auc_val:.4f}")
    LOGGER.info(
        f"Precision: {primary_eval['precision']:.4f} | Recall: {primary_eval['recall']:.4f} | F1: {primary_eval['f1']:.4f}"
    )
    LOGGER.info(f"Brier Score: {brier_val:.4f} | ECE: {ece_val:.4f} | Alert Rate: {primary_eval['alert_rate']:.4%}")
    LOGGER.info(
        f"Expected Business Cost (FN*5 + FP*1): {primary_eval['expected_cost']:.2f} "
        f"({primary_eval['cost_per_1000_machines']:.2f} per 1,000 machines)"
    )
    LOGGER.info("Ablation Study (Cost-Sensitive vs 0.50 vs Max F1):")
    for strat, res in ablation_study.items():
        LOGGER.info(
            f"  - {strat:20s} | Thresh: {res['threshold']:.4f} | Recall: {res['recall']:.4f} | "
            f"Alert Rate: {res['alert_rate']:.4%} | Cost/1k: ${res['cost_per_1000_machines']:.2f}"
        )

    return test_metrics


if __name__ == "__main__":
    metrics = evaluate_model_on_test()
    LOGGER.info(json.dumps(metrics, indent=2, ensure_ascii=True))

