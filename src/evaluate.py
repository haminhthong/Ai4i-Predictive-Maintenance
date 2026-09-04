"""Module đánh giá độc lập mô hình trên tập Test (Hold-out Test Set).

Module thực hiện các bước:
1. Nạp dữ liệu tập Test hoàn toàn độc lập (20% dữ liệu chưa từng thấy).
2. Tải mô hình đã huấn luyện (model.joblib) và cấu hình ngưỡng quyết định (config.json).
3. Đánh giá toàn diện các chỉ số phân loại: PR-AUC, ROC-AUC, Precision, Recall, F1-Score, Brier Score, Alert Rate và Ma trận nhầm lẫn (Confusion Matrix).
4. Xuất kết quả chi tiết ra tệp `reports/test_metrics.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
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


def evaluate_model_on_test() -> dict[str, Any]:
    """Đánh giá mô hình đã lưu trên tập Test độc lập.

    Returns:
        dict[str, Any]: Từ điển chứa tất cả chỉ số đánh giá kỹ thuật và kinh doanh.
    """
    LOGGER.info("Bắt đầu quy trình đánh giá mô hình trên tập Test độc lập...")

    # 1. Nạp tập dữ liệu Test (Hold-out test split)
    _, _, X_test, _, _, y_test = load_data()

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

    threshold = float(config.get("threshold", 0.5))
    LOGGER.info(f"Áp dụng ngưỡng quyết định tối ưu từ config: {threshold:.4f}")

    # 3. Tính toán xác suất dự báo rủi ro hỏng máy
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    # 4. Tính toán các metric kỹ thuật và kinh doanh
    pr_auc_val = float(average_precision_score(y_test, y_probs))
    roc_auc_val = float(roc_auc_score(y_test, y_probs))
    precision_val = float(precision_score(y_test, y_preds, zero_division=0))
    recall_val = float(recall_score(y_test, y_preds, zero_division=0))
    f1_val = float(f1_score(y_test, y_preds, zero_division=0))
    brier_val = float(brier_score_loss(y_test, y_probs))
    alert_rate_val = float(y_preds.mean())
    cm_list = confusion_matrix(y_test, y_preds).tolist()

    test_metrics = {
        "model_version": config.get("version", "v2"),
        "selected_model": config.get("selected_model", "unknown"),
        "threshold": threshold,
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val,
        "precision": precision_val,
        "recall": recall_val,
        "f1": f1_val,
        "brier": brier_val,
        "alert_rate": alert_rate_val,
        "confusion_matrix": cm_list,
    }

    # 5. Lưu báo cáo đánh giá vào tập tin JSON
    reports_path = Path("reports/test_metrics.json")
    save_json(reports_path, test_metrics)

    LOGGER.info("=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ===")
    LOGGER.info(f"PR-AUC: {pr_auc_val:.4f} | ROC-AUC: {roc_auc_val:.4f}")
    LOGGER.info(
        f"Precision: {precision_val:.4f} | Recall: {recall_val:.4f} | F1: {f1_val:.4f}"
    )
    LOGGER.info(f"Brier Score: {brier_val:.4f} | Alert Rate: {alert_rate_val:.4f}")
    LOGGER.info(f"Confusion Matrix [[TN, FP], [FN, TP]]: {cm_list}")

    return test_metrics


if __name__ == "__main__":
    metrics = evaluate_model_on_test()
    LOGGER.info(json.dumps(metrics, indent=2, ensure_ascii=True))
