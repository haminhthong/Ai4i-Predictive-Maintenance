"""Pipeline tiền xử lý và mô hình cho snapshot failure-risk scoring.

Cung cấp:
1. Pipeline tiền xử lý tự động (ColumnTransformer: Median Imputer + Scaler cho cột số, OneHotEncoder cho cột phân loại).
2. Ba mô hình ứng viên: Logistic Regression, Random Forest và HistGradientBoosting.
   Hiệu chuẩn xác suất được thực hiện ở stage riêng sau khi chọn base model.
3. Các hàm đánh giá hiệu năng phân loại (PR-AUC, ROC-AUC, Brier score, ECE).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Xây dựng Scikit-Learn ColumnTransformer xử lý dữ liệu số và dữ liệu phân loại."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )


def build_pipeline(
    classifier: Any,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Ghép Preprocessor và Classifier thành Pipeline hoàn chỉnh."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def get_candidate_models(seed: int = 42) -> dict[str, Any]:
    """Tạo ba base model để so sánh bằng Stratified CV trên Development."""
    def make_log() -> Pipeline:
        return build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed)
        )

    def make_rf() -> Pipeline:
        return build_pipeline(
            RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1
            )
        )

    def make_hgb() -> Pipeline:
        return build_pipeline(
            HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
        )

    return {
        "logistic_regression": make_log(),
        "random_forest": make_rf(),
        "hist_gradient_boosting": make_hgb(),
    }


def calibrate_model(model: Any, method: str = "sigmoid", cv: int = 3) -> Any:
    """Bọc base model bằng calibration cross-fitted sau khi đã chọn model."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Phương pháp calibration phải là 'sigmoid' hoặc 'isotonic'.")
    return CalibratedClassifierCV(model, method=method, cv=cv)


def compute_calibration_curve_and_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """Tính toán Expected Calibration Error (ECE) và chi tiết các bin trong calibration curve."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    curve_points: list[dict[str, float]] = []

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_size = int(np.sum(bin_mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[bin_mask]))
            bin_conf = float(np.mean(y_prob[bin_mask]))
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
            curve_points.append(
                {
                    "bin_lower": float(bin_edges[i]),
                    "bin_upper": float(bin_edges[i + 1]),
                    "count": bin_size,
                    "mean_predicted": bin_conf,
                    "fraction_of_positives": bin_acc,
                }
            )

    return float(ece), curve_points


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> dict[str, Any]:
    """Tính toán toàn diện các chỉ số phân loại và chi phí tổn thất tương đối."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    weighted_cost = float(fn * fn_weight + fp * fp_weight)
    cost_per_1000 = float((weighted_cost / len(y_true)) * 1000.0)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": brier,
        "alert_rate": float(y_pred.mean()),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "weighted_cost": weighted_cost,
        "cost_units_per_1000": cost_per_1000,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
