"""Module huấn luyện mô hình ML và tối ưu hóa ngưỡng quyết định kinh doanh (Cost-Sensitive Threshold Tuning).

Module thực hiện các quy trình chính:
1. Xây dựng Data Preprocessing Pipeline (Imputation, Scaling, One-Hot Encoding).
2. Huấn luyện và सो sánh hai mô hình: Logistic Regression baseline và Sigmoid Calibrated Classifier.
3. So sánh mô hình trên tập Validation theo tiêu chí PR-AUC và Brier Score.
4. Tối ưu ngưỡng quyết định (Business Threshold Tuning) dựa trên ma trận chi phí giả định (False Negative = 5x False Positive).
5. Xuất xuất mô hình (.joblib), tệp cấu hình (config.json) và báo cáo validation (validation_metrics.json).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load_data
from .utils import LOGGER, save_json, set_seed

# Seed cố định cho toàn bộ quá trình huấn luyện
SEED = 42


@dataclass(frozen=True)
class BusinessCosts:
    """Định nghĩa chi phí kinh doanh tương đối dùng để quy đổi xác suất dự báo thành quyết định phát cảnh báo.

    Attributes:
        false_negative (float): Chi phí thiệt hại khi bỏ sót 1 máy hỏng (FN). Mặc định là 5.0.
        false_positive (float): Chi phí kiểm tra khi phát cảnh báo nhầm (FP). Mặc định là 1.0.
    """

    false_negative: float = 5.0
    false_positive: float = 1.0

    def validate(self) -> None:
        """Kiểm tra điều kiện hợp lệ của trọng số chi phí."""
        if self.false_negative <= 0 or self.false_positive <= 0:
            raise ValueError(
                "Chi phí False Negative và False Positive phải là số dương lớn hơn 0."
            )


def select_business_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    costs: BusinessCosts | None = None,
) -> tuple[float, float]:
    """Tìm ngưỡng quyết định (threshold) có tổng chi phí nghiệp vụ nhỏ nhất trên tập Validation.

    Công thức chi phí: Total Cost = (FN_count * FN_cost) + (FP_count * FP_cost)

    Quy tắc hòa (Tie-breaking Rule):
    Nếu có nhiều ngưỡng cho ra cùng một tổng chi phí tối thiểu, thuật toán sẽ chọn ngưỡng cao nhất
    nhằm giảm bớt các cảnh báo giả, từ đó giảm áp lực cho đội ngũ bảo trì vận hành.

    Args:
        labels (np.ndarray): Mảng chứa nhãn thực tế (0 hoặc 1).
        probabilities (np.ndarray): Mảng chứa xác suất dự báo máy hỏng thuộc [0, 1].
        costs (BusinessCosts | None): Đối tượng ma trận chi phí. Mặc định là BusinessCosts(5.0, 1.0).

    Returns:
        tuple[float, float]: (Ngưỡng quyết định tối ưu, Tổng chi phí tương ứng)
    """
    costs = costs or BusinessCosts()
    costs.validate()

    y_true = np.asarray(labels)
    y_prob = np.asarray(probabilities, dtype=float)

    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError(
            "Danh sách nhãn và xác suất phải cùng kích thước và không được rỗng."
        )

    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("Tất cả giá trị xác suất phải nằm trong đoạn [0, 1].")

    # Tạo danh sách các ứng viên ngưỡng từ các giá trị xác suất quan sát thực tế kèm biên [0, 1]
    candidate_thresholds = np.unique(np.r_[0.0, y_prob, 1.0])

    # Tính toán tổng chi phí dự kiến cho từng ngưỡng ứng viên
    expected_costs = np.array(
        [
            costs.false_negative * np.count_nonzero((y_true == 1) & (y_prob < thresh))
            + costs.false_positive
            * np.count_nonzero((y_true == 0) & (y_prob >= thresh))
            for thresh in candidate_thresholds
        ],
        dtype=float,
    )

    minimum_cost = float(expected_costs.min())
    best_candidates = candidate_thresholds[expected_costs == minimum_cost]

    # Ưu tiên chọn ngưỡng lớn nhất để tối ưu chi phí vận hành
    optimal_threshold = float(best_candidates.max())
    return optimal_threshold, minimum_cost


def analyze_cost_sensitivity(
    labels: np.ndarray,
    probabilities: np.ndarray,
    false_negative_costs: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
    false_positive_cost: float = 1.0,
) -> list[dict[str, float]]:
    """Đo độ nhạy của threshold khi giả định chi phí bỏ sót thay đổi."""
    scenarios: list[dict[str, float]] = []
    for false_negative_cost in false_negative_costs:
        costs = BusinessCosts(false_negative_cost, false_positive_cost)
        threshold, expected_cost = select_business_threshold(
            labels, probabilities, costs
        )
        predictions = probabilities >= threshold
        scenarios.append(
            {
                "false_negative_cost": false_negative_cost,
                "false_positive_cost": false_positive_cost,
                "threshold": threshold,
                "expected_cost": expected_cost,
                "alert_rate": float(predictions.mean()),
                "recall": float(
                    predictions[labels == 1].mean() if np.any(labels == 1) else 0.0
                ),
            }
        )
    return scenarios


def build_pipeline(X_sample: Any) -> Pipeline:
    """Xây dựng Scikit-Learn Preprocessing Pipeline tự động phân loại cột số và cột phân loại.

    Args:
        X_sample: Dataframe mẫu dùng để xác định kiểu dữ liệu các cột.

    Returns:
        Pipeline: Pipeline xử lý dữ liệu và mô hình phân loại Logistic Regression baseline.
    """
    numeric_features = [c for c in X_sample.columns if is_numeric_dtype(X_sample[c])]
    categorical_features = [c for c in X_sample.columns if c not in numeric_features]

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Logistic Regression cân bằng trọng số lớp (class_weight='balanced')
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_select_model() -> None:
    """Hàm chính thực thi toàn bộ quy trình huấn luyện, hiệu chỉnh xác suất, đánh giá và lưu mô hình."""
    set_seed(SEED)
    LOGGER.info("Bắt đầu quy trình huấn luyện mô hình Predictive Maintenance...")

    # 1. Nạp dữ liệu
    X_train, X_val, _, y_train, y_val, _ = load_data(seed=SEED)
    LOGGER.info(
        f"Kích thước tập dữ liệu: Train={X_train.shape[0]} mẫu, Validation={X_val.shape[0]} mẫu."
    )

    # 2. Định nghĩa các mô hình ứng viên
    baseline_pipeline = build_pipeline(X_train)
    candidate_models = {
        "logistic_baseline": clone(baseline_pipeline),
        "logistic_sigmoid_calibrated": CalibratedClassifierCV(
            clone(baseline_pipeline),
            method="sigmoid",
            cv=3,
        ),
    }

    leaderboard: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}
    model_val_probs: dict[str, np.ndarray] = {}

    # 3. Huấn luyện và đánh giá trên tập Validation
    for name, model_candidate in candidate_models.items():
        LOGGER.info(f"Đang huấn luyện mô hình: {name}...")
        model_candidate.fit(X_train, y_train)
        val_probs = model_candidate.predict_proba(X_val)[:, 1]

        fitted_models[name] = model_candidate
        model_val_probs[name] = val_probs
        leaderboard[name] = {
            "pr_auc": float(average_precision_score(y_val, val_probs)),
            "brier": float(brier_score_loss(y_val, val_probs)),
        }
        LOGGER.info(
            f"Mô hình '{name}' -> PR-AUC: {leaderboard[name]['pr_auc']:.4f}, Brier Score: {leaderboard[name]['brier']:.4f}"
        )

    # 4. Chiến lược chọn mô hình (Model Selection Strategy):
    # Ưu tiên mô hình có PR-AUC tốt nhất. Nếu mức chênh lệch PR-AUC <= 0.01,
    # chọn mô hình có Brier Score nhỏ hơn (xác suất được hiệu chỉnh chuẩn xác hơn).
    best_pr_auc = max(m["pr_auc"] for m in leaderboard.values())
    eligible_models = [
        name for name, m in leaderboard.items() if m["pr_auc"] >= (best_pr_auc - 0.01)
    ]

    selected_model_name = min(
        eligible_models, key=lambda name: leaderboard[name]["brier"]
    )
    best_model = fitted_models[selected_model_name]
    LOGGER.info(f"Mô hình được chọn chiến thắng: '{selected_model_name}'")

    # 5. Tối ưu ngưỡng quyết định theo chi phí kinh doanh trên tập Validation
    val_selected_probs = model_val_probs[selected_model_name]
    business_costs = BusinessCosts(false_negative=5.0, false_positive=1.0)

    optimal_threshold, min_cost = select_business_threshold(
        y_val.to_numpy(),
        val_selected_probs,
        business_costs,
    )
    cost_sensitivity = analyze_cost_sensitivity(y_val.to_numpy(), val_selected_probs)
    LOGGER.info(
        f"Ngưỡng quyết định tối ưu trên Validation: {optimal_threshold:.4f} "
        f"(Tổng chi phí dự kiến: {min_cost:.2f})"
    )

    # 6. Lưu mô hình (.joblib) và cấu hình hệ thống (config.json)
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    joblib.dump(best_model, models_dir / "model.joblib")
    LOGGER.info("Đã lưu mô hình tại 'models/model.joblib'.")

    model_config = {
        "schema_version": 2,
        "version": "ai4i-calibrated-v3",
        "selected_model": selected_model_name,
        "threshold": optimal_threshold,
        "false_negative_cost": business_costs.false_negative,
        "false_positive_cost": business_costs.false_positive,
        "seed": SEED,
        "features": list(X_train.columns),
        "split_contract": {
            "method": "stratified_random",
            "train_fraction": 0.64,
            "validation_fraction": 0.16,
            "test_fraction": 0.20,
            "limitation": "Dataset không có chuỗi thời gian/machine history đủ để group-time split.",
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_json(models_dir / "config.json", model_config)

    validation_report = {
        "leaderboard": leaderboard,
        "selected_model": selected_model_name,
        "pr_auc": leaderboard[selected_model_name]["pr_auc"],
        "brier": leaderboard[selected_model_name]["brier"],
        "threshold": optimal_threshold,
        "expected_cost": min_cost,
        "cost_sensitivity": cost_sensitivity,
    }
    save_json(reports_dir / "validation_metrics.json", validation_report)

    LOGGER.info(
        f"Huấn luyện hoàn tất! Mô hình: '{selected_model_name}', "
        f"Ngưỡng tối ưu: {optimal_threshold:.4f}, "
        f"Metrics: {leaderboard[selected_model_name]}"
    )


if __name__ == "__main__":
    train_and_select_model()
