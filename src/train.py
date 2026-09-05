"""Module huấn luyện mô hình ML và tối ưu hóa ngưỡng quyết định kinh doanh (Cost-Sensitive Threshold Tuning).

Module thực hiện các quy trình chính:
1. Xây dựng Data Preprocessing Pipeline (Imputation, Scaling, One-Hot Encoding).
2. Benchmark đa mô hình ứng viên (Model Zoo): Logistic Regression, Random Forest, HistGradientBoosting (cả dạng thô và hiệu chỉnh xác suất Sigmoid Calibration).
3. So sánh mô hình trên tập Validation theo tiêu chí PR-AUC và Brier Score để chọn Champion.
4. Tối ưu ngưỡng quyết định (Business Threshold Tuning) dựa trên ma trận chi phí (FN=5x, FP=1x) và ràng buộc công suất bảo trì (Maintenance Capacity Constraint).
5. Ghi nhận checksum SHA256 dữ liệu thô và khoảng phân bố đặc trưng (Feature Ranges) cho OOD detection.
6. Xuất mô hình (.joblib), tệp cấu hình (config.json), feature ranges (feature_ranges.json) và báo cáo validation (validation_metrics.json).
"""

from __future__ import annotations

import datetime
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import compute_dataset_sha256, extract_feature_ranges, load_data
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
    max_alert_rate: float | None = None,
) -> tuple[float, float]:
    """Tìm ngưỡng quyết định (threshold) có tổng chi phí nghiệp vụ nhỏ nhất trên tập Validation.

    Công thức chi phí: Total Cost = (FN_count * FN_cost) + (FP_count * FP_cost)

    Quy tắc hòa (Tie-breaking Rule):
    Nếu có nhiều ngưỡng cho ra cùng một tổng chi phí tối thiểu, thuật toán sẽ chọn ngưỡng cao nhất
    nhằm giảm bớt các cảnh báo giả, từ đó giảm áp lực cho đội ngũ bảo trì vận hành.

    Ràng buộc công suất (Capacity Constraint):
    Nếu `max_alert_rate` được chỉ định (ví dụ 0.05 = 5%), thuật toán sẽ chọn ngưỡng tối thiểu hóa chi phí
    trong số các ngưỡng thỏa mãn Alert Rate <= max_alert_rate.

    Args:
        labels (np.ndarray): Mảng chứa nhãn thực tế (0 hoặc 1).
        probabilities (np.ndarray): Mảng chứa xác suất dự báo máy hỏng thuộc [0, 1].
        costs (BusinessCosts | None): Đối tượng ma trận chi phí. Mặc định là BusinessCosts(5.0, 1.0).
        max_alert_rate (float | None): Ràng buộc tỷ lệ cảnh báo tối đa (Maintenance Capacity).

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

    # Tính toán tổng chi phí dự kiến và alert rate cho từng ngưỡng ứng viên
    expected_costs: list[float] = []
    alert_rates: list[float] = []
    for thresh in candidate_thresholds:
        preds = y_prob >= thresh
        fn_count = np.count_nonzero((y_true == 1) & (~preds))
        fp_count = np.count_nonzero((y_true == 0) & preds)
        cost = costs.false_negative * fn_count + costs.false_positive * fp_count
        expected_costs.append(cost)
        alert_rates.append(float(preds.mean()))

    expected_costs_arr = np.array(expected_costs, dtype=float)
    alert_rates_arr = np.array(alert_rates, dtype=float)

    # Áp dụng ràng buộc công suất bảo trì nếu có
    valid_mask = np.ones_like(expected_costs_arr, dtype=bool)
    if max_alert_rate is not None:
        capacity_mask = alert_rates_arr <= max_alert_rate
        if np.any(capacity_mask):
            valid_mask = capacity_mask
        else:
            LOGGER.warning(
                f"Không có ngưỡng nào thỏa mãn max_alert_rate <= {max_alert_rate:.2%}. Chọn ngưỡng có alert rate nhỏ nhất."
            )
            valid_mask = alert_rates_arr == alert_rates_arr.min()

    filtered_costs = expected_costs_arr[valid_mask]
    filtered_thresholds = candidate_thresholds[valid_mask]

    minimum_cost = float(filtered_costs.min())
    best_candidates = filtered_thresholds[filtered_costs == minimum_cost]

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


def build_pipeline(X_sample: Any, classifier: Any | None = None) -> Pipeline:
    """Xây dựng Scikit-Learn Preprocessing Pipeline tự động phân loại cột số và cột phân loại.

    Args:
        X_sample: Dataframe mẫu dùng để xác định kiểu dữ liệu các cột.
        classifier: Mô hình ước lượng (Estimator). Mặc định là LogisticRegression.

    Returns:
        Pipeline: Pipeline xử lý dữ liệu và mô hình phân loại tương ứng.
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

    if classifier is None:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=SEED,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_select_model() -> None:
    """Hàm chính thực thi toàn bộ quy trình huấn luyện đa mô hình, hiệu chỉnh xác suất, chọn champion và lưu artifact."""
    set_seed(SEED)
    LOGGER.info("Bắt đầu quy trình huấn luyện mô hình Machine Failure Risk System...")

    # 1. Nạp dữ liệu & tính Checksum SHA256
    raw_data_path = Path("data/raw/ai4i2020.csv")
    data_sha256 = compute_dataset_sha256(raw_data_path)
    LOGGER.info(f"Checksum SHA256 tập dữ liệu thô: {data_sha256[:12]}...")

    X_train, X_val, _, y_train, y_val, _ = load_data(path=raw_data_path, seed=SEED)
    LOGGER.info(
        f"Kích thước tập dữ liệu: Train={X_train.shape[0]} mẫu, Validation={X_val.shape[0]} mẫu."
    )

    # Trích xuất khoảng phân bố đặc trưng (Feature ranges) cho OOD detection
    feature_ranges = extract_feature_ranges(X_train)

    # 2. Định nghĩa Model Zoo (Tập hợp các mô hình ứng viên)
    log_reg = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED
    )
    rf_clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=SEED
    )
    hgb_clf = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=SEED
    )

    candidate_models: dict[str, Any] = {
        "logistic_baseline": build_pipeline(X_train, log_reg),
        "logistic_sigmoid_calibrated": CalibratedClassifierCV(
            build_pipeline(X_train, log_reg),
            method="sigmoid",
            cv=3,
        ),
        "random_forest_baseline": build_pipeline(X_train, rf_clf),
        "rf_sigmoid_calibrated": CalibratedClassifierCV(
            build_pipeline(X_train, rf_clf),
            method="sigmoid",
            cv=3,
        ),
        "hist_gb_baseline": build_pipeline(X_train, hgb_clf),
        "hist_gb_sigmoid_calibrated": CalibratedClassifierCV(
            build_pipeline(X_train, hgb_clf),
            method="sigmoid",
            cv=3,
        ),
    }

    leaderboard: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}
    model_val_probs: dict[str, np.ndarray] = {}

    # 3. Huấn luyện và đánh giá trên tập Validation
    for name, model_candidate in candidate_models.items():
        LOGGER.info(f"Đang huấn luyện mô hình ứng viên: {name}...")
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

    # 4. Chiến lược chọn mô hình Champion (Model Selection Strategy):
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
    LOGGER.info(f"Mô hình được chọn làm Champion: '{selected_model_name}'")

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

    # 6. Lưu mô hình (.joblib), feature ranges (feature_ranges.json) và cấu hình hệ thống (config.json)
    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    joblib.dump(best_model, models_dir / "model.joblib")
    save_json(models_dir / "feature_ranges.json", feature_ranges)
    LOGGER.info("Đã lưu mô hình tại 'models/model.joblib' và feature_ranges tại 'models/feature_ranges.json'.")

    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    model_version = f"ai4i-{date_str}-{data_sha256[:7]}"

    model_config = {
        "schema_version": 2,
        "version": model_version,
        "feature_contract_version": "ai4i-features-v2",
        "data_sha256": data_sha256,
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
            "limitation": "Dataset không có chuỗi thời gian/machine history đủ để group-time split hay forecast RUL.",
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
        f"Huấn luyện hoàn tất! Champion Model: '{selected_model_name}', "
        f"Ngưỡng tối ưu: {optimal_threshold:.4f}, "
        f"Metrics: {leaderboard[selected_model_name]}"
    )


if __name__ == "__main__":
    train_and_select_model()

