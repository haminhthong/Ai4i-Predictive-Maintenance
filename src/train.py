"""Huấn luyện, chọn và lưu model snapshot failure-risk."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate

from .contracts import (
    ENGINEERED_FEATURES,
    MODEL_FEATURE_CONTRACT,
    RAW_SENSOR_FEATURES,
)
from .data import (
    audit_dataset,
    compute_dataset_sha256,
    create_or_load_split_manifest,
    extract_feature_ranges,
    load_data,
    load_raw_dataset,
)
from .models import calibrate_model, get_candidate_models
from .policy import find_threshold_maximizing_f1
from .utils import LOGGER, save_json, set_seed, setup_logging

SEED = 42
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")


def cross_validate_candidates(
    candidates: dict[str, Any],
    X_development: Any,
    y_development: Any,
) -> dict[str, dict[str, float]]:
    """Báo cáo mean ± std PR-AUC và Brier qua 5-fold Stratified CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    leaderboard: dict[str, dict[str, float]] = {}
    for name, model in candidates.items():
        result = cross_validate(
            model,
            X_development,
            y_development,
            cv=cv,
            scoring={"pr_auc": "average_precision", "brier": "neg_brier_score"},
            n_jobs=1,
        )
        brier_scores = -result["test_brier"]
        leaderboard[name] = {
            "pr_auc_mean": float(result["test_pr_auc"].mean()),
            "pr_auc_std": float(result["test_pr_auc"].std(ddof=1)),
            "brier_mean": float(brier_scores.mean()),
            "brier_std": float(brier_scores.std(ddof=1)),
        }
    return leaderboard


def choose_model(leaderboard: dict[str, dict[str, float]]) -> str:
    """Chọn PR-AUC CV cao nhất; hòa thì chọn Brier thấp hơn."""
    if not leaderboard:
        raise ValueError("Leaderboard không được rỗng.")
    return min(
        leaderboard,
        key=lambda name: (
            -leaderboard[name]["pr_auc_mean"],
            leaderboard[name]["brier_mean"],
            name,
        ),
    )


def feature_ablation(
    selected_model: str,
    X_development: Any,
    y_development: Any,
    full_score: dict[str, float],
) -> dict[str, Any]:
    """So sánh raw 6 biến với raw + 3 biến engineered bằng cùng model."""
    raw_X = X_development[list(RAW_SENSOR_FEATURES)]
    raw_model = get_candidate_models(SEED, RAW_SENSOR_FEATURES)[selected_model]
    raw_scores = cross_validate_candidates({selected_model: raw_model}, raw_X, y_development)[
        selected_model
    ]
    return {
        "model": selected_model,
        "raw_6": raw_scores,
        "raw_plus_engineered_9": full_score,
        "engineered_features": list(ENGINEERED_FEATURES),
    }


def train_model() -> dict[str, Any]:
    """Chạy CV -> calibration -> threshold validation -> lưu artifact chuẩn."""
    set_seed(SEED)
    raw_path = Path("data/raw/ai4i2020.csv")
    raw_df = load_raw_dataset(raw_path)
    data_hash = compute_dataset_sha256(raw_path)
    audit_dataset(raw_df, sha256_hash=data_hash)
    split_manifest = create_or_load_split_manifest(raw_df, seed=SEED)
    X_development, X_validation, _, y_development, y_validation, _ = load_data(
        path=raw_path,
        seed=SEED,
    )

    candidates = get_candidate_models(SEED)
    leaderboard = cross_validate_candidates(candidates, X_development, y_development)
    selected_model_name = choose_model(leaderboard)
    LOGGER.info("Model được chọn theo PR-AUC CV: %s", selected_model_name)

    model = calibrate_model(candidates[selected_model_name], method="sigmoid", cv=3)
    model.fit(X_development, y_development)
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    review_threshold, validation_f1 = find_threshold_maximizing_f1(
        y_validation.to_numpy(), validation_probabilities
    )

    now = dt.datetime.now(dt.timezone.utc)
    metadata = {
        "model": selected_model_name,
        "model_version": 1,
        "dataset": "AI4I 2020",
        "features": list(MODEL_FEATURE_CONTRACT),
        "raw_features": list(RAW_SENSOR_FEATURES),
        "engineered_features": list(ENGINEERED_FEATURES),
        "calibration": "sigmoid_cv3",
        "seed": SEED,
        "data_sha256": data_hash,
        "trained_at_utc": now.isoformat(),
        "scikit_learn": sklearn.__version__,
        "split": split_manifest["split_fractions"],
    }
    threshold = {
        "review_threshold": review_threshold,
        "selection_metric": "validation_f1",
        "validation_f1": validation_f1,
    }
    validation_report = {
        "selected_model": selected_model_name,
        "selection_rule": "highest_mean_pr_auc_then_lowest_mean_brier",
        "cv_leaderboard": leaderboard,
        "feature_ablation": feature_ablation(
            selected_model_name,
            X_development,
            y_development,
            leaderboard[selected_model_name],
        ),
        "validation_threshold": threshold,
        "split": split_manifest["split_fractions"],
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    save_json(ARTIFACTS_DIR / "metadata.json", metadata)
    save_json(ARTIFACTS_DIR / "threshold.json", threshold)
    save_json(
        ARTIFACTS_DIR / "reference_ranges.json",
        extract_feature_ranges(X_development),
    )
    save_json(REPORTS_DIR / "validation_metrics.json", validation_report)
    save_json(REPORTS_DIR / "feature_ablation.json", validation_report["feature_ablation"])
    return validation_report


if __name__ == "__main__":
    setup_logging()
    train_model()
