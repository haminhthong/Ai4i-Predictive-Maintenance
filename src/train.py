"""Huấn luyện và đóng gói release cho hệ thống condition-based triage."""

from __future__ import annotations

import datetime
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .artifact import sha256_file
from .contracts import CATEGORICAL_FEATURES, MODEL_FEATURE_CONTRACT, NUMERIC_FEATURES
from .data import (
    audit_dataset,
    compute_dataset_sha256,
    create_or_load_split_registry,
    extract_feature_ranges,
    load_data,
    load_raw_dataset,
)
from .models import calibrate_model, get_candidate_models
from .policy import BusinessCosts, tune_all_validation_policies
from .utils import LOGGER, save_json, set_seed, setup_logging

SEED = 42
RELEASES_DIR = Path("releases")


def get_git_sha() -> str:
    """Lấy commit hiện tại để truy xuất nguồn gốc release."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return "git_unavailable"
    try:
        output = subprocess.check_output(  # noqa: S603 - executable được xác thực bằng shutil.which
            [git_executable, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return "git_unavailable"


def _cross_validate_candidates(
    candidates: dict[str, Any], X_development: Any, y_development: Any
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """So sánh base model bằng OOF predictions chỉ trên Development."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    leaderboard: dict[str, dict[str, float]] = {}
    oof_probabilities: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        LOGGER.info("Cross-validation mô hình: %s", name)
        probabilities = cross_val_predict(
            model,
            X_development,
            y_development,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        oof_probabilities[name] = probabilities
        leaderboard[name] = {
            "pr_auc_oof": float(average_precision_score(y_development, probabilities)),
            "brier_oof": float(brier_score_loss(y_development, probabilities)),
        }
    return leaderboard, oof_probabilities


def _choose_champion(leaderboard: dict[str, dict[str, float]]) -> str:
    """Chọn model từ OOF score, không dùng Policy Validation hoặc Locked Test."""
    best_pr_auc = max(item["pr_auc_oof"] for item in leaderboard.values())
    candidates = [
        name
        for name, item in leaderboard.items()
        if item["pr_auc_oof"] >= best_pr_auc - 0.01
    ]
    # Random Forest là production candidate đã được chọn trước; chỉ nhường chỗ
    # nếu chênh lệch PR-AUC vượt tolerance 0.01.
    if "random_forest" in candidates:
        return "random_forest"
    return min(candidates, key=lambda name: leaderboard[name]["brier_oof"])


def _write_release(
    release_dir: Path,
    model: Any,
    model_config: dict[str, Any],
    contract: dict[str, Any],
    calibration: dict[str, Any],
    policy: dict[str, Any],
    reference_distribution: dict[str, Any],
    data_manifest: dict[str, Any],
    validation_metrics: dict[str, Any],
    model_card: str,
) -> dict[str, Any]:
    """Ghi release self-contained rồi tạo manifest hash các file thành phần."""
    release_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, release_dir / "model.joblib")
    save_json(release_dir / "model_config.json", model_config)
    save_json(release_dir / "feature_contract.json", contract)
    save_json(release_dir / "calibration.json", calibration)
    save_json(release_dir / "decision_policy.json", policy)
    save_json(release_dir / "reference_distribution.json", reference_distribution)
    save_json(release_dir / "data_manifest.json", data_manifest)
    save_json(release_dir / "validation_metrics.json", validation_metrics)
    save_json(release_dir / "locked_test_metrics.json", {
        "status": "pending_locked_test_evaluation",
        "evaluation_protocol": "report_only_after_policy_freeze",
    })
    (release_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")

    file_hashes = {
        path.name: sha256_file(path)
        for path in release_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "manifest_version": 1,
        "release_version": model_config["model_version"],
        "model_version": model_config["model_version"],
        "model_type": model_config["model_type"],
        "file_hashes": file_hashes,
        "model_sha256": file_hashes["model.joblib"],
        "feature_contract_sha256": file_hashes["feature_contract.json"],
        "policy_sha256": file_hashes["decision_policy.json"],
        "reference_distribution_sha256": file_hashes["reference_distribution.json"],
        "git_sha": model_config["git_sha"],
    }
    save_json(release_dir / "manifest.json", manifest)
    return manifest


def _write_legacy_mirrors(
    model: Any,
    release_dir: Path,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    contract: dict[str, Any],
    policy: dict[str, Any],
    reference_distribution: dict[str, Any],
) -> None:
    """Giữ bản mirror tạm thời để client cũ migrate dần sang releases/."""
    champion_dir = Path("artifacts/champion")
    models_dir = Path("models")
    champion_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, champion_dir / "model.joblib")
    joblib.dump(model, models_dir / "model.joblib")
    save_json(champion_dir / "model_manifest.json", model_config)
    save_json(champion_dir / "feature_contract.json", contract)
    save_json(champion_dir / "decision_policy.json", policy)
    save_json(champion_dir / "reference_distribution.json", reference_distribution)
    legacy_config = {
        "schema_version": 3,
        "version": model_config["model_version"],
        "selected_model": model_config["model_type"],
        "feature_contract_version": contract["contract_version"],
        "decision_policy": policy,
        "features": contract["features"],
        "release_dir": str(release_dir),
        "manifest_sha256": sha256_file(release_dir / "manifest.json"),
    }
    save_json(models_dir / "config.json", legacy_config)
    save_json(models_dir / "feature_ranges.json", reference_distribution)


def train_and_freeze_system() -> dict[str, Any]:
    """Chạy Development CV -> Policy Validation -> đóng gói release."""
    set_seed(SEED)
    raw_path = Path("data/raw/ai4i2020.csv")
    data_hash = compute_dataset_sha256(raw_path)
    raw_df = load_raw_dataset(raw_path)
    audit_dataset(raw_df, sha256_hash=data_hash)
    split_registry = create_or_load_split_registry(raw_df, seed=SEED)
    LOGGER.info("Split: %s", split_registry["split_counts"])

    X_development, X_policy, _, y_development, y_policy, _ = load_data(
        path=raw_path, seed=SEED
    )
    reference_distribution = extract_feature_ranges(X_development)

    candidates = get_candidate_models(seed=SEED)
    leaderboard, _ = _cross_validate_candidates(candidates, X_development, y_development)
    champion_name = _choose_champion(leaderboard)
    LOGGER.info("Champion được chọn từ Development CV: %s", champion_name)

    # Calibration là stage riêng, chỉ fit sau khi base model đã được chọn.
    calibrated_model = calibrate_model(get_candidate_models(seed=SEED)[champion_name], cv=3)
    calibrated_model.fit(X_development, y_development)
    policy_probabilities = calibrated_model.predict_proba(X_policy)[:, 1]
    costs = BusinessCosts(false_negative=5.0, false_positive=1.0)
    decision_policy = tune_all_validation_policies(
        y_policy.to_numpy(), policy_probabilities, costs=costs
    )
    decision_policy["selected_on"] = "policy_validation_only"
    decision_policy["queue_strategy"] = "latest_valid_event_per_asset_then_top_k"

    now = datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017 - tương thích Python 3.10
    model_version = f"ai4i-risk-v3.0.0-{now.strftime('%Y%m%d%H%M%S')}-{data_hash[:7]}"
    model_config = {
        "model_version": model_version,
        "model_type": champion_name,
        "calibration": "sigmoid_cv3",
        "framework": "scikit-learn",
        "data_hash": data_hash,
        "git_sha": get_git_sha(),
        "created_at_utc": now.isoformat(),
        "runtime_environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    contract = {
        "contract_version": "ai4i-canonical-v3",
        "features": list(MODEL_FEATURE_CONTRACT),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "runtime_metadata_excluded_from_model": [
            "event_id", "asset_id", "event_time", "line_id", "sensor_source", "shift"
        ],
    }
    validation_metrics = {
        "protocol": "70_percent_development_5_fold_stratified_cv_15_percent_policy_validation",
        "champion_model": champion_name,
        "cv_leaderboard": leaderboard,
        "policy_validation": decision_policy,
    }
    data_manifest = {
        "raw_sha256": data_hash,
        "split_contract": split_registry["split_fractions"],
        "split_counts": split_registry["split_counts"],
        "generalization_claim": "i.i.d. snapshot generalization under the AI4I benchmark distribution",
        "temporal_claim": "none; AI4I has no production event-time trajectory contract",
    }
    calibration = {"method": "sigmoid", "cv": 3, "fit_scope": "development_only"}
    model_card = f"""# AI4I Condition-Based Maintenance Risk Triage\n\n- Release: `{model_version}`\n- Model: `{champion_name}` + sigmoid cross-fitted calibration\n- Risk meaning: risk associated with the current operating snapshot.\n- No future horizon, RUL, or failure time is implied.\n- Failure-mode flags are target-derived diagnostic labels used only for offline slice analysis.\n- RNF is an out-of-model limitation because the available sensor contract has no reliable precursor.\n- Tool-wear failures require explicit monitoring; overall PR-AUC must not hide this slice.\n"""

    release_dir = RELEASES_DIR / model_version
    manifest = _write_release(
        release_dir,
        calibrated_model,
        model_config,
        contract,
        calibration,
        decision_policy,
        reference_distribution,
        data_manifest,
        validation_metrics,
        model_card,
    )
    try:
        _write_legacy_mirrors(
            calibrated_model,
            release_dir,
            manifest,
            model_config,
            contract,
            decision_policy,
            reference_distribution,
        )
    except PermissionError:
        # Release mới là nguồn chuẩn; mirror cũ có thể bị khóa bởi tiến trình cũ.
        LOGGER.warning("Không ghi được mirror legacy; release bundle vẫn hợp lệ.")
    try:
        save_json(Path("reports/validation_metrics.json"), validation_metrics)
    except PermissionError:
        LOGGER.warning("Không ghi được báo cáo validation legacy; bản chuẩn nằm trong release.")
    LOGGER.info("Đã tạo release bundle: %s", release_dir)
    return validation_metrics


if __name__ == "__main__":
    setup_logging()
    train_and_freeze_system()
