"""Module điều phối quy trình huấn luyện ngoại tuyến (Offline ML Development Pipeline).

Tuân thủ nghiêm ngặt nguyên tắc:
1. HUẤN LUYỆN VÀ ĐÁNH GIÁ CHỈ TRÊN TẬP TRAIN & VALIDATION.
2. Tuyệt đối không chạm vào hay tối ưu hóa bất kỳ thứ gì trên tập Test.
3. Tách biệt hoàn toàn Model Artifact và Decision Policy Artifact.
4. Tối ưu và đóng băng toàn bộ các ngưỡng quyết định (Fixed, Max-F1, Cost-Sensitive, Capacity Constraints)
   ngay tại tập Validation trước khi xuất bản hệ thống.
"""

from __future__ import annotations

import datetime
import platform
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss

from .contracts import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURE_CONTRACT,
    NUMERIC_FEATURES,
)
from .data import (
    audit_dataset,
    compute_dataset_sha256,
    create_or_load_split_registry,
    extract_feature_ranges,
    load_data,
    load_raw_dataset,
)
from .models import get_candidate_models
from .policy import BusinessCosts, tune_all_validation_policies
from .utils import LOGGER, save_json, set_seed, setup_logging

SEED = 42


def get_git_sha() -> str:
    """Lấy mã Git Commit SHA hiện tại để đảm bảo truy xuất nguồn gốc mã nguồn."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "git_unavailable"


def train_and_freeze_system() -> dict[str, Any]:
    """Hàm điều phối toàn bộ quy trình Offline ML Development."""
    set_seed(SEED)
    LOGGER.info("=== BẮT ĐẦU OFFLINE ML DEVELOPMENT PIPELINE ===")

    # 1. Nạp dữ liệu thô & Kiểm toán (Data Audit)
    raw_csv_path = Path("data/raw/ai4i2020.csv")
    data_sha = compute_dataset_sha256(raw_csv_path)
    LOGGER.info(f"Mã băm SHA256 bộ dữ liệu: {data_sha[:12]}...")

    raw_df = load_raw_dataset(raw_csv_path)
    audit_report = audit_dataset(raw_df, sha256_hash=data_sha)
    LOGGER.info(
        f"Kiểm toán dữ liệu hoàn tất: {audit_report['total_observations']} dòng, "
        f"Tỷ lệ hỏng máy: {audit_report['target_summary']['prevalence_percentage']}"
    )

    # 2. Tạo hoặc nạp Split Registry (Train 64%, Val 16%, Test 20%)
    split_registry = create_or_load_split_registry(raw_df, seed=SEED)
    LOGGER.info(
        f"Phân tách dữ liệu: Train={split_registry['split_counts']['train']}, "
        f"Val={split_registry['split_counts']['validation']}, "
        f"Test={split_registry['split_counts']['test']}"
    )

    # 3. Nạp tập Train & Validation chuẩn hóa (Hold-out Test được giữ kín hoàn toàn)
    X_train, X_val, _, y_train, y_val, _ = load_data(path=raw_csv_path, seed=SEED)
    LOGGER.info(f"Ma trận đặc trưng: {X_train.shape[1]} cột canonical: {list(X_train.columns)}")

    # 4. Trích xuất khoảng phân bố tham chiếu trên tập Train phục vụ Distribution Guardrail
    ref_distributions = extract_feature_ranges(X_train)

    # 5. Huấn luyện Model Zoo và lập Validation Leaderboard
    candidate_models = get_candidate_models(seed=SEED)
    leaderboard: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Any] = {}
    val_probs_dict: dict[str, np.ndarray] = {}

    for name, model_pipeline in candidate_models.items():
        LOGGER.info(f"Đang huấn luyện mô hình: {name}...")
        model_pipeline.fit(X_train, y_train)
        probs_val = model_pipeline.predict_proba(X_val)[:, 1]

        fitted_models[name] = model_pipeline
        val_probs_dict[name] = probs_val
        pr_auc = float(average_precision_score(y_val, probs_val))
        brier = float(brier_score_loss(y_val, probs_val))
        leaderboard[name] = {"pr_auc": pr_auc, "brier": brier}
        LOGGER.info(f" -> {name}: PR-AUC = {pr_auc:.4f}, Brier = {brier:.4f}")

    # 6. Chiến lược lựa chọn Champion Model:
    # Ưu tiên PR-AUC cao nhất. Nếu chênh lệch <= 0.01, chọn mô hình có Brier Score thấp nhất (hiệu chuẩn tốt nhất).
    best_pr = max(m["pr_auc"] for m in leaderboard.values())
    top_candidates = [k for k, m in leaderboard.items() if m["pr_auc"] >= (best_pr - 0.01)]
    champion_name = min(top_candidates, key=lambda k: leaderboard[k]["brier"])
    champion_model = fitted_models[champion_name]
    LOGGER.info(f"🏆 CHAMPION MODEL ĐƯỢC CHỌN: '{champion_name}'")

    # 7. Phát triển và ĐÓNG BĂNG Decision Policy trên tập Validation
    val_champion_probs = val_probs_dict[champion_name]
    costs = BusinessCosts(false_negative=5.0, false_positive=1.0)
    decision_policy = tune_all_validation_policies(
        y_val.to_numpy(), val_champion_probs, costs=costs
    )
    LOGGER.info(
        f"Chính sách quyết định được đóng băng: Primary Threshold = {decision_policy['primary_alert_threshold']:.4f}, "
        f"Critical Threshold = {decision_policy['critical_threshold']:.4f}, "
        f"Chi phí kỳ vọng trên Validation = {decision_policy['validation_expected_cost']:.2f} cost units"
    )

    # 8. Tổ chức và lưu trữ Versioned Artifacts
    git_sha = get_git_sha()
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    model_version = f"ai4i-{date_str}-{data_sha[:7]}"

    # Thư mục chính thức: artifacts/champion/
    champ_dir = Path("artifacts/champion")
    champ_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # a. Model Manifest
    model_manifest = {
        "model_version": model_version,
        "model_type": champion_name,
        "framework": "scikit-learn",
        "data_hash": data_sha,
        "git_sha": git_sha,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "feature_contract_version": "ai4i-canonical-v2",
        "validation_metrics": {
            "pr_auc": leaderboard[champion_name]["pr_auc"],
            "brier_score": leaderboard[champion_name]["brier"],
        },
        "calibration_method": "sigmoid_cv3" if "calibrated" in champion_name else "none",
        "runtime_environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }

    # b. Feature Contract
    feature_contract = {
        "contract_version": "ai4i-canonical-v2",
        "features": list(MODEL_FEATURE_CONTRACT),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "expected_dtypes": {
            "quality_type": "string (L, M, H)",
            "air_temperature_k": "float64 (Kelvin)",
            "process_temperature_k": "float64 (Kelvin)",
            "rotational_speed_rpm": "float64 (RPM)",
            "torque_nm": "float64 (Nm)",
            "tool_wear_min": "float64 (Minutes)",
            "temperature_delta_k": "float64 (Kelvin)",
            "mechanical_power_w": "float64 (Watts)",
            "wear_load_interaction": "float64 (min*Nm)",
        },
    }

    # c. Lưu artifacts
    joblib.dump(champion_model, champ_dir / "model.joblib")
    save_json(champ_dir / "model_manifest.json", model_manifest)
    save_json(champ_dir / "feature_contract.json", feature_contract)
    save_json(champ_dir / "decision_policy.json", decision_policy)
    save_json(champ_dir / "reference_distribution.json", ref_distributions)

    # Lưu bản tương thích ngược cho models/
    joblib.dump(champion_model, models_dir / "model.joblib")
    legacy_config = {
        "schema_version": 2,
        "version": model_version,
        "feature_contract_version": "ai4i-canonical-v2",
        "data_sha256": data_sha,
        "selected_model": champion_name,
        "threshold": decision_policy["primary_alert_threshold"],
        "critical_threshold": decision_policy["critical_threshold"],
        "false_negative_cost": costs.false_negative,
        "false_positive_cost": costs.false_positive,
        "seed": SEED,
        "features": list(MODEL_FEATURE_CONTRACT),
        "decision_policy": decision_policy,
        "split_contract": {
            "method": "stratified_random",
            "train_fraction": 0.64,
            "validation_fraction": 0.16,
            "test_fraction": 0.20,
            "limitation": "Snapshot risk classification; không dự báo RUL do thiếu chuỗi thời gian liên tục.",
        },
        "runtime": model_manifest["runtime_environment"],
    }
    save_json(models_dir / "config.json", legacy_config)
    save_json(models_dir / "feature_ranges.json", ref_distributions)

    # d. Validation Report
    validation_report = {
        "model_version": model_version,
        "champion_model": champion_name,
        "leaderboard": leaderboard,
        "validation_pr_auc": leaderboard[champion_name]["pr_auc"],
        "validation_brier": leaderboard[champion_name]["brier"],
        "decision_policy": decision_policy,
    }
    save_json(reports_dir / "validation_metrics.json", validation_report)

    LOGGER.info("=== QUY TRÌNH HUẤN LUYỆN VÀ ĐÓNG BĂNG HOÀN TẤT THÀNH CÔNG ===")
    return validation_report


if __name__ == "__main__":
    setup_logging()
    train_and_freeze_system()
