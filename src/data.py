"""Tầng dữ liệu (Data Layer): Nạp dữ liệu thô, Kiểm toán (Audit), Phân tách và Đăng ký Split Registry.

Quy trình quản lý dữ liệu:
1. `load_raw_dataset()`: Nạp tệp CSV thô từ `data/raw/ai4i2020.csv`.
2. `audit_dataset()`: Kiểm tra tính toàn vẹn, thiếu sót, trùng lặp và phân bố tỷ lệ lỗi.
3. `create_or_load_split_registry()`: Lưu trữ hoặc nạp chỉ số phân tầng (Split Registry) cố định.
4. `load_data()`: Trả về tập Development / Policy Validation / Locked Test chuẩn hóa features,
   tách biệt hoàn toàn nhãn mục tiêu và metadata chế độ lỗi (Failure modes).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .contracts import (
    FAILURE_MODE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_CONTRACT,
    TARGET_COLUMN,
)
from .features import build_canonical_features, canonicalize_raw_dataframe
from .utils import LOGGER, save_json

DEFAULT_RAW_DATA_PATH = Path("data/raw/ai4i2020.csv")
DEFAULT_SPLIT_MANIFEST_PATH = Path("reports/split_manifest.json")
DEFAULT_AUDIT_REPORT_PATH = Path("reports/data_audit.json")


def _is_valid_split_registry(
    registry: dict[str, Any], total_rows: int, seed: int
) -> bool:
    """Kiểm tra split manifest đủ index, không chồng lấn và bao phủ toàn bộ dữ liệu."""
    expected_fractions = {
        "development": 0.70,
        "policy_validation": 0.15,
        "locked_test": 0.15,
    }
    expected_counts = {
        "development": len(registry.get("development_indices", [])),
        "policy_validation": len(registry.get("policy_indices", [])),
        "test": len(registry.get("test_indices", [])),
    }
    groups = [
        registry.get("development_indices", []),
        registry.get("policy_indices", []),
        registry.get("test_indices", []),
    ]
    flattened = [index for group in groups for index in group]

    return (
        registry.get("seed") == seed
        and registry.get("split_fractions") == expected_fractions
        and registry.get("split_counts") == expected_counts
        and len(flattened) == total_rows
        and len(set(flattened)) == total_rows
        and set(flattened) == set(range(total_rows))
    )


def compute_dataset_sha256(path: str | Path) -> str:
    """Tính mã băm SHA256 của tệp dữ liệu CSV để đảm bảo tính tái lập (Reproducibility)."""
    csv_path = Path(path)
    if not csv_path.exists():
        return "file_not_found"
    hasher = hashlib.sha256()
    with csv_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_dataset(path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Nạp tệp dữ liệu CSV thô từ đĩa.

    Raises:
        FileNotFoundError: Nếu tệp CSV chưa tồn tại.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu tại '{csv_path}'. "
            f"Vui lòng chạy `python scripts/download_data.py` trước khi tiếp tục."
        )
    return pd.read_csv(csv_path)


def audit_dataset(
    df: pd.DataFrame,
    sha256_hash: str | None = None,
    save_path: str | Path | None = DEFAULT_AUDIT_REPORT_PATH,
) -> dict[str, Any]:
    """Kiểm toán toàn diện bộ dữ liệu AI4I 2020: kiểm tra schema, missing, duplicates, prevalence."""
    clean_df = canonicalize_raw_dataframe(df)

    total_rows = len(clean_df)
    duplicate_rows = int(clean_df.duplicated().sum())
    missing_counts = {
        str(k): int(v) for k, v in clean_df.isnull().sum().items() if v > 0
    }

    # Đếm số lượng máy hỏng và tỷ lệ mắc (Prevalence)
    if TARGET_COLUMN in clean_df.columns:
        target_series = clean_df[TARGET_COLUMN].astype(int)
        failure_count = int(target_series.sum())
        normal_count = total_rows - failure_count
        prevalence = float(target_series.mean())
    else:
        failure_count = 0
        normal_count = total_rows
        prevalence = 0.0

    # Thống kê phân bố các failure mode hậu nghiệm
    failure_modes_summary: dict[str, int] = {}
    for col in FAILURE_MODE_COLUMNS:
        if col in clean_df.columns:
            failure_modes_summary[col] = int(clean_df[col].astype(int).sum())

    # Kiểm tra dải giá trị cảm biến cơ bản
    feature_ranges: dict[str, dict[str, float]] = {}
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        s = clean_df[c].dropna()
        feature_ranges[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }

    audit_report = {
        "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
        "total_observations": total_rows,
        "duplicate_rows": duplicate_rows,
        "missing_values_count": missing_counts,
        "has_missing_values": len(missing_counts) > 0,
        "raw_sha256": sha256_hash or "unknown",
        "target_column": TARGET_COLUMN,
        "target_summary": {
            "failure_samples": failure_count,
            "normal_samples": normal_count,
            "prevalence_rate": prevalence,
            "prevalence_percentage": f"{prevalence * 100:.2f}%",
        },
        "latent_failure_modes_count": failure_modes_summary,
        "feature_ranges_raw": feature_ranges,
        "leakage_isolation": {
            "identifier_columns_dropped": list(IDENTIFIER_COLUMNS),
            "failure_modes_quarantined_for_eval_only": list(FAILURE_MODE_COLUMNS),
        },
    }

    if save_path is not None:
        save_json(save_path, audit_report)
        LOGGER.info(f"Đã lưu báo cáo Data Audit tại: {save_path}")

    return audit_report


def create_or_load_split_registry(
    df: pd.DataFrame,
    seed: int = 42,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Tạo hoặc nạp registry Development 70%, Policy 15%, Locked Test 15%.

    Development là nơi dùng Stratified CV để so sánh mô hình và feature contract.
    Policy Validation chỉ dùng để đóng băng ngưỡng/xếp hạng vận hành.
    Locked Test chỉ dùng để báo cáo cuối cùng.
    """
    path = Path(manifest_path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                registry = json.load(f)
            required_keys = {"development_indices", "policy_indices", "test_indices"}
            if required_keys <= set(registry.keys()) and _is_valid_split_registry(
                registry, total_rows=len(df), seed=seed
            ):
                return registry
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning(f"Không thể đọc manifest hiện có, tạo lại: {exc}")

    clean_df = canonicalize_raw_dataframe(df)
    labels = clean_df[TARGET_COLUMN].astype(int).to_numpy()
    indices = np.arange(len(clean_df))

    # Tách Locked Test (15%) trước để không tham gia bất kỳ quyết định nào.
    development_policy_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=labels[indices],
        random_state=seed,
    )

    # Tách Policy Validation (15% tổng) khỏi Development (70% tổng).
    development_idx, policy_idx = train_test_split(
        development_policy_idx,
        test_size=(0.15 / 0.85),
        stratify=labels[development_policy_idx],
        random_state=seed,
    )

    registry = {
        "seed": seed,
        "stratification_target": TARGET_COLUMN,
        "total_samples": len(clean_df),
        "split_fractions": {
            "development": 0.70,
            "policy_validation": 0.15,
            "locked_test": 0.15,
        },
        "split_counts": {
            "development": len(development_idx),
            "policy_validation": len(policy_idx),
            "test": len(test_idx),
        },
        "development_indices": [int(i) for i in development_idx],
        "policy_indices": [int(i) for i in policy_idx],
        "test_indices": [int(i) for i in test_idx],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(path, registry)
    LOGGER.info(f"Đã tạo và lưu Split Manifest cố định tại: {path}")
    return registry


def extract_feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Trích xuất khoảng giá trị phân bố (Min, Max, P0.5, P99.5) phục vụ phân tích guardrail."""
    ranges: dict[str, dict[str, float]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            ranges[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p0_5": float(s.quantile(0.005)),
                "p99_5": float(s.quantile(0.995)),
            }
    return ranges


def load_data(
    path: str | Path = DEFAULT_RAW_DATA_PATH,
    seed: int = 42,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    return_metadata: bool = False,
) -> tuple[Any, ...]:
    """Nạp dữ liệu và chia theo Development / Policy Validation / Locked Test.

    Returns:
        Nếu return_metadata=False:
            (X_development, X_policy, X_test, y_development, y_policy, y_test)
        Nếu return_metadata=True:
            (X_development, X_policy, X_test, y_development, y_policy, y_test, metadata_test)
            trong đó metadata_test là DataFrame chứa các failure mode hậu nghiệm (TWF, HDF, PWF, OSF, RNF)
            của riêng tập Test phục vụ Error Analysis.
    """
    raw_df = load_raw_dataset(path)
    clean_df = canonicalize_raw_dataframe(raw_df)

    if TARGET_COLUMN not in clean_df.columns:
        raise KeyError(f"Dataset thiếu cột target bắt buộc: {TARGET_COLUMN}")

    # 1. Tách nhãn chính
    labels = clean_df[TARGET_COLUMN].astype(int)
    if not labels.isin({0, 1}).all():
        raise ValueError("Target machine_failure chỉ được chứa giá trị 0 hoặc 1.")

    # 2. Tách metadata chế độ hỏng hóc (dành riêng cho error analysis)
    metadata_cols = [c for c in FAILURE_MODE_COLUMNS if c in clean_df.columns]
    modes_df = (
        clean_df[metadata_cols].copy()
        if metadata_cols
        else pd.DataFrame(index=clean_df.index)
    )

    # 3. Xây dựng feature matrix theo đúng Shared Feature Contract
    # Loại bỏ hoàn toàn target, id, và failure modes khỏi features
    features_df = build_canonical_features(
        clean_df, expected_features=MODEL_FEATURE_CONTRACT
    )

    # 4. Nạp hoặc sinh Split Registry
    registry = create_or_load_split_registry(
        raw_df, seed=seed, manifest_path=manifest_path
    )

    development_idx = registry["development_indices"]
    policy_idx = registry["policy_indices"]
    test_idx = registry["test_indices"]

    X_development = features_df.iloc[development_idx].reset_index(drop=True)
    X_policy = features_df.iloc[policy_idx].reset_index(drop=True)
    X_test = features_df.iloc[test_idx].reset_index(drop=True)

    y_development = labels.iloc[development_idx].reset_index(drop=True)
    y_policy = labels.iloc[policy_idx].reset_index(drop=True)
    y_test = labels.iloc[test_idx].reset_index(drop=True)

    if return_metadata:
        metadata_test = modes_df.iloc[test_idx].reset_index(drop=True)
        return (
            X_development,
            X_policy,
            X_test,
            y_development,
            y_policy,
            y_test,
            metadata_test,
        )

    return X_development, X_policy, X_test, y_development, y_policy, y_test
