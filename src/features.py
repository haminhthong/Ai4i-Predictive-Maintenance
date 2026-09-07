"""Module tính toán và quản lý đặc trưng dùng chung (Shared Feature Engineering).

ĐẢM BẢO NGUYÊN TẮC:
1. Duy nhất một nơi cài đặt logic biến đổi đặc trưng cho cả Huấn luyện (Train),
   Đánh giá (Evaluate) và Phục vụ trực tuyến (Serving/API).
2. Triệt tiêu hoàn toàn rủi ro Train-Serving Skew.
3. Không thực hiện fuzzy/prefix matching ngầm - sử dụng đúng lược đồ Canonical.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import (
    ENGINEERED_FEATURES,
    MODEL_FEATURE_CONTRACT,
    RAW_TO_CANONICAL_COLUMN_MAP,
)


def canonicalize_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột của DataFrame thô sang tên canonical chuẩn (snake_case).

    Args:
        df: DataFrame chứa các cột thô từ nguồn dữ liệu CSV hoặc hệ thống MES.

    Returns:
        pd.DataFrame: DataFrame có tên cột được ánh xạ chuẩn xác.
    """
    clean_df = df.copy()

    # Tạo bản đồ ánh xạ dựa trên tên cột hiện có
    rename_mapping: dict[str, str] = {}
    for col in clean_df.columns:
        col_clean = col.strip()
        if col_clean in RAW_TO_CANONICAL_COLUMN_MAP:
            rename_mapping[col] = RAW_TO_CANONICAL_COLUMN_MAP[col_clean]
        else:
            # Tìm kiếm trường hợp không phân biệt hoa/thường nếu không khớp trực tiếp
            lower_matched = next(
                (v for k, v in RAW_TO_CANONICAL_COLUMN_MAP.items() if k.lower() == col_clean.lower()),
                None,
            )
            if lower_matched:
                rename_mapping[col] = lower_matched

    clean_df = clean_df.rename(columns=rename_mapping)

    # Chuẩn hóa giá trị cột quality_type nếu có
    if "quality_type" in clean_df.columns:
        clean_df["quality_type"] = clean_df["quality_type"].astype(str).str.strip().str.upper()

    return clean_df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng kỹ thuật & vật lý mở rộng từ các cảm biến cơ bản.

    Các đặc trưng được tính toán:
    1. `temperature_delta_k`: Chênh lệch nhiệt độ giữa quá trình gia công và nhiệt độ không khí (K).
       - Ý nghĩa: Thermal operating-state proxy phản ánh sự tích nhiệt trong buồng máy.
    2. `mechanical_power_w`: Công suất cơ học thực tế của trục quay (Watts).
       - Công thức chuẩn: P = Tau (Nm) * Omega (rad/s) với Omega = RPM * (2 * pi / 60).
    3. `wear_load_interaction`: Tương tác giữa độ mòn dụng cụ và tải trọng mô-men xoắn (min * Nm).
       - Ý nghĩa: Engineering interaction proxy phản ánh áp lực tích lũy khi dao đã mòn.

    Args:
        df: DataFrame chứa tối thiểu các cột cảm biến thô canonical.

    Returns:
        pd.DataFrame: DataFrame kèm thêm 3 cột đặc trưng kỹ thuật mới.
    """
    data = df.copy()

    required_raw = [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]
    missing_cols = [c for c in required_raw if c not in data.columns]
    if missing_cols:
        raise KeyError(
            f"Thiếu các cột cảm biến bắt buộc để tính toán engineered features: {missing_cols}"
        )

    # 1. Thermal operating-state proxy (K)
    data["temperature_delta_k"] = data["process_temperature_k"] - data["air_temperature_k"]

    # 2. Công suất cơ học thực tế (Watts: P = tau * omega)
    angular_velocity = data["rotational_speed_rpm"] * (2.0 * np.pi / 60.0)
    data["mechanical_power_w"] = data["torque_nm"] * angular_velocity

    # 3. Engineering interaction proxy (min * Nm)
    data["wear_load_interaction"] = data["tool_wear_min"] * data["torque_nm"]

    return data


def validate_feature_contract(
    df: pd.DataFrame,
    expected_features: Sequence[str] = MODEL_FEATURE_CONTRACT,
) -> None:
    """Kiểm tra DataFrame có tuân thủ đúng thứ tự và danh sách đặc trưng của Model Contract hay không.

    Raises:
        ValueError: Nếu thiếu cột, thừa cột, hoặc sai thứ tự cột.
    """
    df_columns = list(df.columns)
    expected_list = list(expected_features)

    if df_columns != expected_list:
        missing = [c for c in expected_list if c not in df_columns]
        extra = [c for c in df_columns if c not in expected_list]
        diff_info = []
        if missing:
            diff_info.append(f"Thiếu các cột: {missing}")
        if extra:
            diff_info.append(f"Cột không nằm trong contract: {extra}")
        if not missing and not extra and df_columns != expected_list:
            diff_info.append("Thứ tự cột không khớp với hợp đồng định sẵn.")
        raise ValueError(
            f"DataFrame không tuân thủ Feature Contract! Chi tiết: {'; '.join(diff_info)}"
        )


def build_canonical_features(
    df: pd.DataFrame,
    expected_features: Sequence[str] = MODEL_FEATURE_CONTRACT,
) -> pd.DataFrame:
    """Hàm chuẩn hóa hoàn chỉnh: Chuẩn hóa tên cột -> Tạo đặc trưng -> Sắp xếp đúng Feature Contract.

    Dùng chung thống nhất cho cả Pipeline Huấn luyện và Suy luận.
    """
    clean_df = canonicalize_raw_dataframe(df)

    # Nếu chưa có các cột engineered features, tính toán bổ sung
    if not all(col in clean_df.columns for col in ENGINEERED_FEATURES):
        clean_df = add_engineered_features(clean_df)

    # Chỉ chọn và sắp xếp các cột theo đúng Feature Contract
    missing = [c for c in expected_features if c not in clean_df.columns]
    if missing:
        raise KeyError(f"Không thể xây dựng feature dataframe do thiếu: {missing}")

    result_df = clean_df[list(expected_features)].copy()
    validate_feature_contract(result_df, expected_features)
    return result_df
