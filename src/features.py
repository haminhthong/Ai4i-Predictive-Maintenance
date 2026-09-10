"""Tính toán và quản lý đặc trưng dùng chung.

ĐẢM BẢO NGUYÊN TẮC:
1. Chỉ có một nơi cài đặt logic biến đổi cho huấn luyện, đánh giá và phục vụ API.
2. Dùng cùng một hợp đồng đặc trưng để tránh lệch giữa huấn luyện và phục vụ.
3. Không tự động đối chiếu mơ hồ theo tiền tố; chỉ dùng lược đồ chuẩn đã khai báo.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .contracts import (
    ENGINEERED_FEATURES,
    MODEL_FEATURE_CONTRACT,
    RAW_SENSOR_FEATURES,
    RAW_TO_CANONICAL_COLUMN_MAP,
    VALID_QUALITY_TYPES,
)


def canonicalize_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột của DataFrame thô sang tên canonical chuẩn (snake_case).

    Args:
        df: DataFrame chứa các cột thô từ nguồn dữ liệu CSV hoặc hệ thống MES.

    Returns:
        pd.DataFrame: DataFrame có tên cột được ánh xạ chuẩn xác.
    """
    clean_df = df.copy()

    # Tạo bảng ánh xạ từ các tên cột thực tế trong DataFrame.
    rename_mapping: dict[str, str] = {}
    for col in clean_df.columns:
        col_clean = col.strip()
        if col_clean in RAW_TO_CANONICAL_COLUMN_MAP:
            rename_mapping[col] = RAW_TO_CANONICAL_COLUMN_MAP[col_clean]
        else:
            # Thử đối chiếu không phân biệt hoa thường nếu chưa khớp trực tiếp.
            lower_matched = next(
                (
                    v
                    for k, v in RAW_TO_CANONICAL_COLUMN_MAP.items()
                    if k.lower() == col_clean.lower()
                ),
                None,
            )
            if lower_matched:
                rename_mapping[col] = lower_matched

    clean_df = clean_df.rename(columns=rename_mapping)

    # Xử lý trường hợp dữ liệu đầu vào có nhiều bí danh cho cùng một trường.
    # Chọn giá trị khác rỗng đầu tiên, đồng thời báo lỗi nếu các giá trị xung đột.
    if clean_df.columns.duplicated().any():
        merged = pd.DataFrame(index=clean_df.index)
        for column in dict.fromkeys(clean_df.columns):
            duplicate_values = clean_df.loc[:, clean_df.columns == column]
            non_null_values = duplicate_values.dropna(axis=1, how="all")
            if non_null_values.shape[1] > 1:
                conflicting_rows = non_null_values.nunique(axis=1, dropna=True) > 1
                if conflicting_rows.any():
                    raise ValueError(
                        f"Các alias của cột '{column}' chứa giá trị mâu thuẫn "
                        f"tại dòng: {list(conflicting_rows[conflicting_rows].index[:5])}."
                    )
            merged[column] = duplicate_values.bfill(axis=1).iloc[:, 0]
        clean_df = merged

    # Chuẩn hóa giá trị loại chất lượng sản phẩm nếu cột này tồn tại.
    if "quality_type" in clean_df.columns:
        clean_df["quality_type"] = clean_df["quality_type"].astype(str).str.strip().str.upper()
        invalid_types = sorted(set(clean_df["quality_type"].dropna()) - set(VALID_QUALITY_TYPES))
        if invalid_types:
            raise ValueError(f"quality_type không hợp lệ: {invalid_types}; chỉ nhận L, M hoặc H.")

    return clean_df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng kỹ thuật & vật lý mở rộng từ các cảm biến cơ bản.

    Các đặc trưng được tính toán:
    1. `temperature_delta_k`: Chênh lệch nhiệt độ giữa quá trình gia công và nhiệt độ không khí (K).
       - Ý nghĩa: chỉ dấu trạng thái nhiệt khi vận hành.
    2. `mechanical_power_w`: Công suất cơ học thực tế của trục quay (Watts).
       - Công thức: P = mô-men xoắn (Nm) * tốc độ góc (rad/s), với tốc độ góc = RPM * (2 * pi / 60).
    3. `wear_load_interaction`: Tương tác giữa độ mòn dụng cụ và tải trọng mô-men xoắn (min * Nm).
       - Ý nghĩa: chỉ dấu tương tác kỹ thuật giữa độ mòn và tải trọng.

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

    # 1. Chỉ dấu trạng thái nhiệt khi vận hành (K).
    data["temperature_delta_k"] = data["process_temperature_k"] - data["air_temperature_k"]

    # 2. Công suất cơ học (W): P = mô-men xoắn * tốc độ góc.
    angular_velocity = data["rotational_speed_rpm"] * (2.0 * np.pi / 60.0)
    data["mechanical_power_w"] = data["torque_nm"] * angular_velocity

    # 3. Chỉ dấu tương tác giữa độ mòn và tải trọng (phút * Nm).
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

    # Khi có đủ biến thô, luôn tính lại đặc trưng dẫn xuất để tránh lệch giữa
    # lúc huấn luyện và lúc phục vụ, hoặc nhận giá trị dẫn xuất do bên gọi tự gửi.
    has_raw_sensor = all(col in clean_df.columns for col in RAW_SENSOR_FEATURES)
    if has_raw_sensor or not all(col in clean_df.columns for col in ENGINEERED_FEATURES):
        clean_df = add_engineered_features(clean_df)

    # Chỉ chọn và sắp xếp cột theo đúng hợp đồng đặc trưng.
    missing = [c for c in expected_features if c not in clean_df.columns]
    if missing:
        raise KeyError(f"Không thể xây dựng feature dataframe do thiếu: {missing}")

    result_df = clean_df[list(expected_features)].copy()
    validate_feature_contract(result_df, expected_features)
    return result_df
