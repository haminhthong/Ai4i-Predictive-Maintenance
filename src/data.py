"""Module xử lý dữ liệu và tạo đặc trưng (Feature Engineering) cho bài toán Machine Failure Risk.

Module này thực hiện các nhiệm vụ chính trong pipeline canonical:
1. Nạp dữ liệu thô AI4I 2020 từ tệp CSV và tính checksum SHA256.
2. Loại bỏ các cột gây rò rỉ dữ liệu (Data Leakage: TWF, HDF, PWF, OSF, RNF) và định danh (UDI, Product ID).
3. Tạo các đặc trưng vật lý bổ sung (Engineered Features) dựa trên cơ học & nhiệt động lực học chuẩn:
   - `temperature_delta`: Chênh lệch nhiệt độ vận hành và không khí.
   - `mechanical_power`: Công suất cơ học thực tế (Watts: Torque * angular_velocity).
   - `wear_load_interaction`: Tương tác tải trọng mòn công cụ (Tool wear x Torque).
4. Phân chia dữ liệu phân tầng (Stratified Random Split): Train 64%, Validation 16%, Test 20%.
5. Trích xuất khoảng phân bố (Quantile bounds P0.5 - P99.5) phục vụ giám sát OOD tại API.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Các tên cột nhãn mục tiêu có thể xuất hiện trong dataset AI4I 2020
TARGET_CANDIDATES = ["Machine failure", "Machine failure ", "machine_failure"]


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


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Tạo thêm các đặc trưng vật lý biểu diễn trạng thái hoạt động của máy móc.

    Lưu ý: Chỉ sử dụng các thông số cảm biến thu thập tại thời điểm suy luận (Inference snapshot),
    không sử dụng bất kỳ nhãn hậu nghiệm nào để tránh Data Leakage.

    Các đặc trưng được tạo:
    - `temperature_delta`: Độ chênh lệch nhiệt độ giữa quá trình vận hành và không khí (K).
    - `mechanical_power`: Công suất cơ học thực tính bằng Watts (P = Torque * angular_velocity, rad/s).
    - `wear_load_interaction`: Tải trọng ma sát tích lũy (Độ mòn x Mô-men xoắn, min*Nm).

    Args:
        features (pd.DataFrame): Dataframe chứa thông số cảm biến ban đầu.

    Returns:
        pd.DataFrame: Dataframe đã bổ sung 3 đặc trưng vật lý mới.
    """
    df = features.copy()

    def get_column_by_prefix(prefix: str) -> str:
        """Tìm chính xác tên cột trong dataframe dựa trên tiền tố không phân biệt chữ hoa/thường."""
        matched = next(
            (col for col in df.columns if col.lower().startswith(prefix.lower())),
            None,
        )
        if matched is None:
            raise KeyError(f"Không tìm thấy cột cảm biến chứa tiền tố: '{prefix}'")
        return matched

    air_temp_col = get_column_by_prefix("air temperature")
    proc_temp_col = get_column_by_prefix("process temperature")
    speed_col = get_column_by_prefix("rotational speed")
    torque_col = get_column_by_prefix("torque")
    wear_col = get_column_by_prefix("tool wear")

    # 1. Chênh lệch nhiệt độ (Thermal Delta - Kelvin)
    df["temperature_delta"] = df[proc_temp_col] - df[air_temp_col]

    # 2. Công suất cơ học thực tế (Mechanical Power in Watts: P = Tau * Omega)
    # Omega (rad/s) = RPM * 2*pi / 60
    angular_velocity = df[speed_col] * (2.0 * np.pi / 60.0)
    df["mechanical_power"] = df[torque_col] * angular_velocity

    # 3. Tải trọng tương tác mòn công cụ (Wear-Load Interaction: min * Nm)
    df["wear_load_interaction"] = df[wear_col] * df[torque_col]

    return df


def extract_feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Trích xuất khoảng giá trị phân bố (Min, Max, P0.5, P99.5) phục vụ kiểm tra OOD."""
    ranges: dict[str, dict[str, float]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            ranges[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "p0_5": float(s.quantile(0.005)),
                "p99_5": float(s.quantile(0.995)),
            }
    return ranges


def load_data(
    path: str | Path = "data/raw/ai4i2020.csv",
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Nạp dữ liệu thô, loại bỏ các cột rò rỉ, tạo đặc trưng và phân chia dataset.

    Args:
        path (str | Path): Đường dẫn tới tệp CSV chứa dữ liệu AI4I 2020.
        seed (int): Random seed dùng cho việc phân chia dữ liệu. Mặc định là 42.

    Returns:
        tuple chứa (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại '{csv_path}'. "
            f"Vui lòng chạy `python scripts/download_data.py` trước khi huấn luyện."
        )

    data = pd.read_csv(csv_path)

    # Xác định cột target trong dataset
    target_col = next((c for c in TARGET_CANDIDATES if c in data.columns), None)
    if target_col is None:
        raise KeyError(
            f"Không tìm thấy cột nhãn target. Danh sách cột hiện tại: {list(data.columns)}"
        )

    labels = data[target_col].astype(int)

    # Loại bỏ 100% các cột gây Data Leakage (TWF, HDF, PWF, OSF, RNF) và các thuộc tính ID (UDI, Product ID)
    drop_columns = [target_col, "TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]
    features = data.drop(columns=[c for c in drop_columns if c in data.columns])

    # Tạo đặc trưng vật lý nâng cao
    features = add_engineered_features(features)

    # Phân chia dữ liệu phân tầng (Stratified Split): 80% (Train + Val), 20% Test
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    # Tiếp tục chia 80% (Train + Val) thành 80% Train (tương đương 64% tổng) và 20% Val (tương đương 16% tổng)
    train_x, val_x, train_y, val_y = train_test_split(
        train_val_x,
        train_val_y,
        test_size=0.2,
        stratify=train_val_y,
        random_state=seed,
    )

    return train_x, val_x, test_x, train_y, val_y, test_y

