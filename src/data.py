"""Module xử lý dữ liệu và tạo đặc trưng (Feature Engineering) cho bài toán Predictive Maintenance.

Module này thực hiện các nhiệm vụ chính:
1. Nạp dữ liệu thô AI4I 2020 từ tệp CSV.
2. Loại bỏ các cột gây rò rỉ dữ liệu (Data Leakage) và các thuộc tính định danh cá thể (UDI, Product ID).
3. Tạo các đặc trưng vật lý bổ sung (Engineered Features) dựa trên tri thức miền (Domain Knowledge).
4. Phân chia dữ liệu theo tỷ lệ phân tầng (Stratified Split) thành 3 tập: Train, Validation và Test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Các tên cột nhãn mục tiêu có thể xuất hiện trong dataset AI4I 2020
TARGET_CANDIDATES = ["Machine failure", "Machine failure ", "machine_failure"]


def add_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Tạo thêm các đặc trưng vật lý biểu diễn trạng thái hoạt động của máy móc.

    Lưu ý: Chỉ sử dụng các thông số cảm biến thu thập tại thời điểm suy luận (Inference),
    không sử dụng bất kỳ nhãn hậu nghiệm nào để tránh Data Leakage.

    Các đặc trưng được tạo:
    - `temperature_delta`: Độ chênh lệch nhiệt độ giữa quá trình vận hành và không khí (K).
    - `power_proxy`: Chỉ số đại diện cho công suất hoạt động (Tốc độ quay x Mô-men xoắn).
    - `strain_proxy`: Tải trọng tích lũy tác động lên công cụ (Độ mòn x Mô-men xoắn).

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

    # 1. Chênh lệch nhiệt độ (Thermal Delta)
    df["temperature_delta"] = df[proc_temp_col] - df[air_temp_col]

    # 2. Công suất cơ học xấp xỉ (Power Proxy: RPM * Nm)
    df["power_proxy"] = df[speed_col] * df[torque_col]

    # 3. Ứng suất cơ học tích lũy lên công cụ (Strain Proxy: min * Nm)
    df["strain_proxy"] = df[wear_col] * df[torque_col]

    return df


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

    # Loại bỏ các cột gây Data Leakage (TWF, HDF, PWF, OSF, RNF) và các thuộc tính ID không chứa giá trị học (UDI, Product ID)
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
