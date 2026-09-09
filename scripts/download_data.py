"""Script tự động tải bộ dữ liệu AI4I 2020 Predictive Maintenance Dataset từ UCI Repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utils import LOGGER, setup_logging


def download_ai4i_dataset(output_dir: str | Path = "data/raw") -> Path:
    """Tải tập dữ liệu AI4I 2020 từ UCI ML Repository và lưu vào tệp CSV thô.

    Args:
        output_dir (str | Path): Thư mục lưu tệp dữ liệu CSV thô.

    Returns:
        Path: Đường dẫn tuyệt đối tới tệp CSV đã tạo.
    """
    setup_logging()
    LOGGER.info(
        "Bắt đầu tải bộ dữ liệu AI4I 2020 Predictive Maintenance Dataset (ID=601) từ UCI ML Repository..."
    )

    # Fetch dữ liệu từ UCI ML Repo (ID: 601)
    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Thêm các cột target vào chung 1 Dataframe
    full_df = pd.concat([X, y], axis=1)

    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    file_path = target_path / "ai4i2020.csv"

    full_df.to_csv(file_path, index=False)
    LOGGER.info(
        f"Tải thành công! Kích thước dữ liệu: {full_df.shape[0]} dòng, {full_df.shape[1]} cột. "
        f"Lưu tại: {file_path.resolve()}"
    )

    return file_path.resolve()


if __name__ == "__main__":
    download_ai4i_dataset()
