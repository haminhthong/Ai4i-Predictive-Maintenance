"""Tiện ích ghi log, cố định seed ngẫu nhiên và ghi tệp JSON."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# Bộ ghi log dùng chung cho các module ghi báo cáo và trạng thái pipeline.
LOGGER = logging.getLogger("ai_predictive_maintenance")


def setup_logging() -> None:
    """Cấu hình định dạng và mức độ ghi log cho toàn bộ ứng dụng.

    Mức log mặc định là INFO, có thể thay đổi qua biến môi trường LOG_LEVEL.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Cố định bộ sinh số ngẫu nhiên để kết quả có thể tái lập.

    Args:
        seed (int): Giá trị seed ngẫu nhiên. Mặc định là 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Lưu dữ liệu dạng dictionary vào tệp JSON với định dạng tiếng Việt chuẩn UTF-8.

    Args:
        path (str | Path): Đường dẫn tệp JSON cần lưu.
        payload (dict[str, Any]): Dữ liệu dictionary cần ghi vào tệp.
    """
    file_path = Path(path)
    # Tạo các thư mục cha trước khi ghi nếu chúng chưa tồn tại.
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(f"Đã lưu thành công dữ liệu JSON tại: {file_path.resolve()}")
