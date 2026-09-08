"""Các hàm tiện ích hệ thống: Cấu hình logging, thiết lập random seed cố định và xử lý tệp JSON."""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Khởi tạo logger hệ thống
LOGGER = logging.getLogger("ai_predictive_maintenance")


def normalize_event_time(value: str | None) -> str | None:
    """Chuẩn hóa thời gian có timezone về ISO-8601 UTC dạng hậu tố `Z`."""
    if value is None:
        return None

    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("event_time phải là chuỗi ISO-8601 hợp lệ.") from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("event_time phải kèm timezone, ví dụ `2026-09-07T10:15:00Z`.")

    return (
        parsed.astimezone(timezone.utc)  # noqa: UP017 - tương thích Python 3.10
        .isoformat()
        .replace("+00:00", "Z")
    )


def setup_logging() -> None:
    """Cấu hình định dạng và mức độ ghi log (logging) cho toàn bộ ứng dụng.

    Mức log mặc định là INFO, có thể thay đổi qua biến môi trường LOG_LEVEL.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42) -> None:
    """Cố định giá trị sinh số ngẫu nhiên (random seed) để đảm bảo tính tái lập (reproducibility).

    Args:
        seed (int): Giá trị seed ngẫu nhiên. Mặc định là 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Nếu dự án không dùng PyTorch, ghi log debug và bỏ qua
        LOGGER.debug("Không tìm thấy PyTorch; đã đặt seed cho Python random và NumPy.")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Lưu dữ liệu dạng dictionary vào tệp JSON với định dạng tiếng Việt chuẩn UTF-8.

    Args:
        path (str | Path): Đường dẫn tệp JSON cần lưu.
        payload (dict[str, Any]): Dữ liệu dictionary cần ghi vào tệp.
    """
    file_path = Path(path)
    # Tự động tạo các thư mục cha nếu chưa tồn tại
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(f"Đã lưu thành công dữ liệu JSON tại: {file_path.resolve()}")
