"""Package `src`: mã nguồn chính của AI4I Maintenance Risk Triage.

Các module trong package:
- `contracts`: Định nghĩa Data Contracts, canonical snake_case schema, và ranh giới leakage.
- `features`: Tính đặc trưng dùng chung cho huấn luyện và phục vụ mô hình.
- `data`: Tầng dữ liệu: nạp dữ liệu thô, audit, tạo/nạp split manifest.
- `models`: Các mô hình Logistic, RF, HistGB và hiệu chỉnh sigmoid.
- `policy`: Chọn ngưỡng F1 và xếp hạng Top-K theo risk.
- `input_validation`: Cảnh báo input ngoài range quan sát.
- `train`: Đánh giá chéo, chọn mô hình, hiệu chỉnh và lưu artifact.
- `evaluate`: Đánh giá Test hold-out và phân tích cơ chế hỏng.
- `inference`: Chấm điểm snapshot và xếp hạng batch từ artifact duy nhất.
- `api`: FastAPI với `/health`, `/score` và `/rank`.
- `utils`: Tiện ích ghi log, cố định seed và ghi JSON.
"""

from .utils import LOGGER, save_json, set_seed, setup_logging

__all__ = ["LOGGER", "save_json", "set_seed", "setup_logging"]
__version__ = "1.0.0"
