"""Package `src`: mã nguồn chính của AI4I Maintenance Risk Triage.

Các module trong package:
- `contracts`: Định nghĩa Data Contracts, canonical snake_case schema, và ranh giới leakage.
- `features`: Shared Feature Engineering dùng chung cho train và serving (triệt tiêu skew).
- `data`: Tầng dữ liệu: nạp dữ liệu thô, audit, tạo/nạp split manifest.
- `models`: Model Zoo (Logistic, RF, HistGB, Sigmoid Calibration), pipeline tiền xử lý.
- `policy`: Chọn threshold F1 và xếp hạng Top-K theo risk.
- `input_validation`: Cảnh báo input ngoài range quan sát.
- `train`: CV, chọn model, calibration và lưu artifact.
- `evaluate`: Đánh giá Test hold-out và phân tích failure modes.
- `inference`: Score snapshot và rank batch từ artifact duy nhất.
- `api`: FastAPI với `/health`, `/score` và `/rank`.
- `utils`: Tiện ích logging, random seed, JSON serialization.
"""

from .utils import LOGGER, save_json, set_seed, setup_logging

__all__ = ["LOGGER", "save_json", "set_seed", "setup_logging"]
__version__ = "1.0.0"
