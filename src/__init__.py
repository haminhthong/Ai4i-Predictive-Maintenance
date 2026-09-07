"""Package `src`: Mã nguồn chính của hệ thống AI4I Machine Failure Risk Decision System.

Các module trong package:
- `contracts`: Định nghĩa Data Contracts, canonical snake_case schema, và ranh giới leakage.
- `features`: Shared Feature Engineering dùng chung cho train và serving (triệt tiêu skew).
- `data`: Tầng dữ liệu: nạp dữ liệu thô, audit, tạo/nạp split manifest.
- `models`: Model Zoo (Logistic, RF, HistGB, Sigmoid Calibration), pipeline tiền xử lý.
- `policy`: Tối ưu hóa ngưỡng quyết định theo chi phí và ràng buộc công suất.
- `train`: Điều phối quy trình huấn luyện ngoại tuyến trên Train + Validation.
- `evaluate`: Đánh giá độc lập trên Locked Test Set và phân tích lát cắt failure modes.
- `inference`: Động cơ suy luận (RiskInferenceService) tích hợp Reliability Gate và 4-block schema.
- `monitoring`: Giám sát dải phân bố (Distribution Range Guardrail) và chỉ số PSI/Workload.
- `api`: FastAPI RESTful Service.
- `utils`: Tiện ích logging, random seed, JSON serialization.
"""

from .utils import LOGGER, save_json, set_seed, setup_logging

__all__ = ["LOGGER", "save_json", "set_seed", "setup_logging"]
__version__ = "2.1.0"