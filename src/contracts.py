"""Hợp đồng dữ liệu dùng chung cho toàn bộ hệ thống.

`quality_type` là đặc trưng chất lượng sản phẩm của AI4I, không phải định danh máy.
Định danh tài sản và thời gian sự kiện chỉ thuộc hợp đồng runtime, không đi vào mô hình.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 1. CỘT DỮ LIỆU THÔ VÀ BẢNG ÁNH XẠ SANG TÊN CHUẨN
# ---------------------------------------------------------------------------

# Ánh xạ tên cột từ file nguồn sang tên chuẩn dạng snake_case.
RAW_TO_CANONICAL_COLUMN_MAP: Final[dict[str, str]] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "quality_type",
    "product_quality_type": "quality_type",
    "Air temperature [K]": "air_temperature_k",
    "Air temperature": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Process temperature": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Rotational speed": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Torque": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Tool wear": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "failure_twf",
    "HDF": "failure_hdf",
    "PWF": "failure_pwf",
    "OSF": "failure_osf",
    "RNF": "failure_rnf",
}

# Cột nhãn mục tiêu chính của bài toán phân loại rủi ro
TARGET_COLUMN: Final[str] = "machine_failure"

# Cột định danh phải loại khỏi đặc trưng để tránh học thuộc dữ liệu hoặc rò rỉ thông tin.
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = ("udi", "product_id")

# Các cờ cơ chế hỏng hóc chỉ biết sau khi máy đã hỏng.
# Chúng không được đưa vào đặc trưng; chỉ dùng để phân tích lỗi trên tập kiểm tra.
FAILURE_MODE_COLUMNS: Final[tuple[str, ...]] = (
    "failure_twf",  # Hỏng do mòn dụng cụ.
    "failure_hdf",  # Hỏng do tản nhiệt kém.
    "failure_pwf",  # Hỏng do công suất.
    "failure_osf",  # Hỏng do quá tải.
    "failure_rnf",  # Hỏng ngẫu nhiên.
)

# Tên tiếng Việt mô tả chi tiết từng cơ chế hỏng hóc.
FAILURE_MODE_DESCRIPTIONS: Final[dict[str, str]] = {
    "failure_twf": "Hỏng do mòn dụng cụ",
    "failure_hdf": "Hỏng do tản nhiệt kém",
    "failure_pwf": "Hỏng do công suất bất thường",
    "failure_osf": "Hỏng do quá tải lực căng hoặc mô-men",
    "failure_rnf": "Hỏng hóc ngẫu nhiên",
}

# ---------------------------------------------------------------------------
# 2. HỢP ĐỒNG ĐẶC TRƯNG ĐẦU VÀO CỦA MÔ HÌNH
# ---------------------------------------------------------------------------

# Các biến vận hành thô được đo tại thời điểm quan sát.
RAW_SENSOR_FEATURES: Final[tuple[str, ...]] = (
    "quality_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

# Các đặc trưng kỹ thuật được tính từ biến vận hành thô.
ENGINEERED_FEATURES: Final[tuple[str, ...]] = (
    "temperature_delta_k",
    "mechanical_power_w",
    "wear_load_interaction",
)

# Danh sách đầy đủ, theo đúng thứ tự mà mô hình nhận vào.
MODEL_FEATURE_CONTRACT: Final[tuple[str, ...]] = (
    *RAW_SENSOR_FEATURES,
    *ENGINEERED_FEATURES,
)

# Phân loại cột số và cột phân loại cho bước tiền xử lý.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("quality_type",)
NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    f for f in MODEL_FEATURE_CONTRACT if f not in CATEGORICAL_FEATURES
)

# Các giá trị hợp lệ của loại chất lượng sản phẩm.
VALID_QUALITY_TYPES: Final[frozenset[str]] = frozenset({"L", "M", "H"})

# Mã dòng tùy chọn để nối kết quả dự đoán với bản ghi batch.
# Trường này không được đưa vào hợp đồng đặc trưng của mô hình.
RUNTIME_METADATA_FIELDS: Final[tuple[str, ...]] = ("record_id",)

# Tên trường công khai dùng cho loại chất lượng sản phẩm trong API.
PUBLIC_PRODUCT_TYPE_FIELD: Final[str] = "product_quality_type"
