"""Hợp đồng dữ liệu dùng chung cho toàn bộ hệ thống.

`quality_type` là đặc trưng chất lượng sản phẩm của AI4I, không phải định danh máy.
Định danh tài sản và thời gian sự kiện chỉ thuộc hợp đồng runtime, không đi vào mô hình.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# 1. CÁC CỘT TRONG BỘ DỮ LIỆU THÔ VÀ BẢN ĐỒ CHUYỂN ĐỔI SANG CANONICAL
# ---------------------------------------------------------------------------

# Bản đồ chuẩn hóa tên cột thô (Raw Headers) sang Canonical snake_case names
RAW_TO_CANONICAL_COLUMN_MAP: Final[dict[str, str]] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "quality_type",
    "product_quality_type": "quality_type",
    "product_type": "quality_type",
    "machine_type": "quality_type",
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

# Cột định danh thiết bị - BẮT BUỘC LOẠI BỎ KHỎI FEATURES ĐỂ TRÁNH OVERFITTING / LEAKAGE
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = ("udi", "product_id")

# Các cờ cơ chế hỏng hóc chi tiết (Failure Modes) - LƯU Ý BẢO MẬT:
# Các cờ này là THÔNG TIN HẬU NGHIỆM (chỉ biết sau khi máy đã hỏng).
# TUYỆT ĐỐI KHÔNG ĐƯA VÀO ĐẶC TRƯNG HUẤN LUYỆN HOẶC SUY LUẬN.
# Chỉ được giữ lại riêng biệt làm Metadata phục vụ Error Analysis trên tập Test.
FAILURE_MODE_COLUMNS: Final[tuple[str, ...]] = (
    "failure_twf",  # Tool Wear Failure
    "failure_hdf",  # Heat Dissipation Failure
    "failure_pwf",  # Power Failure
    "failure_osf",  # Overstrain Failure
    "failure_rnf",  # Random Failure
)

# Tên mô tả chi tiết của từng cơ chế hỏng hóc
FAILURE_MODE_DESCRIPTIONS: Final[dict[str, str]] = {
    "failure_twf": "Tool Wear Failure (Hỏng do mòn dụng cụ)",
    "failure_hdf": "Heat Dissipation Failure (Hỏng do giải nhiệt kém)",
    "failure_pwf": "Power Failure (Hỏng do công suất bất thường)",
    "failure_osf": "Overstrain Failure (Hỏng do quá tải lực căng/mô-men)",
    "failure_rnf": "Random Failure (Hỏng hóc ngẫu nhiên)",
}

# ---------------------------------------------------------------------------
# 2. HỢP ĐỒNG ĐẶC TRƯNG ĐẦU VÀO CỦA MÔ HÌNH (FEATURE CONTRACT)
# ---------------------------------------------------------------------------

# Các cảm biến thô hợp lệ đo tại thời điểm quan sát
RAW_SENSOR_FEATURES: Final[tuple[str, ...]] = (
    "quality_type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

# Các đặc trưng dẫn xuất / kỹ thuật (Engineered Features)
ENGINEERED_FEATURES: Final[tuple[str, ...]] = (
    "temperature_delta_k",
    "mechanical_power_w",
    "wear_load_interaction",
)

# Danh sách đầy đủ toàn bộ đặc trưng đi vào mô hình theo ĐÚNG THỨ TỰ CANONICAL
MODEL_FEATURE_CONTRACT: Final[tuple[str, ...]] = (
    *RAW_SENSOR_FEATURES,
    *ENGINEERED_FEATURES,
)

# Phân loại cột số và cột phân loại phục vụ tiền xử lý Pipeline
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("quality_type",)
NUMERIC_FEATURES: Final[tuple[str, ...]] = tuple(
    f for f in MODEL_FEATURE_CONTRACT if f not in CATEGORICAL_FEATURES
)

# Giá trị phân loại hợp lệ của quality_type
VALID_QUALITY_TYPES: Final[frozenset[str]] = frozenset({"L", "M", "H"})

# Một mã dòng tùy chọn để nối prediction về bản ghi batch.
# Trường này không được đưa vào MODEL_FEATURE_CONTRACT.
RUNTIME_METADATA_FIELDS: Final[tuple[str, ...]] = ("record_id",)

# Tên public của trường phân loại sản phẩm trong API.
PUBLIC_PRODUCT_TYPE_FIELD: Final[str] = "product_quality_type"
