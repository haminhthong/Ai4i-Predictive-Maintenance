# Feature Contract — AI4I Risk Triage

Tài liệu này là hợp đồng chung giữa `src.data`, `src.features`, `src.train`, `src.inference` và release artifact. Bất kỳ thay đổi tên, thứ tự hoặc công thức feature nào cũng phải cập nhật contract version và chạy lại toàn bộ pipeline.

## 1. Ranh giới dữ liệu

### Được phép đi vào model

Model nhận đúng 9 cột canonical theo thứ tự dưới đây:

| Thứ tự | Feature | Nhóm | Kiểu | Ý nghĩa |
|---:|---|---|---|---|
| 1 | `quality_type` | Raw | categorical | Chất lượng sản phẩm `L`, `M`, `H` |
| 2 | `air_temperature_k` | Raw | numeric | Nhiệt độ không khí, Kelvin |
| 3 | `process_temperature_k` | Raw | numeric | Nhiệt độ quy trình, Kelvin |
| 4 | `rotational_speed_rpm` | Raw | numeric | Tốc độ quay |
| 5 | `torque_nm` | Raw | numeric | Mô-men xoắn |
| 6 | `tool_wear_min` | Raw | numeric | Thời gian mòn dụng cụ |
| 7 | `temperature_delta_k` | Engineered | numeric | Chênh lệch nhiệt độ |
| 8 | `mechanical_power_w` | Engineered | numeric | Công suất cơ học ước tính |
| 9 | `wear_load_interaction` | Engineered | numeric | Tương tác mòn và tải |

Public API dùng `product_quality_type`; `src.features.canonicalize_raw_dataframe()` chuẩn hóa trường này về `quality_type` trước khi tạo feature.

### Bắt buộc loại khỏi model

- Target: `machine_failure`.
- Identifier: `udi`, `product_id` (tương ứng `UDI`, `Product ID`).
- Failure-mode metadata: `failure_twf`, `failure_hdf`, `failure_pwf`, `failure_osf`, `failure_rnf`.
- Runtime metadata: `event_id`, `asset_id`, `event_time`, `line_id`, `sensor_source`, `shift`.

Failure-mode flags là thông tin hậu nghiệm, chỉ được giữ ở test metadata để phân tích slice. Không được đưa chúng vào training, calibration hoặc `/score`.

## 2. Công thức engineered features

```text
temperature_delta_k   = process_temperature_k - air_temperature_k
mechanical_power_w    = torque_nm * rotational_speed_rpm * 2π / 60
wear_load_interaction = tool_wear_min * torque_nm
```

`mechanical_power_w` là đại lượng vật lý suy ra từ mô-men và tốc độ quay. `wear_load_interaction` là engineering proxy để biểu diễn tải trong điều kiện dụng cụ đã mòn; đây không phải causal failure model.

## 3. Quy trình canonical

1. Đọc raw DataFrame.
2. Đổi raw headers và alias API về canonical snake_case.
3. Kiểm tra cột bắt buộc, loại trùng cột alias và kiểm tra `quality_type ∈ {L, M, H}`.
4. Tạo ba engineered features bằng đúng công thức ở trên.
5. Chọn đúng `MODEL_FEATURE_CONTRACT` và đúng thứ tự.
6. Gửi DataFrame đó vào preprocessing pipeline của model.

Training và inference đều gọi cùng builder. Không tạo feature thủ công riêng trong API hoặc dashboard.
Khi raw sensor có mặt, builder luôn tính lại ba engineered features; giá trị engineered do caller gửi lên không được tin cậy.

## 4. Validation và missing data

- Header phải map được vào canonical name.
- `quality_type` không hợp lệ bị từ chối.
- Field số phải chuyển được sang numeric và không được chứa giá trị thiếu sau bước chuẩn hóa.
- API yêu cầu `event_time` có timezone nếu trường này được cung cấp và chuẩn hóa về UTC.
- Nếu release thiếu reference distribution hoặc không verify được checksum, inference chuyển sang `UNAVAILABLE`; hệ thống fail-closed và không phát `NO_ALERT`.

## 5. Leakage boundary

`machine_failure` là target để huấn luyện và đánh giá. `TWF/HDF/PWF/OSF/RNF` chỉ dùng để cắt lát Locked Test sau khi prediction đã hoàn tất. `UDI` và `Product ID` không được dùng vì có thể tạo shortcut theo dòng hoặc sản phẩm.

Contract này mô tả snapshot classification. Không có event-time sequence, censoring, horizon hoặc nhãn RUL; do đó không được diễn giải feature vector như một dự báo tương lai.

## 6. Nguồn kiểm tra

- Canonical constants: `src/contracts.py`.
- Chuẩn hóa và feature builder: `src/features.py`.
- Nạp/split dữ liệu: `src/data.py`.
- Kiểm tra release contract: `src/artifact.py` và `src/inference.py`.
- Test contract: `tests/test_smoke.py` và `tests/test_triage_flow.py`.
