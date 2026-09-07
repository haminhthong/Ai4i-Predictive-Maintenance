# 📑 Báo Cáo Phân Tích Chuyên Sâu & Cơ Sở Nghiên Cứu (Research & Engineering Report)

> **Dự án**: AI4I Machine Failure Risk Decision System  
> **Tác giả**: AI Engineer Portfolio Project  
> **Phiên bản**: 2.1.0 | Canonical Architecture  
> **Cập nhật**: 2026-09-07  

---

## 1. Cơ sở Nghiên cứu & Dữ liệu Nền tảng (Literature & Dataset)

- **Tài liệu nghiên cứu gốc**:
  - Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. 3rd International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74. DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)
- **Tập dữ liệu chuẩn**:
  - UCI Machine Learning Repository: *AI4I 2020 Predictive Maintenance Dataset* (Dataset ID: 601).
  - Checksum SHA256: `9b01f38d2d2b860f0ad699bac001875eafbfb6b7159ccd2d746808c73c43e77d`.
  - Quy mô: 10,000 dòng quan sát cảm biến; 339 ca hỏng máy (**3.39%** prevalence), 9,661 ca hoạt động bình thường (**96.61%**).
  - Phân bố phân cấp sản phẩm (`quality_type`): L (Low) chiếm ~50%, M (Medium) chiếm ~30%, H (High) chiếm ~20%.

---

## 2. Kiến trúc Chống Rò rỉ Dữ liệu & Biến đổi Đặc trưng (Anti-Leakage & Features)

### 2.1 Ranh giới Rò rỉ Dữ liệu (Leakage Boundary)
Dataset AI4I 2020 chứa 5 cờ nguyên nhân cụ thể gây hỏng máy:
- `TWF`: Tool Wear Failure
- `HDF`: Heat Dissipation Failure
- `PWF`: Power Failure
- `OSF`: Overstrain Failure
- `RNF`: Random Failure

Các cờ này thu thập **hậu nghiệm** (sau khi máy hỏng). Nếu đưa vào đặc trưng huấn luyện, mô hình sẽ bị rò rỉ dữ liệu nghiêm trọng. Hệ thống loại bỏ 100% các cột này và cột ID khỏi ma trận đặc trưng $X$, chuyển toàn bộ sang **Evaluation Metadata** chỉ phục vụ phân tích lát cắt (slice analysis) sau cùng.

### 2.2 Đặc trưng Kỹ thuật Dẫn xuất (Shared Feature Engineering)
Thực thi duy nhất tại `src/features.py` để triệt tiêu Train-Serving Skew:
1. `mechanical_power_w = torque_nm * (rotational_speed_rpm * 2 * pi / 60)`: Công suất cơ học thực tế tính bằng Watts ($P = \tau\omega$).
2. `temperature_delta_k = process_temperature_k - air_temperature_k`: Thermal operating-state proxy phản ánh mức độ chênh lệch nhiệt buồng gia công và môi trường.
3. `wear_load_interaction = tool_wear_min * torque_nm`: Engineering interaction proxy mô phỏng tương tác giữa độ mòn dao và tải trọng mô-men.

---

## 3. Thực nghiệm Huấn luyện & Chọn Champion (Validation Benchmark)

Phân tầng dữ liệu: Train (6,400 mẫu, 64%), Validation (1,600 mẫu, 16%), Locked Test (2,000 mẫu, 20%). Chỉ số index lưu tại `reports/split_manifest.json`.

### Kết quả Validation Leaderboard:
| Model Candidate | Validation PR-AUC | Validation Brier Score | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Logistic Baseline** | `0.4679` | `0.1136` | Xác suất bị thổi phồng do mất cân bằng lớp |
| **Logistic Sigmoid Calibrated** | `0.4759` | `0.0232` | Brier Score giảm mạnh nhờ Platt scaling |
| **Random Forest Baseline** | `0.8469` | `0.0111` | Khả năng nắm bắt tương tác phi tuyến xuất sắc |
| **RF Sigmoid Calibrated** | **`0.8613`** | **`0.0092`** | 🏆 **Champion Model (PR-AUC cao nhất, Brier thấp nhất)** |
| **HistGB Baseline** | `0.8356` | `0.0105` | Gradient Boosting hiệu quả trên bảng dữ liệu vừa |
| **HistGB Sigmoid Calibrated** | `0.8527` | `0.0098` | Hiệu chuẩn xác suất tốt |

---

## 4. Tối ưu Hóa Ngưỡng Quyết định & Đóng Băng Policy trên Validation

Tất cả các ngưỡng quyết định được tối ưu trên tập Validation với ma trận chi phí tương đối:
$$\text{Cost} = (FN \times 5.0) + (FP \times 1.0)$$

- **Primary Alert Threshold ($\theta^*$)**: `0.3574`
- **Critical Threshold**: `0.4872` (kích hoạt hành động `PRIORITY_REVIEW`)
- **Capacity Constrained Thresholds**:
  - Max 3% alerts: `0.4595`
  - Max 2% alerts: `0.8458`

---

## 5. Đánh giá Độc lập trên Locked Hold-out Test Set (2,000 mẫu)

### 5.1 Chỉ số Tổng thể (Zero-Leakage Protocol)
- **PR-AUC**: `0.8666`
- **ROC-AUC**: `0.9711`
- **Precision**: `88.89%`
- **Recall**: `82.35%`
- **F1-Score**: `0.8550`
- **Brier Score**: `0.0084`
- **Expected Calibration Error (ECE)**: `0.0037` (0.37%)
- **Alert Rate**: `3.15%`
- **Weighted Decision Cost**: `67.00` cost units (`33.50` cost units / 1,000 observations)
- **Confusion Matrix [[TN, FP], [FN, TP]]**: `[[1925, 7], [12, 56]]`

### 5.2 Threshold Strategy Ablation Study
| Chiến lược Threshold | Ngưỡng ($\theta$) | Precision | Recall | F1-Score | Alert Rate | Weighted Cost / 1k |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed 0.50** | `0.5000` | `94.74%` | `79.41%` | `0.8640` | `2.85%` | `36.50` |
| **Max F1 (Frozen Val)** | `0.3574` | `88.89%` | `82.35%` | `0.8550` | `3.15%` | `33.50` |
| **Cost-Sensitive ($\theta^*$)** | **`0.3574`** | **`88.89%`** | **`82.35%`** | **`0.8550`** | **`3.15%`** | **`33.50`** |
| **Capacity Constrained 3%** | `0.4595` | `93.10%` | `79.41%` | `0.8571` | `2.90%` | `37.00` |
| **Capacity Constrained 2%** | `0.8458` | `95.56%` | `63.24%` | `0.7611` | `2.25%` | `63.50` |

---

## 6. Phân tích Lát cắt Cơ chế Hỏng hóc (Failure-Mode Slice Analysis)

Bằng việc đối chiếu với metadata 5 dạng hỏng hóc trong tập Test:

| Cơ chế Hỏng hóc | Số ca trong Test | Phát hiện | Tỷ lệ Recall | Nhận định Kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Heat Dissipation (HDF)** | 29 | 29 | **100.00%** | Mô hình nhận diện hoàn hảo nhờ đặc trưng `temperature_delta_k`. |
| **Overstrain (OSF)** | 16 | 16 | **100.00%** | Bắt trọn 100% ca lỗi nhờ đặc trưng `wear_load_interaction`. |
| **Power Failure (PWF)** | 13 | 12 | **92.31%** | Bắt tốt các đột biến công suất cơ học `mechanical_power_w`. |
| **Tool Wear (TWF)** | 10 | 1 | **10.00%** | Mòn dụng cụ tiệm tiến khó phát hiện qua snapshot đơn lẻ nếu không có tải mô-men cực đoan. |
| **Random Failure (RNF)** | 4 | 0 | **0.00%** | Hỏng hóc ngẫu nhiên theo định nghĩa không mang đặc trưng suy thoái cảm biến. |

### Giải phẫu 12 ca False Negatives (Bỏ sót):
- **9 ca là lỗi Tool Wear (TWF)**: Dao bị mòn vượt ngưỡng tuổi thọ nhưng tại thời điểm snapshot máy đang phay tải nhẹ (mô-men xoắn trung bình $37.82\text{ Nm}$), dẫn đến xác suất rủi ro dự báo chỉ dao động từ $0.15 - 0.30$ (dưới ngưỡng cảnh báo $0.3574$).
- **1 ca Power Failure (PWF)**: Công suất nằm cận biên dao động bình thường.
- **2 ca hỏng hóc khác**.

---

## 7. Kết luận & Khuyến nghị Kỹ thuật

1. **Khắc phục Rò rỉ Ngưỡng Quyết định**: Việc tách rời hoàn toàn khâu tune ngưỡng sang Validation giúp loại bỏ hoàn toàn hiện tượng overfitting và đảm bảo tính khách quan 100% của báo cáo Locked Test.
2. **Giá trị của Đặc trưng Vật lý & Tương tác**: Cả 3 đặc trưng kỹ thuật (`temperature_delta_k`, `mechanical_power_w`, `wear_load_interaction`) đóng vai trò quyết định giúp phát hiện trọn vẹn $100\%$ các ca hỏng hóc do giải nhiệt và quá tải lực.
3. **Phân cấp Rủi ro & Reliability Gate**: Việc bổ sung Distribution Range Guardrails giúp phát hiện sớm các dị thường cảm biến và chuyển trạng thái tin cậy sang `DEGRADED`, giảm thiểu rủi ro vận hành trong môi trường công nghiệp thực tế.
