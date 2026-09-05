# 📑 Báo Cáo Phân Tích Chuyên Sâu & Cơ Sở Nghiên Cứu (Research & Engineering Report)

> **Dự án**: Machine Failure Risk & Maintenance Decision System (AI4I 2020)  
> **Tác giả**: AI Engineer Portfolio Project  
> **Ngày cập nhật**: 2026-09-05  

---

## 1. Nghiên cứu & Dữ liệu Nền tảng (Literature & Dataset)

- **Tài liệu nghiên cứu gốc**:
  - Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. 3rd International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74. DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)
- **Tập dữ liệu chuẩn**:
  - UCI Machine Learning Repository: *AI4I 2020 Predictive Maintenance Dataset* (Dataset ID: 601).
  - Checksum SHA256 dữ liệu thô: `9b01f38d2d2b860f0ad699bac001875eafbfb6b7159ccd2d746808c73c43e77d`.
  - Tỷ lệ phân bố nhãn: 10.000 mẫu dữ liệu cảm biến; nhãn `Machine failure` chiếm 339 mẫu (~**3.39%**), 9.661 mẫu bình thường (~**96.61%**).

---

## 2. Phân tích Luồng Kỹ thuật & Phương pháp luận (Engineering Methodology)

### 2.1 Chống rò rỉ dữ liệu (Anti-Leakage Architecture)
Trong bộ dữ liệu AI4I 2020 có các biến nhãn phụ:
- `TWF` (Tool Wear Failure)
- `HDF` (Heat Dissipation Failure)
- `PWF` (Power Failure)
- `OSF` (Overstrain Failure)
- `RNF` (Random Failure)

Các biến này mô tả nguyên nhân cụ thể gây hỏng máy thu thập *hậu nghiệm* (chỉ biết sau khi sự cố xảy ra). Nếu đưa các thuộc tính này vào làm đặc trưng huấn luyện, mô hình sẽ đạt ROC-AUC gần như 1.0 ảo do Data Leakage. Pipeline loại bỏ 100% các cột này và thuộc tính ID (`UDI`, `Product ID`), chỉ giữ lại các tín hiệu cảm biến thời gian thực tại thời điểm suy luận.

### 2.2 Tạo đặc trưng vật lý nâng cao (Physics-Based Feature Engineering)
Dựa trên nguyên lý vận hành cơ khí và nhiệt động lực học chuẩn trong nhà máy:
1. `temperature_delta = process_temperature - air_temperature`: Chênh lệch nhiệt độ vận hành và môi trường ($K$), biểu thị mức độ nhiệt ma sát sinh ra.
2. `mechanical_power = torque * (rotational_speed * 2 * pi / 60)`: Công suất cơ học thực tế tính bằng Watts ($P = \tau \cdot \omega$).
3. `wear_load_interaction = tool_wear * torque`: Tương tác tải trọng ma sát cơ học tích lũy lên công cụ theo thời gian ($\text{min} \cdot \text{Nm}$).

---

## 3. So sánh Kết quả Thực nghiệm & Model Zoo Benchmark

### 3.1 Validation Benchmark & Dynamic Champion Selection (Tập Validation 1,600 mẫu)

| Model Candidate | Validation PR-AUC | Validation Brier Score | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Logistic Baseline** | `0.4679` | `0.1136` | Xác suất bị thổi phồng do `class_weight='balanced'` |
| **Logistic Sigmoid Calibrated** | `0.4759` | `0.0232` | Brier Score giảm mạnh 80% nhờ Platt Scaling |
| **Random Forest Baseline** | `0.8469` | `0.0111` | Khả năng phân biệt rủi ro phi tuyến vượt trội |
| **RF Sigmoid Calibrated** | **`0.8613`** | **`0.0092`** | 🏆 **Dynamic Champion (PR-AUC cao nhất & Brier thấp nhất)** |
| **HistGB Baseline** | `0.8356` | `0.0105` | Gradient Boosting hiệu quả trên bảng dữ liệu nhỏ |
| **HistGB Sigmoid Calibrated** | `0.8527` | `0.0098` | Calibration giúp cải thiện độ tin cậy xác suất |

### 3.2 Đánh giá Độc lập nghiệm thu trên tập Test (2,000 mẫu chưa từng thấy)

Mô hình Champion (`rf_sigmoid_calibrated`) đạt các chỉ số ấn tượng trên tập Test độc lập:

- **PR-AUC**: `0.8666`
- **ROC-AUC**: `0.9711`
- **Precision**: `88.89%`
- **Recall**: `82.35%`
- **F1-Score**: `0.8550`
- **Brier Score**: `0.0084`
- **Expected Calibration Error (ECE)**: `0.0037` (0.37%)
- **Alert Rate**: `3.15%`
- **Optimal Cost Threshold ($\theta^*$)**: `0.3574`
- **Expected Business Cost ($FN*5 + FP*1$)**: `$67.00` ($33.50 per 1,000 machines)
- **Confusion Matrix [[TN, FP], [FN, TP]]**: `[[1925, 7], [12, 56]]`

### 3.3 Threshold Strategy Ablation Study trên Tập Test

| Chiến lược Threshold | Ngưỡng ($\theta$) | Precision | Recall | Alert Rate | Test Expected Cost | Cost / 1,000 Machines |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold 0.50** | `0.5000` | `94.74%` | `79.41%` | `2.85%` | `$73.00` | `$36.50` |
| **Max F1 Strategy** | `0.3965` | `91.80%` | `82.35%` | `3.05%` | `$65.00` | `$32.50` |
| **Cost-Sensitive Tuned ($\theta^*$)** | **`0.3574`** | **`88.89%`** | **`82.35%`** | **`3.15%`** | **`$67.00`** | **`$33.50`** |

---

## 4. Tác động Kinh doanh & Kết luận

1. **Tối ưu Chi phí Nghiệp vụ**: Với ngưỡng chi phí tối ưu $0.3574$, hệ thống phát hiện thành công **82.35%** tổng số ca hỏng máy thực tế chỉ với tỷ lệ phát cảnh báo dừng máy rất thấp là **3.15%**, tiết kiệm đáng kể chi phí so với ngưỡng cố định 0.5.
2. **Hiệu chỉnh Xác suất Chuẩn xác**: Chỉ số ECE đạt **0.37%** và Brier Score giảm xuống **0.0084**, đảm bảo xác suất trả về từ API phản ánh chính xác rủi ro thực tế trong nhà máy.
3. **Phân định Cảnh báo**: Phân tách rõ ràng giữa **Mã lý do vận hành (Reason Codes)** (cho kỹ sư vận hành) và **Giải thích đặc trưng mô hình ML (Model Explanations)** (cho AI/MLOps engineer).
