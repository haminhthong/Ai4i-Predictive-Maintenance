# 📑 Báo Cáo Phân Tích Chuyên Sâu & Cơ Sở Nghiên Cứu (Research & Engineering Report)

> **Dự án**: Predictive Maintenance & Failure Risk Service (AI4I 2020)  
> **Tác giả**: AI Engineer Portfolio Project  
> **Ngày cập nhật**: 2026-08-30  

---

## 1. Nghiên cứu & Dữ liệu Nền tảng (Literature & Dataset)

- **Tài liệu nghiên cứu gốc**:
  - Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*. 3rd International Conference on Artificial Intelligence for Industries (AI4I), pp. 69-74. DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)
- **Tập dữ liệu chuẩn**:
  - UCI Machine Learning Repository: *AI4I 2020 Predictive Maintenance Dataset* (Dataset ID: 601).
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

Các biến này mô tả nguyên nhân cụ thể gây hỏng máy thu thập *hậu nghiệm* (sau khi sự cố xảy ra). Nếu đưa các thuộc tính này vào làm đặc trưng huấn luyện, mô hình sẽ đạt ROC-AUC gần như 1.0 ảo do Data Leakage. Pipeline loại bỏ 100% các cột này, chỉ giữ lại các tín hiệu cảm biến thời gian thực.

### 2.2 Tạo đặc trưng vật lý nâng cao (Physics-Based Feature Engineering)
Dựa trên nguyên lý vận hành cơ khí và nhiệt động lực học trong nhà máy:
1. `temperature_delta = process_temperature - air_temperature`: Biểu thị lượng nhiệt sinh ra trong quá trình ma sát vận hành.
2. `power_proxy = rotational_speed * torque`: Đại diện cho công suất cơ học đầu ra.
3. `strain_proxy = tool_wear * torque`: Đại diện cho ứng suất lực tích lũy theo thời gian lên mũi khoan/công cụ.

---

## 3. So sánh Kết quả Thực nghiệm (Experimental Results Comparison)

### 3.1 Giai đoạn 1: Baseline Logistic Regression
- **Phương pháp**: Stratified Split 80/20/20, `LogisticRegression(class_weight='balanced')`.
- **Kết quả Validation**: PR-AUC = 0.4679, Brier Score = **0.1136**.
- **Hạn chế**: Do sử dụng `class_weight='balanced'`, mô hình dự báo xác suất bị thổi phồng, Brier Score cao (xác suất chưa được hiệu chỉnh).

### 3.2 Giai đoạn 2: Nâng cấp Sigmoid Calibration & Cost Threshold Tuning (v2)
- **Phương pháp**: 
  - Thêm đặc trưng vật lý `temperature_delta`, `power_proxy`, `strain_proxy`.
  - Hiệu chỉnh xác suất bằng `CalibratedClassifierCV(method='sigmoid', cv=3)` (Platt Scaling).
  - Tối ưu ngưỡng phát cảnh báo dựa trên hàm chi phí kinh doanh ($FN:FP = 5:1$).
- **Kết quả Validation**: PR-AUC = **0.4759**, Brier Score giảm mạnh còn **0.0232** (giảm ~80% sai số xác suất).
- **Kết quả nghiệm thu độc lập trên tập Test (2.000 mẫu)**:
  - **PR-AUC**: `0.4379`
  - **ROC-AUC**: `0.9349`
  - **Precision**: `0.4250`
  - **Recall**: `0.5000`
  - **F1-Score**: `0.4595`
  - **Brier Score**: `0.0245`
  - **Alert Rate**: `4.00%`
  - **Optimal Threshold**: `0.1824`
  - **Confusion Matrix**: `[[1886, 46], [34, 34]]`

---

## 4. Tác động Kinh doanh & Kết luận

1. **Hiệu quả kinh tế**: Với ngưỡng tối ưu $0.1824$, hệ thống giúp phát hiện sớm **50%** tổng số ca sự cố hỏng máy thực tế chỉ với tỷ lệ phát cảnh báo dừng máy là **4.0%**, bảo vệ hệ thống sản xuất khỏi các đợt ngừng máy ngoài kế hoạch gây thiệt hại nặng nề.
2. **Độ tin cậy của xác suất**: Việc giảm Brier score xuống **0.0245** đảm bảo xác suất rủi ro xuất ra từ API phản ánh chính xác tỷ lệ xác suất thực tế, phục vụ việc lập kế hoạch bảo trì chủ động.
3. **Mã lý do vận hành (Reason Codes)**: Giúp đội ngũ kỹ sư bảo trì tại nhà máy hiểu ngay nguyên nhân vật lý vượt ngưỡng (ví dụ: công cụ mòn trên 200 phút hoặc mô-men xoắn quá tải) để tiến hành kiểm tra linh kiện nhanh chóng.
