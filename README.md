# ⚙️ Machine Failure Risk & Maintenance Decision System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://www.docker.com/)

> **Predictive Maintenance Risk Decision Platform**: A leakage-safe machine-failure risk scoring system with calibrated probabilities, cost-sensitive maintenance decisions, operational reason codes, and production-oriented serving.

---

## 📌 1. Positioning & Core Problem Statement

Trong các nhà máy sản xuất công nghiệp, sự cố máy móc hỏng hóc đột ngột (**Unplanned Downtime**) gây ra thiệt hại kinh tế rất lớn. Tuy nhiên, việc dừng máy kiểm tra quá thường xuyên (**Over-maintenance**) cũng làm gia tăng chi phí nhân công và gián đoạn dây chuyền.

### 🔴 Phân định Bài toán (Problem Definition)
* **Bài toán thực tế hiện tại**: Đây là hệ thống **Phân loại Rủi ro Sự cố Máy móc từ Ảnh chụp Trạng thái Cảm biến (Snapshot Machine-Failure Risk Classification)** tại thời điểm suy luận (Inference time).
* **Điều không overclaim**: Do bộ dữ liệu AI4I 2020 không chứa chuỗi dữ liệu thời gian (Time-series machine trajectory / history), hệ thống **không overclaim** làm dự báo thời gian sử dụng còn lại (**Remaining Useful Life - RUL**) hay dự báo hỏng hóc trước $N$ giờ.
* **Thời gian thực (Real-time Serving)**: Hệ thống phục vụ suy luận thời gian thực qua REST API / Dashboard cho từng điểm dữ liệu cảm biến thu thập từ máy.

---

## 🏗️ 2. Canonical 7-Stage Pipeline Architecture

Hệ thống được thiết kế và vận hành theo **Canonical 7-Stage Pipeline** duy nhất thống nhất từ mã nguồn, thử nghiệm đến phục vụ thực tế:

```text
1. DATA INGESTION
   AI4I 2020 Sensor Observations
        ↓
2. ANTI-LEAKAGE & FEATURE ENGINEERING
   Drop: Target (Machine failure), Failure Flags (TWF/HDF/PWF/OSF/RNF), IDs (UDI, Product ID)
   Engineer: temperature_delta, mechanical_power (Watts), wear_load_interaction
        ↓
3. DATA PARTITION
   Stratified Random Split (Train 64% / Validation 16% / Test 20%)
        ↓
4. MODEL DEVELOPMENT & CALIBRATION
   Benchmark Model Zoo (Logistic Regression, Random Forest, HistGradientBoosting)
   Evaluate Raw vs. Sigmoid Calibrated (PR-AUC & Brier Score Trade-off) -> Dynamic Champion Selection
        ↓
5. DECISION OPTIMIZATION
   Validation Probabilities + Business Cost Matrix (FN=5x, FP=1x) + Maintenance Capacity Sweep
        ↓
6. FINAL TEST EVALUATION
   Untouched Test Set: PR-AUC, ROC-AUC, Precision, Recall, F1, Brier, ECE, Alert Rate, Confusion Matrix
   Test Expected Business Cost & Threshold Strategy Ablation Study
        ↓
7. SERVING & API
   Sensor Request -> Shared Preprocessing -> Failure Risk Score + Model Explanation
   -> Decision Policy (Alert / Risk Tier) + Operational Reason Codes -> FastAPI / Streamlit UI
```

---

## 🛡️ 3. Anti-Leakage Design & Feature Engineering

### 3.1 Phòng chống Rò rỉ Dữ liệu (Anti-Leakage Architecture)

> [!IMPORTANT]
> **Key Strength**: *The model predicts machine-failure risk using ONLY information available at inference time.*

Dataset AI4I 2020 chứa các cột nguyên nhân hỏng hóc bao gồm: `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), `RNF` (Random Failure). 
Các cột này là **thông tin hậu nghiệm** (chỉ biết sau khi máy đã thực sự xảy ra sự cố). Nếu đưa các biến này vào huấn luyện, mô hình sẽ bị rò rỉ dữ liệu (Data Leakage) nghiêm trọng. Pipeline tự động **loại bỏ 100% các cột này** cùng thuộc tính định danh (`UDI`, `Product ID`) trước khi chia tập dữ liệu.

### 3.2 Kỹ thuật Đặc trưng Vật lý (Physics-Based Feature Engineering)

Dựa trên nguyên lý cơ học & nhiệt động lực học trong sản xuất:
1. **`temperature_delta`** $= T_{\text{process}} - T_{\text{air}}$ (Kelvin K): Độ chênh lệch nhiệt độ giữa quá trình vận hành và không khí xung quanh, đại diện cho nhiệt ma sát tích tụ.
2. **`mechanical_power`** $= \tau \cdot \omega = \text{Torque} \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$ (Watts): Công suất cơ học thực tế của trục máy tính theo đơn vị công suất chuẩn.
3. **`wear_load_interaction`** $= \text{Tool Wear} \times \text{Torque}$ (min·Nm): Tải trọng lực tương tác tích lũy tác động lên công cụ theo thời gian.

---

## 🔬 4. Model Benchmark & Dynamic Champion Selection

Hệ thống đánh giá đa mô hình (Model Zoo) gồm Logistic Regression, Random Forest và HistGradientBoosting (cả dạng thô và hiệu chỉnh xác suất `CalibratedClassifierCV` bằng phương pháp Sigmoid Calibration):

### Kết quả Validation Leaderboard

| Model Candidate | Validation PR-AUC | Validation Brier Score | Trạng thái |
| :--- | :---: | :---: | :---: |
| Logistic Baseline | `0.4679` | `0.1136` | Candidate |
| Logistic Sigmoid Calibrated | `0.4759` | `0.0232` | Candidate |
| Random Forest Baseline | `0.8469` | `0.0111` | Candidate |
| **RF Sigmoid Calibrated** | **`0.8613`** | **`0.0092`** | 🏆 **Selected Champion** |
| HistGB Baseline | `0.8356` | `0.0105` | Candidate |
| HistGB Sigmoid Calibrated | `0.8527` | `0.0098` | Candidate |

> [!NOTE]
> **Dynamic Champion Selection**: Mô hình Champion không cố định cứng trong mã nguồn mà được lựa chọn tự động dựa trên **PR-AUC tối ưu nhất**; nếu mức chênh lệch PR-AUC $\le 0.01$, mô hình có **Brier Score thấp nhất** (xác suất chuẩn xác nhất) sẽ được chọn.

---

## 💰 5. Cost-Sensitive Threshold Optimization & Test Ablation

### 5.1 Tối ưu Ngưỡng Quyết định theo Chi phí Kinh doanh
Hệ thống không dùng ngưỡng cố định `0.5`. Ngưỡng cảnh báo tối ưu ($\theta^*$) được tìm trên tập Validation bằng cách tối thiểu hóa tổng chi phí nghiệp vụ với kịch bản minh họa:

$$\text{Total Cost} = (\text{FN} \times C_{\text{FN}}) + (\text{FP} \times C_{\text{FP}})$$

với chi phí bỏ sót 1 máy hỏng ($C_{\text{FN}} = 5.0$) và chi phí kiểm tra nhầm ($C_{\text{FP}} = 1.0$).

### 5.2 Bảng Đánh giá Ablation Study trên Tập Test Độc lập (2,000 mẫu)

Bảng so sánh 3 chiến lược threshold trên tập Hold-out Test chưa từng thấy:

| Threshold Strategy | Threshold ($\theta$) | Precision | Recall | Alert Rate | Test Expected Cost | Cost / 1,000 Machines |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold 0.50** | `0.5000` | `94.74%` | `79.41%` | `2.85%` | `$73.00` | `$36.50` |
| **Max F1 Strategy** | `0.3965` | `91.80%` | `82.35%` | `3.05%` | `$65.00` | `$32.50` |
| **Cost-Sensitive Tuned ($\theta^*$)** | **`0.3574`** | **`88.89%`** | **`82.35%`** | **`3.15%`** | **`$67.00`** | **`$33.50`** |

* **Brier Score trên Test**: `0.0084`
* **Expected Calibration Error (ECE)**: `0.0037` (0.37%)
* **Confusion Matrix [[TN, FP], [FN, TP]]**: `[[1925, 7], [12, 56]]`

---

## 🔌 6. RESTful API & Dashboard Interface

Hệ thống phục vụ qua **FastAPI RESTful Service** và giao diện **Streamlit Interactive Dashboard**:

### 6.1 Endpoints Specification

* `GET /health/live`: Liveness probe.
* `GET /health/ready`: Readiness probe (kiểm tra mô hình, tệp config, feature contract version).
* `POST /predict-risk`: Suy luận rủi ro, phân cấp rủi ro (`HIGH`, `MEDIUM`, `LOW`), kiểm tra OOD (Out-Of-Distribution) và trả về phản hồi cấu trúc:

**Example Request Payload (`POST /predict-risk`):**
```json
{
  "Type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1200.0,
  "torque_nm": 60.0,
  "tool_wear_min": 210.0
}
```

**Example Structured Response (`200 OK`):**
```json
{
  "failure_risk": 0.8452,
  "threshold": 0.3574,
  "risk_tier": "HIGH",
  "alert": true,
  "reason_codes": [
    "TOOL_WEAR_HIGH",
    "TORQUE_HIGH",
    "ROTATIONAL_SPEED_LOW"
  ],
  "model_version": "ai4i-20260905-9b01f38",
  "prediction": {
    "failure_probability": 0.8452,
    "model_version": "ai4i-20260905-9b01f38",
    "ood_warning": false,
    "ood_features": []
  },
  "decision": {
    "maintenance_alert": true,
    "threshold": 0.3574,
    "risk_tier": "HIGH",
    "cost_scenario": "FN5_FP1"
  },
  "model_explanation": [
    {
      "feature": "wear_load_interaction",
      "value": 12600.0,
      "type": "observed_feature"
    }
  ]
}
```

---

## 🚀 7. Hướng dẫn Cài đặt & Vận hành

### Yêu cầu Tiên quyết
* **Python**: `3.11` trở lên
* **Git** & **Pip**

### Cài đặt & Khởi chạy nhanh
```bash
# 1. Cài đặt môi trường
pip install -r requirements.txt

# 2. Tải dữ liệu thô AI4I 2020
python scripts/download_data.py

# 3. Huấn luyện đa mô hình & Chọn Champion
python -m src.train

# 4. Đánh giá độc lập trên tập Test
python -m src.evaluate

# 5. Mở RESTful API Service
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6. Khởi chạy Streamlit Dashboard
streamlit run app.py

# 7. Chạy Automated Test Suite
python -m pytest -v
```

---

## 🐳 8. Containerization (Production-Oriented Setup)

Dự án cung cấp **production-oriented container setup** với Dockerfile tối ưu multi-stage build:

```bash
# Build image
docker build -t machine-failure-risk-system:latest .

# Run container
docker run -d -p 8000:8000 --name risk-service machine-failure-risk-system:latest

# Check readiness probe
curl http://127.0.0.1:8000/health/ready
```

---

## 🗺️ 9. Limitations & Future Roadmap

### Hạn chế Hiện tại
1. Dữ liệu AI4I 2020 là dữ liệu dạng Snapshot cảm biến, chưa có chuỗi thời gian liên tục (Time trajectories) để làm dự báo RUL.
2. Chi phí nghiệp vụ FN:FP = 5:1 là giả định kịch bản minh họa (Illustrative business scenario).

### Ưu tiên Phát triển Tương lai (Roadmap)
* **P2**: Tích hợp Drift Monitoring (PSI index cho cảm biến & risk score distribution).
* **P2**: Xây dựng Deployment Quality Gate tự động trước khi promote model mới.
* **P3**: Mở rộng bài toán Task B với bộ dữ liệu NASA C-MAPSS / PHM cho Remaining Useful Life (RUL) forecasting.

---

## 📄 10. Giấy phép & Tác quyền

Dự án được xây dựng phục vụ cho mục đích học tập, nghiên cứu và triển khai sản xuất tiêu chuẩn AI Engineering. Dữ liệu thuộc bản quyền UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset).

