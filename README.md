# ⚙️ Predictive Maintenance AI Service (AI4I 2020 Dataset)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://www.docker.com/)

Hệ thống AI dự báo rủi ro hỏng hóc máy móc sản xuất thời gian thực, tự động tối ưu hóa ngưỡng cảnh báo bảo trì dựa trên ma trận chi phí nghiệp vụ và cung cấp giao diện Dashboard minh bạch kèm mã lý do vận hành (Reason Codes).

---

## 📌 1. Tổng quan Dự án

Trong các nhà máy sản xuất công nghiệp, sự cố máy móc hỏng hóc đột ngột (**Unplanned Downtime**) gây ra thiệt hại kinh tế rất lớn. Tuy nhiên, việc dừng máy kiểm tra quá thường xuyên (**Over-maintenance**) cũng làm gia tăng chi phí nhân công và gián đoạn dây chuyền.

Dự án này xây dựng một giải pháp **Predictive Maintenance (Bảo trì Dự báo)** toàn diện dựa trên bộ dữ liệu **AI4I 2020 Predictive Maintenance Dataset** (từ UCI Machine Learning Repository), giúp:
1. **Dự báo xác suất hỏng hóc** của thiết bị từ dữ liệu cảm biến thời gian thực.
2. **Hiệu chỉnh xác suất (Probability Calibration)** để xác suất dự báo phản ánh đúng tần suất rủi ro thực tế.
3. **Tối ưu ngưỡng quyết định (Cost-Sensitive Threshold Tuning)** dựa trên giả định chi phí nghiệp vụ thực tế: **Chi phí bỏ sót 1 máy hỏng (False Negative) cao gấp 5 lần chi phí kiểm tra nhầm (False Positive)**.
4. **Phân cấp rủi ro (Risk Tiering)** thành 3 mức: `HIGH`, `MEDIUM`, `LOW`.
5. **Cung cấp mã lý do vận hành (Operational Reason Codes)** giúp kỹ sư vận hành hiểu rõ nguyên nhân gây ra rủi ro.

---

## 🏗️ 2. Kiến trúc & ML Pipeline

```
┌─────────────────┐    ┌─────────────────────────┐    ┌──────────────────────────┐
│   UCI Dataset   │ ──►│ Anti-Leakage & Features │ ──►│ Sigmoid Calibration      │
│  (AI4I 2020)    │    │ (Delta, Power, Strain)  │    │ Logistic Regression      │
└─────────────────┘    └─────────────────────────┘    └──────────────────────────┘
                                                                   │
                                                                   ▼
┌─────────────────┐    ┌─────────────────────────┐    ┌──────────────────────────┐
│ Streamlit UI    │ ◄──│ FastAPI Service         │ ◄──│ Cost-Tuned Threshold     │
│ & Reason Codes  │    │ /predict-risk & /health │    │ (FN=5x, FP=1x)           │
└─────────────────┘    └─────────────────────────┘    └──────────────────────────┘
```

### 2.1 Phòng chống Rò rỉ Dữ liệu (Data Leakage)
Dataset AI4I chứa các cột nguyên nhân hỏng hóc như `TWF`, `HDF`, `PWF`, `OSF`, `RNF`. Các cột này là nhãn hậu nghiệm (chỉ biết sau khi máy đã hỏng). Pipeline tự động **loại bỏ 100% các cột này** và các thuộc tính ID (`UDI`, `Product ID`) khỏi quá trình huấn luyện để đảm bảo tính thực tế khi triển khai.

### 2.2 Kỹ thuật Đặc trưng Vật lý (Feature Engineering)
Dựa trên tri thức miền (Domain Knowledge), pipeline tính toán 3 đặc trưng vật lý bổ sung:
* **`temperature_delta`** $= T_{\text{process}} - T_{\text{air}}$ (K): Độ chênh lệch nhiệt độ giữa vận hành và không khí, phản ánh mức độ tích nhiệt ma sát.
* **`power_proxy`** $= \text{Rotational Speed} \times \text{Torque}$ (RPM·Nm): Công suất cơ học xấp xỉ của thiết bị.
* **`strain_proxy`** $= \text{Tool Wear} \times \text{Torque}$ (min·Nm): Tải trọng lực tích lũy tác động lên công cụ theo thời gian.

### 2.3 Mô hình hóa & Hiệu chỉnh Xác suất
* **Baseline**: Logistic Regression với `class_weight='balanced'`.
* **Calibrated Model**: Sigmoid Calibrated Classifier (`CalibratedClassifierCV` với 3-fold Cross Validation).
* **Tiêu chuẩn lựa chọn**: Mô hình được chọn dựa trên **PR-AUC** tối ưu và **Brier Score** thấp nhất trên tập Validation.

### 2.4 Tối ưu Ngưỡng Quyết định theo Chi phí (Cost-Sensitive Tuning)
Hệ thống không sử dụng ngưỡng mặc định 0.5. Ngưỡng cảnh báo tối ưu ($\theta^*$) được quét trên tập Validation để tối thiểu hóa hàm tổng chi phí nghiệp vụ:

$$\text{Total Cost} = (\text{FN} \times C_{\text{FN}}) + (\text{FP} \times C_{\text{FP}})$$

Với $C_{\text{FN}} = 5.0$ và $C_{\text{FP}} = 1.0$.

---

## 📁 3. Cấu trúc Thư mục Dự án

```text
Predictive-Maintenance-Ai4i/
├── .dockerignore
├── .env.example
├── .gitattributes
├── .gitignore
├── Dockerfile                  # Cấu hình containerization sản xuất
├── Makefile                    # Phím tắt các lệnh vận hành hệ thống
├── README.md                   # Tài liệu hướng dẫn chi tiết dự án
├── RESEARCH_REPORT.md          # Báo cáo nghiên cứu chuyên sâu về bài toán
├── app.py                      # Dashboard giao diện người dùng Streamlit
├── pytest.ini                  # Cấu hình kiểm thử tự động Pytest
├── requirements.txt            # Danh sách thư viện phụ thuộc Python
├── data/                       # Thư mục chứa dữ liệu
│   ├── raw/                    # Dữ liệu thô AI4I 2020 (ai4i2020.csv)
│   └── processed/              # Dữ liệu qua xử lý
├── models/                     # Thư mục lưu trữ mô hình & cấu hình
│   ├── config.json             # Metadata cấu hình phiên bản & threshold
│   └── model.joblib            # Artifact mô hình đã huấn luyện (.joblib)
├── reports/                    # Báo cáo đánh giá mô hình
│   ├── test_metrics.json       # Kết quả kiểm thử độc lập trên tập Test
│   └── validation_metrics.json # Báo cáo chi tiết trên tập Validation
├── scripts/                    # Scripts tiện ích
│   └── download_data.py        # Script tải dữ liệu từ UCI ML Repo
├── src/                        # Mã nguồn cốt lõi (Source Code)
│   ├── __init__.py
│   ├── api.py                  # Dịch vụ FastAPI RESTful Endpoints
│   ├── data.py                 # Pipeline xử lý dữ liệu & feature engineering
│   ├── evaluate.py             # Đánh giá độc lập mô hình trên tập Test
│   ├── train.py                # Huấn luyện mô hình & tối ưu threshold
│   └── utils.py                # Utilities (logging, set seed, JSON IO)
└── tests/                      # Bộ test tự động (Test Suite)
    ├── __init__.py
    └── test_smoke.py           # Smoke tests & API integration tests
```

---

## 🚀 4. Hướng dẫn Cài đặt & Vận hành

### Yêu cầu Tiên quyết
* **Python**: `3.11` trở lên
* **Git** & **Pip**

### Bước 1: Cài đặt Môi trường & Thư viện
```bash
# Tạo môi trường ảo (khuyên dùng)
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate
# Hoặc trên Linux/macOS: source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
# Hoặc sử dụng Makefile:
make setup
```

### Bước 2: Tải Bộ dữ liệu AI4I 2020
Tải bộ dữ liệu thô tự động từ UCI ML Repository về thư mục `data/raw/`:
```bash
make download
# Hoặc lệnh trực tiếp: python scripts/download_data.py
```

### Bước 3: Huấn luyện Mô hình & Tối ưu Ngưỡng
Huấn luyện Logistic Regression baseline và Sigmoid Calibrated Classifier, tìm ngưỡng chi phí tối ưu và lưu artifact tại `models/`:
```bash
make train
# Hoặc lệnh trực tiếp: python -m src.train
```

### Bước 4: Đánh giá Mô hình trên Tập Test Độc lập
Chạy đánh giá độc lập mô hình đã lưu trên tập Test (20% hold-out test set chưa từng thấy):
```bash
make evaluate
# Hoặc lệnh trực tiếp: python -m src.evaluate
```

### Bước 5: Khởi chạy RESTful API Service (FastAPI)
Khởi chạy dịch vụ API suy luận rủi ro thời gian thực:
```bash
make serve
# Hoặc lệnh trực tiếp: python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
* **Swagger UI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Bước 6: Khởi chạy Dashboard Trực quan (Streamlit)
Mở giao diện tương tác trực quan cho kỹ sư vận hành nhà máy:
```bash
make dashboard
# Hoặc lệnh trực tiếp: streamlit run app.py
```
Dashboard sẽ tự động mở tại giao diện trình duyệt: [http://localhost:8501](http://localhost:8501)

### Bước 7: Chạy Bộ Kiểm thử Tự động (Automated Test Suite)
Chạy toàn bộ bài test kiểm tra chất lượng mã nguồn và API endpoints:
```bash
make test
# Hoặc lệnh trực tiếp: python -m pytest -v
```

---

## 🐳 5. Triển khai ứng dụng với Docker

Ứng dụng cung cấp sẵn `Dockerfile` tiêu chuẩn sản xuất (Multi-stage/Slim base) kèm cơ chế `HEALTHCHECK`:

### 1. Build Docker Image
```bash
docker build -t predictive-maintenance-ai:latest .
```

### 2. Khởi chạy Container
```bash
docker run -d -p 8000:8000 --name pm-ai-service predictive-maintenance-ai:latest
```

### 3. Kiểm tra Trạng thái Container
```bash
docker ps
curl http://127.0.0.1:8000/health
```

---

## 🔌 6. RESTful API Specification

### 6.1 `GET /health`
Kiểm tra sức khỏe hệ thống và trạng thái sẵn sàng của mô hình.

**Example Response (`200 OK`):**
```json
{
  "status": "ok",
  "model_ready": true,
  "model_version": "ai4i-calibrated-v3"
}
```

---

### 6.2 `POST /predict-risk`
Dự báo xác suất hỏng máy từ thông số cảm biến đầu vào.

**Example Request Payload (`POST`):**
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

**Example Response (`200 OK`):**
```json
{
  "failure_risk": 0.8452,
  "threshold": 0.3541,
  "risk_tier": "HIGH",
  "alert": true,
  "reason_codes": [
    "TOOL_WEAR_HIGH",
    "TORQUE_HIGH",
    "ROTATIONAL_SPEED_LOW"
  ],
  "model_version": "ai4i-calibrated-v3"
}
```

**Mô tả Mã Lý do Vận hành (Operational Reason Codes):**
* `TOOL_WEAR_HIGH`: Thời gian độ mòn công cụ $\ge 200$ phút.
* `TORQUE_HIGH`: Mô-men xoắn vận hành $\ge 55$ Nm (vượt ngưỡng chịu tải).
* `ROTATIONAL_SPEED_LOW`: Tốc độ quay $\le 1300$ RPM (bất thường quay chậm).
* `TEMPERATURE_DELTA_LOW`: Chênh lệch nhiệt độ $\le 8.6$ K (khả năng tản nhiệt kém).

---

## 📊 7. Kết quả Đánh giá Mô hình (Model Performance)

Bảng tổng hợp chỉ số kỹ thuật và kinh doanh được ghi nhận từ tập **Hold-out Test** (20% dữ liệu độc lập):

| Metric | Giá trị (Test Set) | Giải thích |
| :--- | :---: | :--- |
| **PR-AUC** | `0.75+` | Chỉ số quan trọng hàng đầu cho dữ liệu mất cân bằng lớp cao (Imbalanced Data) |
| **ROC-AUC** | `0.95+` | Khả năng phân biệt giữa máy bình thường và máy hỏng hóc |
| **Brier Score** | `< 0.03` | Độ tin cậy của xác suất sau khi hiệu chỉnh Sigmoid |
| **Ngưỡng tối ưu ($\theta^*$)** | `~0.35` | Ngưỡng tối ưu hóa chi phí nghiệp vụ ($C_{\text{FN}} = 5.0, C_{\text{FP}} = 1.0$) |
| **Recall (Sensitivity)** | `> 85%` | Tỷ lệ phát hiện thành công các sự cố máy bị hỏng |
| **Alert Rate** | `~ 5 - 8%` | Tỷ lệ đưa ra cảnh báo bảo trì trên tổng số máy vận hành |

---

## 📄 8. Giấy phép & Tác quyền

Dự án được xây dựng phục vụ cho mục đích học tập, nghiên cứu và triển khai sản xuất tiêu chuẩn AI Engineering. Dữ liệu thuộc bản quyền UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset).
