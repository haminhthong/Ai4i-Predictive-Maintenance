# ⚙️ AI4I Machine Failure Risk Decision System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.1-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-19%20Passed-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://www.docker.com/)

> **A production-oriented machine-failure risk system using leakage-safe sensor features, calibrated ensemble models, cost- and capacity-aware alert policies, reliability guardrails, failure-mode error analysis, versioned artifacts, FastAPI, and monitoring-ready serving.**

---

## 🧭 Pipeline Tổng quan (CV & Technical Summary)

```text
AI4I Sensor Snapshot
       ↓
Data Contract & Leakage Control (Quarantine TWF/HDF/PWF/OSF/RNF & Drop IDs)
       ↓
Physics-Informed & Interaction Features (Delta T, Mechanical Power, Wear x Load)
       ↓
Calibrated Model Zoo Selection (PR-AUC + Brier Score Dynamic Selection)
       ↓
Calibrated Failure Risk Probability
       ↓
Reliability Gate (Distribution Range Guardrail P0.5 - P99.5)
       ↓
Cost- & Capacity-Aware Decision Policy (Frozen Validation Thresholds)
       ↓
Inspection Alert & Operational Reason Codes
       ↓
FastAPI Serving & Monitoring Engine
```

---

## 📌 1. Business Problem: Bài toán Nghiệp vụ trong Nhà máy

Trong môi trường gia công cơ khí và sản xuất công nghiệp hiện đại:
* **Sự cố máy dừng đột ngột (Unplanned Downtime)** gây thiệt hại nghiêm trọng: tắc nghẽn dây chuyền, hỏng phôi sản phẩm, trễ đơn hàng và nguy hiểm cho công nhân. Chi phí cho một ca hỏng hóc bị bỏ sót thường rất lớn ($FN \gg FP$).
* **Bảo trì phòng ngừa quá mức (Over-Maintenance / False Alarms)**: Nếu hệ thống báo động quá nhạy và dừng máy vô tội vạ, đội ngũ kỹ thuật sẽ quá tải, công suất nhà máy suy giảm và nhân viên mất lòng tin vào hệ thống cảnh báo (Alarm Fatigue).
* **Ràng buộc công suất thực tế (Maintenance Capacity Constraint)**: Đội ngũ bảo trì tại mỗi phân xưởng chỉ có đủ nhân lực để xử lý một tỷ lệ thiết bị nhất định mỗi ngày (ví dụ: tối đa $2\% - 5\%$ số máy trên dây chuyền).

**Hệ thống AI đóng vai trò gì?**
Cung cấp một bộ lọc thông minh: Tại mỗi thời điểm snapshot cảm biến, hệ thống tính toán xác suất rủi ro hỏng hóc, kiểm tra độ tin cậy phân bố, áp dụng chính sách ngưỡng tối ưu chi phí và đưa ra quyết định: **Cho phép máy chạy tiếp (`NO_ALERT`)**, **Đưa vào hàng đợi kiểm tra định kỳ (`REVIEW_REQUIRED`)**, hoặc **Kích hoạt quy trình khẩn cấp (`PRIORITY_REVIEW`)**.

---

## 🎯 2. What the System Predicts / Does NOT Predict

Để đảm bảo tính trung thực và chuẩn mực cao nhất của một kỹ sư AI (AI Engineer Rigor):

### 🟢 Hệ thống DỰ BÁO:
* **Snapshot Machine-Failure Risk Classification**: Tại một thời điểm quan sát cụ thể (Inference snapshot), ước lượng xác suất rủi ro máy móc gặp sự cố dựa trên các thông số vận hành tức thời.
* **Cost-Sensitive Decision Action**: Chuyển đổi xác suất rủi ro thành hành động kiểm tra bảo trì tối ưu chi phí và phù hợp với năng lực xử lý của nhà máy.

### 🔴 Hệ thống KHÔNG DỰ BÁO (Overclaim Prevention):
* **KHÔNG PHẢI** dự báo khi nào máy sẽ hỏng.
* **KHÔNG PHẢI** Remaining Useful Life (RUL) hay dự báo tuổi thọ còn lại theo chu kỳ.
* **KHÔNG PHẢI** dự báo máy sẽ hỏng trong $N$ giờ tới.
* *Lý do kỹ thuật*: Bộ dữ liệu AI4I 2020 là các lát cắt quan sát rời rạc (snapshot observations), không chứa chuỗi thời gian liên tục (time-series trajectory / machine lifetime history) để có thể huấn luyện các mô hình RUL/Time-to-failure đáng tin cậy.

---

## 🏗️ 3. Canonical Architecture (Kiến trúc Hệ thống)

### 3.1 Offline ML Development Pipeline
```text
                  AI4I 2020 RAW DATA (10,000 rows)
                                ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA CONTRACT & DATA AUDIT                               │
│    Schema validation, missing check (0), duplicate check (0) │
│    Prevalence (3.39%), feature ranges, SHA256 verification  │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LEAKAGE QUARANTINE BOUNDARY                              │
│    DROP FROM FEATURES: Machine failure (target), UDI, ProdID│
│    QUARANTINE failure modes: TWF, HDF, PWF, OSF, RNF        │
│    (KEEP strictly as offline evaluation metadata)           │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. SHARED FEATURE CONTRACT                                  │
│    Raw sensors + temperature_delta_k + mechanical_power_w   │
│    + wear_load_interaction (Canonical snake_case schema)    │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
                 Stratified Split Registry
               ┌───────────────┼───────────────┐
               │ 64%           │ 16%           │ 20%
           Train Set     Validation Set   Locked Test Set
               │               │
               ↓               │
┌────────────────────────┐     │
│ 4. MODEL DEVELOPMENT   │     │
│    Logistic Regression │     │
│    Random Forest       │     │
│    HistGradientBoosting│     │
│    + Sigmoid Calibration     │
└──────────────┬─────────┘     │
               ↓               │
         Validation Leaderboard (PR-AUC + Brier Score)
               ↓
        Selected Champion: rf_sigmoid_calibrated
               ↓               │
┌──────────────────────────────┴──────────────────────────────┐
│ 5. DECISION POLICY DEVELOPMENT (Frozen on Validation)       │
│    Tuning on Validation Probabilities + Business Costs      │
│    (FN=5x, FP=1x) + Maintenance Capacity Sweep (5%, 3%, 2%) │
│    Freeze primary_alert_threshold & critical_threshold      │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
                       FREEZE ENTIRE SYSTEM
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. LOCKED TEST EVALUATION (Zero-Leakage Oracle Protocol)     │
│    PR-AUC / ROC-AUC / Precision / Recall / F1 / Brier / ECE │
│    Weighted Decision Cost / Failure-Mode Slices / FN-FP dive│
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. VERSIONED ARTIFACTS                                      │
│    model.joblib, model_manifest.json, feature_contract.json │
│    decision_policy.json, reference_distribution.json        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Online Serving Pipeline
```text
Sensor Snapshot Payload
          ↓
Input Contract Validation (Pydantic SensorPayload)
          ↓
Shared Feature Builder (Zero Train-Serving Skew)
          ↓
Champion Model Inference (Calibrated Failure Probability)
          ↓
Reliability Gate (Distribution Range Guardrail P0.5 - P99.5)
          ↓
Frozen Decision Policy Engine
   ┌──────┼───────────────────┐
   ↓      ↓                   ↓
NO_ALERT  REVIEW_REQUIRED  PRIORITY_REVIEW
   │      │                   │
   │      └─────────┬─────────┘
   │                ↓
   │         Operational Reason Codes + Feature Context
   │                ↓
   └──────────> 4-Block API Response (FastAPI / Streamlit UI)
                    ↓
              Prediction Log & Workload / Drift Monitor
```

---

## 🛡️ 4. AI4I Dataset & Leakage Boundary

Bộ dữ liệu **AI4I 2020 Predictive Maintenance Dataset** (UCI ML Repo ID: 601) bao gồm 10,000 dòng dữ liệu quan sát trạng thái máy phay gia công cơ khí.

### Ranh giới Rò rỉ Dữ liệu (Leakage Quarantine)
Dataset chứa 5 cột thể hiện nguyên nhân cụ thể dẫn đến hỏng máy:
1. `TWF` (Tool Wear Failure - Hỏng do mòn dụng cụ)
2. `HDF` (Heat Dissipation Failure - Hỏng do giải nhiệt kém)
3. `PWF` (Power Failure - Hỏng do công suất bất thường)
4. `OSF` (Overstrain Failure - Hỏng do quá tải lực căng/mô-men)
5. `RNF` (Random Failure - Hỏng hóc ngẫu nhiên)

> [!CAUTION]
> **Data Leakage Risk**: Các cờ này là **thông tin hậu nghiệm** (chỉ xác định sau khi máy đã thực sự dừng hỏng và kỹ sư vào mổ xẻ tìm nguyên nhân). Nếu đưa vào làm đặc trưng huấn luyện, mô hình sẽ đạt ROC-AUC giả mạo ~1.0 nhưng vô giá trị trong triển khai thực tế.
> 
> **Giải pháp kiến trúc**: Pipeline loại bỏ 100% các cột này khỏi feature matrix $X$. Chúng được **cách ly nghiêm ngặt thành Evaluation Metadata** chỉ xuất hiện ở bước đánh giá cuối cùng để đo lường tỷ lệ phát hiện từng dạng lỗi.

---

## 📋 5. Data Contract & Data Audit

Hệ thống loại bỏ hoàn toàn việc sử dụng tên cột thô chứa khoảng trắng hay ký tự đặc biệt (`Air temperature [K]`, `Tool wear [min]`) và fuzzy matching `startswith()`. Mọi thành phần đều tuân thủ `src/contracts.py`:

| Canonical Name | Raw Header Alias | Dtype | Unit | Miêu tả |
| :--- | :--- | :---: | :---: | :--- |
| `quality_type` | `Type` | string | Cat (L/M/H) | Phân cấp sản phẩm (L=50%, M=30%, H=20%) |
| `air_temperature_k` | `Air temperature` | float64 | Kelvin (K) | Nhiệt độ môi trường buồng máy |
| `process_temperature_k` | `Process temperature` | float64 | Kelvin (K) | Nhiệt độ quá trình gia công |
| `rotational_speed_rpm` | `Rotational speed` | float64 | RPM | Tốc độ quay của trục chính |
| `torque_nm` | `Torque` | float64 | Nm | Mô-men xoắn hoạt động |
| `tool_wear_min` | `Tool wear` | float64 | Minutes | Thời gian tích lũy mòn dụng cụ |

### Kết quả Data Audit (`reports/data_audit.json`):
* Tổng số quan sát: `10,000` dòng | Trùng lặp: `0` dòng | Khuyết thiếu (Missing): `0`.
* Checksum SHA256: `9b01f38d2d2b860f0ad699bac001875eafbfb6b7159ccd2d746808c73c43e77d`.
* Tỷ lệ mắc (Prevalence): `339` ca hỏng máy (**3.39%**) - Bài toán mất cân bằng dữ liệu thực tế (Imbalanced Classification).

---

## ⚙️ 6. Shared Feature Engineering & Diễn giải Kỹ thuật

Triệt tiêu hoàn toàn **Train-Serving Skew** bằng một module duy nhất `src/features.py`. Cả huấn luyện, đánh giá và FastAPI phục vụ đều dùng chung một bộ biến đổi đặc trưng:

1. **`mechanical_power_w`** $= \tau \cdot \omega = \text{torque\_nm} \times \left(\text{rotational\_speed\_rpm} \times \frac{2\pi}{60}\right)$ (Watts):
   * *Cơ sở*: Công suất cơ học thực tế của trục truyền động theo định luật vật lý cơ học ($P = \tau\omega$).
2. **`temperature_delta_k`** $= \text{process\_temperature\_k} - \text{air\_temperature\_k}$ (Kelvin):
   * *Cơ sở*: **Thermal operating-state proxy** thể hiện sự chênh nhiệt giữa buồng gia công và môi trường, phản ánh trạng thái giải nhiệt của thiết bị.
3. **`wear_load_interaction`** $= \text{tool\_wear\_min} \times \text{torque\_nm}$ (min·Nm):
   * *Cơ sở*: **Engineering interaction proxy** mô phỏng sự tương tác khi dao đã mòn kết hợp với tải trọng mô-men xoắn cao.

---

## 🗂️ 7. Split Protocol & Split Registry

Hệ thống áp dụng **Stratified Random Split** đảm bảo tỷ lệ nhãn mục tiêu `machine_failure` đồng nhất giữa các tập:
* **Train Set**: `6,400` mẫu ($64\%$)
* **Validation Set**: `1,600` mẫu ($16\%$)
* **Locked Test Set**: `2,000` mẫu ($20\%$, $68$ ca hỏng máy)

Để đảm bảo tính tái lập (Reproducibility), toàn bộ chỉ số index của 10,000 dòng được lưu trữ cố định tại **`reports/split_manifest.json`**. Train, evaluate và failure-mode error analysis cam kết luôn đọc trên chính xác cùng một tập dữ liệu.

---

## 🔬 8. Model Development & Validation Leaderboard

Mô hình được huấn luyện trên tập Train và so sánh công bằng trên tập Validation độc lập:

| Model Candidate | Validation PR-AUC | Validation Brier Score | Trạng thái |
| :--- | :---: | :---: | :---: |
| Logistic Baseline | `0.4679` | `0.1136` | Candidate |
| Logistic Sigmoid Calibrated | `0.4759` | `0.0232` | Candidate |
| Random Forest Baseline | `0.8469` | `0.0111` | Candidate |
| **RF Sigmoid Calibrated** | **`0.8613`** | **`0.0092`** | 🏆 **Selected Champion** |
| HistGB Baseline | `0.8356` | `0.0105` | Candidate |
| HistGB Sigmoid Calibrated | `0.8527` | `0.0098` | Candidate |

> **Chiến lược chọn Champion**: Ưu tiên PR-AUC cao nhất; nếu chênh lệch $\le 0.01$, lựa chọn mô hình có Brier Score thấp nhất (hiệu chuẩn xác suất tốt nhất). **Random Forest Sigmoid Calibrated** được chọn làm Champion.

---

## 🎯 9. Probability Calibration

Đối với bài toán ra quyết định theo chi phí, giá trị đầu ra của mô hình bắt buộc phải là **xác suất phản ánh đúng tần suất thực tế (Empirical Frequency)** thay vì chỉ là điểm phân hạng (ranking score).
* Phương pháp: **Sigmoid Calibration (Platt Scaling)** với 3-fold Cross Validation trên tập Train.
* Trên tập Locked Test:
  * **Brier Score**: `0.0084` (gần 0 hoàn hảo).
  * **Expected Calibration Error (ECE)**: `0.0037` (**0.37%**).
* *Lưu ý*: Kết quả hiệu chuẩn được kiểm chứng trên phân bố AI4I; việc triển khai tại từng nhà máy thực tế trong tương lai đòi hỏi kiểm định lại phân bố ngoại suy (External Recalibration).

---

## ⚖️ 10. Decision Policy: Chi phí Nghiệp vụ & Ràng buộc Công suất

### Ma trận Chi phí Trọng số (Relative Cost Units)
Hệ thống sử dụng ma trận chi phí tương đối:
$$\text{Weighted Cost} = (FN \times 5.0) + (FP \times 1.0)$$
* Chi phí bỏ sót 1 ca máy hỏng ($FN$) nặng gấp **5 lần** chi phí kiểm tra nhầm ($FP$).
* *Lưu ý*: Đây là **Cost Units (Đơn vị chi phí tương đối)**, không dùng ký hiệu tiền tệ ảo `$`.

### Đóng băng Ngưỡng Quyết định trên Validation
Toàn bộ các ngưỡng quyết định được tối ưu **HOÀN TOÀN TRÊN TẬP VALIDATION** và ghi nhận vào `artifacts/champion/decision_policy.json`:
* **Primary Alert Threshold ($\theta^*$)**: `0.3574`
* **Critical Escalation Threshold**: `0.4872` (ngưỡng phân vị cao, kích hoạt hành động khẩn cấp `PRIORITY_REVIEW`).

---

## 🔒 11. Locked Test Evaluation & Zero-Leakage Protocol

> [!IMPORTANT]
> **Khắc phục lỗi P0 Protocol**: Không thực hiện bất kỳ tối ưu hóa hay tìm kiếm ngưỡng nào trên tập Test. `src/evaluate.py` chỉ nạp các ngưỡng đã đóng băng từ `decision_policy.json` và đánh giá:

### Bảng Kết quả Đánh giá Ablation Study trên Locked Test (2,000 mẫu độc lập)

| Chiến lược Threshold | Ngưỡng ($\theta$) | Precision | Recall | F1-Score | Alert Rate | Weighted Cost | Cost / 1,000 obs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed 0.50** | `0.5000` | `94.74%` | `79.41%` | `0.8640` | `2.85%` | `73.00` | `36.50` |
| **Max-F1 (Frozen from Val)** | `0.3574` | `88.89%` | `82.35%` | `0.8550` | `3.15%` | `67.00` | `33.50` |
| **Cost-Sensitive ($\theta^*$)** | **`0.3574`** | **`88.89%`** | **`82.35%`** | **`0.8550`** | **`3.15%`** | **`67.00`** | **`33.50`** |
| **Capacity Constrained 3%** | `0.4595` | `93.10%` | `79.41%` | `0.8571` | `2.90%` | `74.00` | `37.00` |
| **Capacity Constrained 2%** | `0.8458` | `95.56%` | `63.24%` | `0.7611` | `2.25%` | `127.00` | `63.50` |

* **PR-AUC trên Locked Test**: `0.8666`
* **ROC-AUC trên Locked Test**: `0.9711`
* **Confusion Matrix [[TN, FP], [FN, TP]]**: `[[1925, 7], [12, 56]]`
* *Nhận xét*: Chính sách tối ưu chi phí đạt **Recall 82.35%** với tỷ lệ cảnh báo kiểm tra chỉ **3.15%**, tiết kiệm chi phí tổn thất đáng kể so với ngưỡng cố định `0.5`.

---

## ⚙️ 12. Failure-Mode Slice Analysis & Giải phẫu Lỗi (Error Analysis)

Nhờ cô lập 5 cờ nguyên nhân hỏng hóc thành **Evaluation Metadata**, hệ thống thực hiện phân tích chuyên sâu khả năng phát hiện theo từng dạng hư hỏng cơ khí:

### 12.1 Tỷ lệ Bắt lỗi theo Cơ chế Hỏng hóc (Failure-Mode Recall)
| Cơ chế Hỏng hóc | Số ca trong Test | Phát hiện | Bỏ sót | Tỷ lệ Recall | Đánh giá Kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **HDF (Heat Dissipation)** | 29 | 29 | 0 | **100.00%** | Mô hình bắt trọn vẹn nhờ đặc trưng `temperature_delta_k`. |
| **OSF (Overstrain)** | 16 | 16 | 0 | **100.00%** | Bắt trọn vẹn nhờ đặc trưng tương tác `wear_load_interaction`. |
| **PWF (Power Failure)** | 13 | 12 | 1 | **92.31%** | Nhận diện chính xác sự bất thường của `mechanical_power_w`. |
| **TWF (Tool Wear Failure)** | 10 | 1 | 9 | **10.00%** | Dao mòn dần không tạo đột biến nhiệt/mô-men tức thời. |
| **RNF (Random Failure)** | 4 | 0 | 4 | **0.00%** | Hỏng hóc ngẫu nhiên không có dấu hiệu suy thoái cảm biến. |

### 12.2 Giải phẫu 12 ca Bỏ sót (False Negatives) và 7 ca Báo nhầm (False Positives)
* **12 ca False Negatives**: Có tới **9 ca là lỗi Tool Wear Failure (TWF)** đơn lẻ và **1 ca PWF**. Các ca này có mô-men xoắn ở mức trung bình (mean: $37.82\text{ Nm}$), không phát sinh quá tải đột ngột, dẫn đến xác suất rủi ro snapshot chưa vượt ngưỡng.
* **7 ca False Positives**: Các thiết bị có độ mòn dao cao ($>200\text{ min}$) và nhiệt độ gia công tiệm cận biên nhưng chưa thực sự dừng máy tại thời điểm snapshot.

---

## 🛡️ 13. Reliability Guardrails (Kiểm soát Vùng Tin cậy)

Hệ thống sử dụng **Distribution Range Guardrails** dựa trên phân vị $P_{0.5} - P_{99.5}$ của từng đặc trưng cảm biến trên tập Train:
* **Không mạo danh Joint OOD**: Đây là kiểm tra biên phân bố đơn biến, cảnh báo operator khi cảm biến đo được giá trị cực trị chưa từng xuất hiện.
* **Tác động lên Quyết định**: Khi kích hoạt cảnh báo phân bố:
  * Trạng thái độ tin cậy chuyển sang: `reliability.status = "DEGRADED"`.
  * Khuyến nghị vận hành tự động nâng cấp lên: `REVIEW_REQUIRED` (yêu cầu kỹ thuật viên kiểm tra trực tiếp cảm biến).

---

## 🚀 14. Serving Architecture & RESTful API (4-Block Schema)

### 14.1 Kiến trúc 4 Khối Độc lập (`POST /predict-risk`)
Mỗi phản hồi từ API được cấu trúc thành 4 khối minh bạch:

```json
{
  "prediction": {
    "failure_risk": 0.8452,
    "model_version": "ai4i-20260907-9b01f38",
    "model_type": "rf_sigmoid_calibrated"
  },
  "reliability": {
    "status": "NOMINAL",
    "distribution_warning": false,
    "warning_features": []
  },
  "decision": {
    "action": "PRIORITY_REVIEW",
    "alert_threshold": 0.3574,
    "critical_threshold": 0.4872,
    "maintenance_alert": true,
    "policy_version": "maintenance-policy-v2",
    "cost_scenario": "FN5_FP1"
  },
  "operational_context": {
    "reason_codes": [
      "TOOL_WEAR_HIGH",
      "TORQUE_HIGH",
      "ROTATIONAL_SPEED_LOW"
    ],
    "feature_context": [
      {"feature": "tool_wear_min", "value": 210.0, "type": "observed_feature"},
      {"feature": "torque_nm", "value": 60.0, "type": "observed_feature"}
    ]
  }
}
```

* **Operational Reason Codes**: Các cờ heuristic độc lập (`TOOL_WEAR_HIGH`, `TORQUE_HIGH`, `ROTATIONAL_SPEED_LOW`, `TEMPERATURE_DELTA_LOW`) giúp kỹ sư nhà máy hiểu ngay trạng thái vận hành.
* **Feature Context**: Thể hiện minh bạch giá trị quan sát của mô hình ensemble (không mạo danh giải thích xấp xỉ).

---

## 📈 15. Monitoring Pipeline Architecture (Thiết kế Giám sát)

Kiến trúc giám sát vận hành được chuẩn bị sẵn trong `src/monitoring.py`:
* **Feature Drift Monitoring**: Đo lường độ trôi dạt phân bố cảm biến theo thời gian thực thông qua chỉ số **Population Stability Index (PSI)** (cảnh báo khi $PSI \ge 0.1$, kích hoạt retraining khi $PSI \ge 0.2$).
* **Prediction Drift & Workload Monitoring**: Theo dõi tỷ lệ phát cảnh báo (Alert Rate) và giá trị rủi ro trung bình theo cửa sổ trượt (Rolling Window $N=500$) để kịp thời phát hiện tình trạng quá tải bảo trì.

---

## 💻 16. Reproducibility: Hướng dẫn Vận hành & Tái lập

### Bước 1: Cài đặt Môi trường
```bash
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Bước 2: Tải Dữ liệu Gốc
```bash
python scripts/download_data.py
```

### Bước 3: Huấn luyện Đa Mô hình, Chọn Champion & Đóng băng Policy
```bash
python -m src.train
```

### Bước 4: Đánh giá Độc lập trên Locked Test & Phân tích Lát cắt
```bash
python -m src.evaluate
```

### Bước 5: Chạy Toàn bộ 19 Bài Test Kiểm định Bất biến Kiến trúc
```bash
python -m pytest tests/ -v
```

### Bước 6: Khởi chạy FastAPI Service
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Truy cập Swagger UI tại: `http://localhost:8000/docs`

### Bước 7: Khởi chạy Streamlit Dashboard
```bash
streamlit run app.py
```

---

## ⚠️ 17. Limitations: Giới hạn Kỹ thuật Hiện tại

1. **Dữ liệu Bán Tổng hợp (Synthetic AI4I)**: AI4I 2020 là bộ dữ liệu mô phỏng dựa trên thông số máy phay thực tế. Khi triển khai tại nhà máy mới, cần thu thập dữ liệu calibration ngoại vi.
2. **Snapshot Risk Classification**: Không thay thế được việc dự báo chuỗi suy thoái dài hạn nếu nhà máy cần lên lịch trước hàng tuần.

---

## 🔮 18. Future Work & RUL Task Expansion

* **Task B (Dự báo RUL)**: Mở rộng bài toán sang bộ dữ liệu chuỗi thời gian liên tục **NASA C-MAPSS Turbofan Engine Degradation** hoặc **FEMTO Bearing Dataset** để xây dựng mô hình dự báo Remaining Useful Life (RUL) với Transformer / Temporal CNN.
* **Auto-Retraining Trigger**: Kết nối PSI monitoring pipeline để tự động cảnh báo và kích hoạt quy trình CI/CD huấn luyện lại mô hình khi phân bố nguyên liệu gia công thay đổi.

---

## 📄 License & Attribution

Dự án được xây dựng phục vụ cho mục đích học tập, nghiên cứu và triển khai sản xuất tiêu chuẩn AI Engineering. Dữ liệu thuộc bản quyền UCI Machine Learning Repository (AI4I 2020 Predictive Maintenance Dataset).
