# AI4I Condition-Based Maintenance Risk Triage

Đây là hệ thống triage bảo trì dựa trên trạng thái vận hành hiện tại của một operating snapshot AI4I.
Hệ thống nhận event cảm biến, kiểm tra chất lượng dữ liệu, tính risk score đã calibration, rồi xếp tài sản vào maintenance queue theo capacity.

Risk score có nghĩa là:

> Mức rủi ro gắn với trạng thái vận hành của snapshot hiện tại dưới phân bố AI4I.

Score không phải xác suất máy sẽ hỏng trong tương lai, không dự báo thời điểm hỏng, không dự báo RUL và không xác nhận được “hỏng trong 24 giờ tới”.
AI4I là dữ liệu snapshot i.i.d.; repo không claim temporal generalization hay future unseen machines.

## Luồng logic

```text
OFFLINE
AI4I raw snapshot
    -> schema/audit + SHA256
    -> quarantine target, identifier và failure-mode metadata
    -> shared feature contract
    -> Development 70%
    -> 5-fold Stratified CV: Logistic / Random Forest / HistGB
    -> chọn production candidate
    -> sigmoid calibration trên Development
    -> Policy Validation 15%: freeze policy và queue policy
    -> Locked Test 15%: report only
    -> immutable release bundle + SHA256 manifest

ONLINE
sensor event + asset/time metadata
    -> schema và hard physical limits
    -> shared feature builder
    -> snapshot_failure_risk
    -> reliability gate: NOMINAL / DEGRADED / UNAVAILABLE
    -> risk event store
    -> latest valid event / asset
    -> sort risk giảm dần
    -> top-K queue + priority override
    -> technician review -> offline outcome monitoring/retraining
```

## Data contract

API dùng `product_quality_type` với giá trị `L`, `M`, `H`. Đây là chất lượng sản phẩm của AI4I, không phải machine identity.
Trong feature contract nội bộ, trường này được canonical thành `quality_type` để khớp dữ liệu gốc.

Metadata runtime gồm `event_id`, `asset_id`, `event_time`, `line_id`, `sensor_source` và `shift`.
Metadata được dùng để logging, monitoring, queueing và maintenance history; không đi vào model features.

Model chỉ nhận 9 features theo đúng thứ tự:

```text
quality_type
air_temperature_k
process_temperature_k
rotational_speed_rpm
torque_nm
tool_wear_min
temperature_delta_k
mechanical_power_w
wear_load_interaction
```

Ba engineered features là domain-informed operating-state features:

```text
temperature_delta_k   = process_temperature_k - air_temperature_k
mechanical_power_w    = torque_nm * rotational_speed_rpm * 2π / 60
wear_load_interaction = tool_wear_min * torque_nm
```

`mechanical_power_w` là đại lượng vật lý; `wear_load_interaction` là engineering proxy, không phải causal failure model.
Các cột `machine_failure`, `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` không được đưa vào inference feature vector.
Failure-mode flags là target-derived diagnostic labels chỉ dùng cho offline slice analysis.

## Model lifecycle

- Development 70%: dùng 5-fold Stratified CV để so sánh base models, raw/engineered contract và calibration.
- Policy Validation 15%: chỉ dùng để đóng băng threshold triage và policy vận hành.
- Locked Test 15%: chỉ report, không chọn model, feature hay threshold.
- Split hiện tại là `stratified_random`, không phải temporal validation.
- Production candidate là Random Forest khi PR-AUC nằm trong tolerance 0.01 của ứng viên tốt nhất; Logistic là baseline và HistGradientBoosting là challenger.
- Calibration là stage riêng: Random Forest -> sigmoid cross-fitted calibration -> risk score.

### Reliability gate

P0.5–P99.5 chỉ là univariate distribution range guardrail, không phải OOD probability hay model uncertainty.

- `NOMINAL`: reference distribution tồn tại và snapshot nằm trong dải tham chiếu.
- `DEGRADED`: có cảm biến nằm ngoài dải tham chiếu.
- `UNAVAILABLE`: thiếu reference distribution; hệ thống fail-closed, yêu cầu review và không phát `NO_ALERT`.

Reason/observed conditions như `TOOL_WEAR_HIGH` hoặc `TORQUE_HIGH` là heuristic vận hành độc lập.
Chúng không phải explanation của Random Forest. Model attribution nếu cần chỉ thực hiện offline.

## Maintenance queue

`POST /maintenance/queue/build` không dùng một threshold cố định để giả vờ capacity-aware.
Queue lấy event hợp lệ mới nhất của mỗi asset, giữ toàn bộ `PRIORITY_REVIEW` như priority override, sau đó xếp `REVIEW_REQUIRED` theo risk giảm dần và lấy tối đa `capacity` event.

Risk events và prediction được lưu trong SQLite tại `data/runtime/risk_events.sqlite3` mặc định; có thể đổi bằng biến môi trường `RISK_EVENT_DB_PATH`.
Technician review là dữ liệu feedback cho offline QA và retraining, không retrain trực tiếp khi người dùng click.

## API

### Chấm điểm snapshot

`POST /score` là endpoint canonical. `/predict-risk` được giữ làm alias tương thích.

```json
{
  "event_id": "evt_123",
  "asset_id": "MACHINE_042",
  "event_time": "2026-09-07T10:15:00Z",
  "shift": "2026-09-07-NIGHT",
  "product_quality_type": "M",
  "air_temperature_k": 300.2,
  "process_temperature_k": 310.1,
  "rotational_speed_rpm": 1510,
  "torque_nm": 42.1,
  "tool_wear_min": 178
}
```

Response canonical gồm `risk`, `reliability`, `triage` và `observed_conditions`:

```json
{
  "event_id": "evt_123",
  "asset_id": "MACHINE_042",
  "risk": {
    "snapshot_failure_risk": 0.684,
    "model_version": "ai4i-risk-v3.0.0"
  },
  "reliability": {
    "status": "NOMINAL",
    "distribution_warning": false,
    "warning_features": []
  },
  "triage": {
    "priority": "HIGH",
    "queue_eligible": true,
    "policy_version": "maintenance-policy-v2"
  },
  "observed_conditions": ["TOOL_WEAR_HIGH"]
}
```

Các block cũ `prediction`, `decision`, `operational_context` và trường phẳng cũ vẫn được trả về để client cũ migrate dần.

### Dựng queue

`POST /maintenance/queue/build`

```json
{
  "shift": "2026-09-07-NIGHT",
  "capacity": 30
}
```

`capacity=30` giới hạn queue thường ở 30; số lượng có thể lớn hơn chỉ khi có priority override rõ ràng.
Event dùng để dựng queue phải có cùng `shift` với request queue; event không có shift sẽ không bị xếp nhầm vào scheduling window.

`POST /maintenance/reviews` lưu `technician_action`, `confirmed_issue`, `failure_mode` và ghi chú vào `maintenance_reviews`.
Review chỉ tạo dữ liệu outcome cho QA/retraining offline; API không cập nhật model ngay lập tức.

### Health

- `GET /health/live`: tiến trình còn sống.
- `GET /health/ready`: kiểm tra model, release checksum, feature contract, reference distribution và sample inference.
- `GET /health`: trạng thái tổng quát.

## Release bundle

Mỗi release nằm trong `releases/<version>/` và tự chứa:

```text
model.joblib
model_config.json
feature_contract.json
calibration.json
decision_policy.json
reference_distribution.json
data_manifest.json
validation_metrics.json
locked_test_metrics.json
MODEL_CARD.md
manifest.json
```

`manifest.json` lưu SHA256 cho model, contract, policy, reference distribution và toàn bộ file thành phần.
Nếu hash mismatch, thiếu file, contract sai thứ tự hoặc thiếu reference distribution thì readiness không đạt.
Thư mục `artifacts/champion` và `models` chỉ là mirror migration cho client cũ; release bundle mới là nguồn chuẩn.

## Báo cáo hiện tại

Locked Test mới nhất của release Random Forest có:

- PR-AUC: `0.9302`
- ROC-AUC: `0.9794`
- Brier: `0.0055`
- ECE: `0.0080`
- Precision: `0.8070`
- Recall: `0.9020`

Slice limitation cần đọc cùng overall metrics:

- TWF recall: `20%` — detector yếu với tool-wear-related failures.
- RNF recall: `0%` — xem là out-of-model scope vì không có precursor đáng tin trong sensor contract.
- HDF/PWF/OSF được báo cáo riêng trong `reports/failure_mode_analysis.json`.
- So sánh TWF detected/missed nằm trong `reports/twf_error_analysis.json`.

Metric headline cho maintenance là PR-AUC và Failure Capture@K; F1 chỉ là metric phụ.
PSI hoặc drift không tự động retrain. Drift chỉ tạo trigger điều tra vì có thể do regime mới, vật liệu, calibration sensor hoặc maintenance event.

## Chạy dự án

```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
uvicorn src.api:app --reload
streamlit run app.py
pytest -q
```

Nếu dùng runtime Python bundled của Codex, thay `python` bằng đường dẫn Python tương ứng của workspace.

## Cấu trúc chính

```text
src/
  artifact.py       # release và checksum
  contracts.py      # canonical schema và leakage boundary
  data.py           # audit và Development/Policy/Test split
  features.py       # shared feature builder
  models.py         # preprocessing, candidates và metrics
  train.py          # CV -> policy validation -> release
  evaluate.py       # locked test report only
  policy.py         # triage và top-K queue
  inference.py      # risk snapshot -> triage -> event
  monitoring.py     # guardrail, PSI, runtime workload
  storage.py        # SQLite event store
  api.py            # FastAPI
app.py              # Streamlit dashboard
releases/           # immutable self-contained bundles
reports/            # validation, locked test và error analysis
tests/              # architectural invariants và integration tests
```
