"""Chính sách triage và xếp hàng bảo trì.

Tất cả ngưỡng quyết định phải được đóng băng trên Policy Validation.
Locked Test chỉ dùng để báo cáo, không dùng để tìm kiếm hay điều chỉnh ngưỡng.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class BusinessCosts:
    """Trọng số chi phí tương đối (Relative Cost Units) dùng để quy đổi xác suất thành quyết định vận hành.

    Attributes:
        false_negative: Trọng số chi phí khi bỏ sót 1 máy hỏng (FN). Mặc định là 5.0 cost units.
        false_positive: Trọng số chi phí khi đưa máy vào kiểm tra không cần thiết (FP). Mặc định là 1.0 cost units.
    """

    false_negative: float = 5.0
    false_positive: float = 1.0

    def validate(self) -> None:
        if self.false_negative <= 0 or self.false_positive <= 0:
            raise ValueError(
                "Trọng số chi phí False Negative và False Positive phải là số dương lớn hơn 0."
            )


def find_threshold_minimizing_cost(
    labels: np.ndarray,
    probabilities: np.ndarray,
    costs: BusinessCosts | None = None,
    max_alert_rate: float | None = None,
) -> tuple[float, float]:
    """Tìm ngưỡng quyết định tối thiểu hóa tổng tổn thất trọng số trên tập Validation.

    Cost = (FN * FN_weight) + (FP * FP_weight)

    Quy tắc hòa (Tie-break): Chọn ngưỡng cao nhất để giảm bớt cảnh báo giả.
    Ràng buộc công suất (Capacity constraint): Nếu max_alert_rate được chỉ định, chỉ chọn
    ngưỡng có alert_rate <= max_alert_rate.
    """
    costs = costs or BusinessCosts()
    costs.validate()

    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)

    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Danh sách nhãn và xác suất phải cùng kích thước và không được rỗng.")

    candidate_thresholds = np.unique(np.r_[0.0, y_prob, 1.0])

    expected_costs: list[float] = []
    alert_rates: list[float] = []

    for thresh in candidate_thresholds:
        preds = y_prob >= thresh
        fn_count = int(np.count_nonzero((y_true == 1) & (~preds)))
        fp_count = int(np.count_nonzero((y_true == 0) & preds))
        cost = costs.false_negative * fn_count + costs.false_positive * fp_count
        expected_costs.append(cost)
        alert_rates.append(float(preds.mean()))

    expected_costs_arr = np.array(expected_costs, dtype=float)
    alert_rates_arr = np.array(alert_rates, dtype=float)

    valid_mask = np.ones_like(expected_costs_arr, dtype=bool)
    if max_alert_rate is not None:
        capacity_mask = alert_rates_arr <= max_alert_rate
        if np.any(capacity_mask):
            valid_mask = capacity_mask
        else:
            valid_mask = alert_rates_arr == alert_rates_arr.min()

    filtered_costs = expected_costs_arr[valid_mask]
    filtered_thresholds = candidate_thresholds[valid_mask]

    minimum_cost = float(filtered_costs.min())
    best_candidates = filtered_thresholds[filtered_costs == minimum_cost]

    # Ưu tiên chọn ngưỡng lớn nhất để giảm tải cảnh báo
    optimal_threshold = float(best_candidates.max())
    return optimal_threshold, minimum_cost


def find_threshold_maximizing_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Tìm ngưỡng quyết định tối ưu hóa F1-Score trên tập Validation.

    Dùng làm chiến lược đối chứng (Comparator) chuẩn mực.
    """
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)

    candidate_thresholds = np.unique(np.r_[0.0, y_prob, 1.0])
    f1_scores = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in candidate_thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    return float(candidate_thresholds[best_idx]), float(f1_scores[best_idx])


def find_critical_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    alert_threshold: float,
    min_precision: float = 0.90,
    min_gap: float = 0.02,
) -> float:
    """Xác định ngưỡng cảnh báo khẩn cấp (PRIORITY_REVIEW) trên tập Validation.

    Ngưỡng này đòi hỏi độ chuẩn xác cao (Precision >= min_precision) hoặc tương đương
    phân vị xác suất cao của các ca hỏng hóc, đảm bảo phân cấp rõ ràng với alert_threshold.
    """
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)

    lower_bound = float(alert_threshold + min_gap)
    candidate_thresholds = np.unique(np.r_[y_prob[y_prob >= lower_bound], 1.0])
    candidate_thresholds.sort()

    valid_critical = []
    for t in candidate_thresholds:
        preds = y_prob >= t
        if np.sum(preds) > 0:
            prec = np.sum((y_true == 1) & preds) / np.sum(preds)
            if prec >= min_precision:
                valid_critical.append(float(t))

    if valid_critical:
        return float(valid_critical[0])

    # Nếu không đạt precision mong muốn, lấy phân vị thứ 75 của xác suất các ca hỏng thật trên Validation
    failure_probs = y_prob[y_true == 1]
    if len(failure_probs) > 0:
        p75 = float(np.percentile(failure_probs, 75))
        return max(lower_bound, p75)

    return max(lower_bound, 0.70)


def tune_all_validation_policies(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    costs: BusinessCosts | None = None,
) -> dict[str, Any]:
    """Huấn luyện và đóng băng tất cả các chiến lược quyết định dựa trên tập Validation."""
    costs = costs or BusinessCosts()

    # 1. Chiến lược Cost-Sensitive tối ưu unconstrained
    cost_thresh, min_cost = find_threshold_minimizing_cost(y_val, probs_val, costs)

    # 2. Chiến lược Max-F1 tối ưu trên Validation
    max_f1_thresh, _ = find_threshold_maximizing_f1(y_val, probs_val)

    # 3. Chiến lược Ngưỡng cố định mặc định 0.50
    fixed_thresh = 0.50

    # 4. Ràng buộc công suất bảo trì (Capacity Constraints): 5%, 3%, 2%
    cap_5_thresh, _ = find_threshold_minimizing_cost(
        y_val, probs_val, costs, max_alert_rate=0.05
    )
    cap_3_thresh, _ = find_threshold_minimizing_cost(
        y_val, probs_val, costs, max_alert_rate=0.03
    )
    cap_2_thresh, _ = find_threshold_minimizing_cost(
        y_val, probs_val, costs, max_alert_rate=0.02
    )

    # 5. Ngưỡng khẩn cấp (Critical Escalation Threshold)
    critical_thresh = find_critical_threshold(y_val, probs_val, alert_threshold=cost_thresh)

    return {
        "policy_version": "maintenance-policy-v2",
        "selected_on": "validation_set",
        "cost_weights": {
            "false_negative": costs.false_negative,
            "false_positive": costs.false_positive,
            "unit": "relative_cost_units",
        },
        "primary_alert_threshold": cost_thresh,
        "critical_threshold": critical_thresh,
        "frozen_thresholds": {
            "fixed_0_50": fixed_thresh,
            "max_f1_validation": max_f1_thresh,
            "cost_sensitive_validation": cost_thresh,
            "capacity_constrained_5pct": cap_5_thresh,
            "capacity_constrained_3pct": cap_3_thresh,
            "capacity_constrained_2pct": cap_2_thresh,
        },
        "validation_expected_cost": min_cost,
    }


def map_decision_action(
    failure_risk: float,
    alert_threshold: float,
    critical_threshold: float,
    distribution_warning: bool = False,
) -> str:
    """Ánh xạ từ xác suất rủi ro và trạng thái reliability sang hành động quyết định cụ thể.

    Hành động:
    - `PRIORITY_REVIEW`: Rủi ro vượt ngưỡng khẩn cấp -> Đưa vào hàng đợi bảo trì ưu tiên cao.
    - `REVIEW_REQUIRED`: Rủi ro vượt ngưỡng cảnh báo -> Đưa vào hàng đợi kiểm tra định kỳ.
    - `NO_ALERT`: Thiết bị hoạt động an toàn trong ngưỡng cho phép.

    Tác động của Reliability Gate:
    Nếu `distribution_warning=True` (nằm ngoài dải phân bố cảm biến chuẩn), dù risk chưa vượt ngưỡng,
    hệ thống vẫn có thể gắn cờ khuyến nghị kiểm tra thủ công.
    """
    if failure_risk >= critical_threshold:
        return "PRIORITY_REVIEW"
    if failure_risk >= alert_threshold:
        return "REVIEW_REQUIRED"
    if distribution_warning:
        # Nếu cảm biến bất thường ngoài khoảng quan sát, cần review thủ công
        return "REVIEW_REQUIRED"
    return "NO_ALERT"


def build_maintenance_queue(
    risk_events: Iterable[dict[str, Any]],
    capacity: int,
) -> list[dict[str, Any]]:
    """Xây dựng queue top-K từ các risk event mới nhất của từng tài sản.

    `PRIORITY_REVIEW` là override rõ ràng và được giữ lại ngay cả khi vượt capacity.
    Các event `UNAVAILABLE` không được đưa vào queue bảo trì vì chúng cần data review.
    """
    if capacity < 0:
        raise ValueError("capacity phải là số nguyên không âm.")

    latest_by_asset: dict[str, dict[str, Any]] = {}

    def event_key(event: dict[str, Any]) -> tuple[int, str]:
        value = str(event.get("event_time", ""))
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return (1, parsed.isoformat())
        except ValueError:
            return (0, value)

    for raw_event in risk_events:
        event = dict(raw_event)
        asset_id = str(event.get("asset_id", "")).strip()
        if not asset_id:
            continue
        if str(event.get("reliability_status", "NOMINAL")) == "UNAVAILABLE":
            continue
        current = latest_by_asset.get(asset_id)
        if current is None or event_key(event) >= event_key(current):
            latest_by_asset[asset_id] = event

    eligible = [
        event
        for event in latest_by_asset.values()
        if event.get("queue_eligible", True)
        and event.get("action") in {"REVIEW_REQUIRED", "PRIORITY_REVIEW"}
    ]
    priority = sorted(
        (event for event in eligible if event.get("action") == "PRIORITY_REVIEW"),
        key=lambda event: float(event.get("risk_score", 0.0)),
        reverse=True,
    )
    regular = sorted(
        (event for event in eligible if event.get("action") == "REVIEW_REQUIRED"),
        key=lambda event: float(event.get("risk_score", 0.0)),
        reverse=True,
    )[:capacity]

    queue: list[dict[str, Any]] = []
    for event in [*priority, *regular]:
        item = dict(event)
        item["rank"] = len(queue) + 1
        queue.append(item)
    return queue


def failure_capture_at_k(
    labels: np.ndarray,
    probabilities: np.ndarray,
    fraction: float,
) -> float:
    """Tính tỷ lệ failure nằm trong nhóm rủi ro cao nhất ở capacity fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải nằm trong khoảng (0, 1].")
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0 or y_true.sum() == 0:
        return 0.0
    count = max(1, int(np.ceil(y_true.size * fraction)))
    top_indices = np.argsort(-y_prob, kind="stable")[:count]
    return float(y_true[top_indices].sum() / y_true.sum())
