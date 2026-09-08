"""Module giám sát độ trôi dữ liệu (Data & Prediction Drift) và phân tích độ tin cậy vận hành (Operational Reliability).

Cung cấp:
1. Kiểm tra Distribution Range Guardrails dựa trên khoảng phân vị P0.5 - P99.5 của Development.
2. Tính toán chỉ số Population Stability Index (PSI) cho các đặc trưng cảm biến.
3. Giám sát tải vận hành (Workload & Alert Rate Monitoring).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def check_distribution_guardrails(
    features_df: pd.DataFrame,
    reference_ranges: dict[str, dict[str, float]],
) -> tuple[bool, list[str]]:
    """Kiểm tra xem các giá trị đặc trưng quan sát có vi phạm biên dải phân bố huấn luyện (P0.5 - P99.5) hay không.

    LƯU Ý: Đây là Distribution Range Guardrail (kiểm tra phân bố đơn biến), KHÔNG PHẢI
    OOD Detector đa biến toàn phần (Joint Distribution OOD). Khi vi phạm, nó cảnh báo
    rằng cảm biến đang vận hành ở vùng dữ liệu cực trị chưa từng thấy lúc huấn luyện.

    Returns:
        tuple[bool, list[str]]: (has_warning, warning_details_list)
    """
    if not reference_ranges:
        return False, ["Guardrail unavailable: reference_ranges missing"]

    warnings: list[str] = []
    for col in features_df.columns:
        if col in reference_ranges and pd.api.types.is_numeric_dtype(features_df[col]):
            val = float(features_df[col].iloc[0])
            p0_5 = reference_ranges[col].get("p0_5", -float("inf"))
            p99_5 = reference_ranges[col].get("p99_5", float("inf"))
            if val < p0_5 or val > p99_5:
                warnings.append(
                    f"{col}={val:.2f} (ngoài dải huấn luyện P0.5-P99.5: [{p0_5:.2f}, {p99_5:.2f}])"
                )

    return len(warnings) > 0, warnings


def assess_distribution_guardrails(
    features_df: pd.DataFrame,
    reference_ranges: dict[str, dict[str, float]],
) -> tuple[str, list[str]]:
    """Đánh giá guardrail với ba trạng thái NOMINAL, DEGRADED và UNAVAILABLE."""
    if not reference_ranges:
        return "UNAVAILABLE", ["Guardrail unavailable: thiếu reference distribution."]
    missing_ranges = [
        column
        for column in features_df.columns
        if pd.api.types.is_numeric_dtype(features_df[column])
        and (
            column not in reference_ranges
            or "p0_5" not in reference_ranges[column]
            or "p99_5" not in reference_ranges[column]
        )
    ]
    if missing_ranges:
        return "UNAVAILABLE", [
            f"Guardrail unavailable: thiếu reference range cho {missing_ranges}."
        ]
    has_warning, warnings = check_distribution_guardrails(features_df, reference_ranges)
    return ("DEGRADED" if has_warning else "NOMINAL"), warnings


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Tính toán chỉ số Population Stability Index (PSI) đo mức độ trôi phân bố giữa 2 tập dữ liệu.

    Quy ước đánh giá PSI:
    - PSI < 0.1: Không có sự dịch chuyển đáng kể (Stable).
    - 0.1 <= PSI < 0.2: Có sự dịch chuyển nhẹ (Moderate Shift), cần theo dõi.
    - PSI >= 0.2: Dịch chuyển đáng kể, cần điều tra trước khi quyết định retrain.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Phân vị cố định dựa trên tập expected
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Quy đổi thành tỷ lệ phần trăm kèm smoothing epsilon
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


class DriftMonitor:
    """Theo dõi trôi dạt phân bố và tải cảnh báo theo cửa sổ trượt (Rolling Window)."""

    def __init__(self, window_size: int = 500) -> None:
        self.window_size = window_size
        self.risk_scores_window: list[float] = []
        self.alerts_window: list[bool] = []
        self.reliability_window: list[str] = []
        self.asset_ids_window: list[str] = []

    def record_prediction(
        self,
        failure_risk: float,
        alert: bool,
        asset_id: str | None = None,
        reliability_status: str = "NOMINAL",
    ) -> None:
        """Ghi nhận một lượt suy luận mới vào cửa sổ theo dõi."""
        self.risk_scores_window.append(failure_risk)
        self.alerts_window.append(alert)
        self.reliability_window.append(reliability_status)
        self.asset_ids_window.append(asset_id or "unknown")

        if len(self.risk_scores_window) > self.window_size:
            self.risk_scores_window.pop(0)
            self.alerts_window.pop(0)
            self.reliability_window.pop(0)
            self.asset_ids_window.pop(0)

    def get_workload_summary(self) -> dict[str, Any]:
        """Tổng kết tải vận hành trong cửa sổ hiện tại."""
        total = len(self.alerts_window)
        if total == 0:
            return {
                "window_samples": 0,
                "mean_risk": 0.0,
                "current_alert_rate": 0.0,
                "reliability_warning_rate": 0.0,
            }

        return {
            "window_samples": total,
            "mean_risk": float(np.mean(self.risk_scores_window)),
            "current_alert_rate": float(np.mean(self.alerts_window)),
            "alert_count": int(np.sum(self.alerts_window)),
            "reliability_warning_rate": float(
                np.mean([status == "DEGRADED" for status in self.reliability_window])
            ),
            "unique_assets": len(set(self.asset_ids_window)),
        }
