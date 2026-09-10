"""Kiểm tra cảnh báo range cho input trước khi hiển thị kết quả."""

from __future__ import annotations

import pandas as pd


def check_input_ranges(
    features_df: pd.DataFrame,
    reference_ranges: dict[str, dict[str, float]],
) -> list[str]:
    """Trả về cảnh báo đơn biến; không biến cảnh báo thành lỗi suy luận."""
    if not reference_ranges:
        return []

    warnings: list[str] = []
    for column in features_df.columns:
        if column not in reference_ranges or not pd.api.types.is_numeric_dtype(features_df[column]):
            continue
        value = float(features_df[column].iloc[0])
        bounds = reference_ranges[column]
        lower = bounds.get("p0_5")
        upper = bounds.get("p99_5")
        if lower is None or upper is None or lower <= value <= upper:
            continue
        warnings.append(
            f"{column}={value:.2f} ngoài dải quan sát P0.5-P99.5 [{lower:.2f}, {upper:.2f}]"
        )
    return warnings
