"""Threshold và ranking đơn giản cho snapshot failure-risk."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def find_threshold_maximizing_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Chọn threshold tối đa F1 trên Validation, không dùng Test."""
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        raise ValueError("Nhãn và xác suất phải cùng kích thước và không được rỗng.")
    thresholds = np.unique(np.r_[0.0, y_prob, 1.0])
    scores = [f1_score(y_true, y_prob >= threshold, zero_division=0) for threshold in thresholds]
    best_score = max(scores)
    # Khi F1 hòa, chọn threshold cao hơn để giảm số snapshot bị đưa đi review.
    best_threshold = max(
        threshold
        for threshold, score in zip(thresholds, scores, strict=True)
        if score == best_score
    )
    return float(best_threshold), float(best_score)


def decision_from_risk(failure_risk: float, review_threshold: float) -> str:
    """Ánh xạ risk liên tục về một trong hai quyết định của demo."""
    if not 0.0 <= failure_risk <= 1.0:
        raise ValueError("failure_risk phải nằm trong khoảng [0, 1].")
    if not 0.0 <= review_threshold <= 1.0:
        raise ValueError("review_threshold phải nằm trong khoảng [0, 1].")
    return "REVIEW_REQUIRED" if failure_risk >= review_threshold else "NO_ALERT"


def rank_rows(rows: Iterable[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """Sắp xếp các snapshot theo risk giảm dần và gắn rank."""
    if top_k is not None and top_k < 1:
        raise ValueError("top_k phải lớn hơn 0.")
    ranked = sorted(rows, key=lambda row: float(row["failure_risk"]), reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [{**row, "rank": index} for index, row in enumerate(ranked, start=1)]


def failure_capture_at_k(labels: np.ndarray, probabilities: np.ndarray, fraction: float) -> float:
    """Tỷ lệ failure nằm trong nhóm snapshot có risk cao nhất."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải nằm trong khoảng (0, 1].")
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0 or y_true.sum() == 0:
        return 0.0
    count = max(1, int(np.ceil(y_true.size * fraction)))
    top_indices = np.argsort(-y_prob, kind="stable")[:count]
    return float(y_true[top_indices].sum() / y_true.sum())


def queue_precision_at_k(labels: np.ndarray, probabilities: np.ndarray, fraction: float) -> float:
    """Precision của nhóm snapshot risk cao nhất ở một tỷ lệ K."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải nằm trong khoảng (0, 1].")
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    if y_true.shape != y_prob.shape or y_true.size == 0:
        return 0.0
    count = max(1, int(np.ceil(y_true.size * fraction)))
    top_indices = np.argsort(-y_prob, kind="stable")[:count]
    return float(y_true[top_indices].mean())
