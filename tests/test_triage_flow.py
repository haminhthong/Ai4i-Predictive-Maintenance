"""Test threshold, ranking và feature ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.policy import (
    decision_from_risk,
    failure_capture_at_k,
    find_threshold_maximizing_f1,
    queue_precision_at_k,
    rank_rows,
)
from src.train import choose_model


def test_f1_threshold_is_selected_on_validation() -> None:
    labels = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.4, 0.1])
    threshold, score = find_threshold_maximizing_f1(labels, probabilities)
    assert threshold == 0.6
    assert score == 1.0


def test_decision_has_only_two_actions() -> None:
    assert decision_from_risk(0.2, 0.5) == "NO_ALERT"
    assert decision_from_risk(0.5, 0.5) == "REVIEW_REQUIRED"


def test_rank_rows_returns_top_k() -> None:
    rows = [
        {"record_id": "a", "failure_risk": 0.1},
        {"record_id": "b", "failure_risk": 0.9},
        {"record_id": "c", "failure_risk": 0.5},
    ]
    ranked = rank_rows(rows, top_k=2)
    assert [row["record_id"] for row in ranked] == ["b", "c"]
    assert [row["rank"] for row in ranked] == [1, 2]


def test_top_k_business_metrics_rank_by_risk() -> None:
    labels = np.array([1, 0, 1, 0])
    probabilities = np.array([0.2, 0.9, 0.8, 0.1])
    assert failure_capture_at_k(labels, probabilities, 0.5) == 0.5
    assert queue_precision_at_k(labels, probabilities, 0.5) == 0.5


def test_model_selection_uses_pr_auc_then_brier() -> None:
    leaderboard = {
        "logistic_regression": {"pr_auc_mean": 0.8, "brier_mean": 0.02},
        "random_forest": {"pr_auc_mean": 0.9, "brier_mean": 0.04},
        "hist_gradient_boosting": {"pr_auc_mean": 0.9, "brier_mean": 0.01},
    }
    assert choose_model(leaderboard) == "hist_gradient_boosting"


def test_validation_report_contains_feature_ablation() -> None:
    report = json.loads(Path("reports/validation_metrics.json").read_text(encoding="utf-8"))
    ablation = report["feature_ablation"]
    assert set(ablation) >= {"raw_6", "raw_plus_engineered_9"}
