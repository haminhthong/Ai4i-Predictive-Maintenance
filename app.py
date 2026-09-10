"""Dashboard Streamlit cho scoring một snapshot và ranking batch."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.contracts import FAILURE_MODE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.features import canonicalize_raw_dataframe
from src.inference import RiskInferenceService

st.set_page_config(
    page_title="AI4I Maintenance Risk Triage",
    page_icon="⚙️",
    layout="wide",
)


def load_test_report() -> dict:
    """Đọc metric đã sinh bởi bước evaluate nếu file tồn tại."""
    path = Path("reports/final_test_metrics.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_summary(report: dict) -> None:
    metrics = report.get("test_performance", {})
    st.subheader("Locked Test")
    columns = st.columns(5)
    for column, (label, key) in zip(
        columns,
        (
            ("PR-AUC", "pr_auc"),
            ("ROC-AUC", "roc_auc"),
            ("Brier", "brier"),
            ("ECE", "ece"),
            ("Recall", "recall"),
        ),
        strict=True,
    ):
        column.metric(label, f"{metrics[key]:.4f}" if key in metrics else "N/A")


def render_single(service: RiskInferenceService) -> None:
    st.subheader("Single Snapshot")
    with st.form("single_snapshot"):
        record_id = st.text_input("Record ID (tùy chọn)")
        quality = st.selectbox("Product quality type", ["L", "M", "H"], index=1)
        left, right = st.columns(2)
        with left:
            air_temperature = st.number_input("Air temperature (K)", 290.0, 320.0, 300.0)
            process_temperature = st.number_input("Process temperature (K)", 290.0, 330.0, 310.0)
            speed = st.number_input("Rotational speed (rpm)", 500.0, 4000.0, 1500.0)
        with right:
            torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
            tool_wear = st.number_input("Tool wear (min)", 0.0, 300.0, 120.0)
        submitted = st.form_submit_button("Score snapshot")

    if not submitted:
        return
    payload = {
        "record_id": record_id or None,
        "product_quality_type": quality,
        "air_temperature_k": air_temperature,
        "process_temperature_k": process_temperature,
        "rotational_speed_rpm": speed,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
    }
    result = service.predict(payload)
    result_columns = st.columns(3)
    result_columns[0].metric("Failure risk", f"{result['failure_risk']:.2%}")
    result_columns[1].metric("Decision", result["decision"])
    result_columns[2].metric("Review threshold", f"{result['threshold']:.4f}")
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)
    else:
        st.success("Snapshot nằm trong dải quan sát tham chiếu.")


def render_batch(service: RiskInferenceService) -> None:
    st.subheader("Batch Ranking")
    st.caption(
        "CSV cần có 6 raw sensor variables; các cột target/failure-mode không được dùng để score."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    top_k = st.number_input("Số dòng hiển thị", min_value=1, max_value=10_000, value=20)
    if uploaded is None:
        return

    raw_df = canonicalize_raw_dataframe(pd.read_csv(uploaded))
    forbidden_metadata = [
        column for column in (TARGET_COLUMN, *FAILURE_MODE_COLUMNS) if column in raw_df.columns
    ]
    if forbidden_metadata:
        st.error(f"CSV chứa cột hậu nghiệm không được score: {forbidden_metadata}")
        return
    raw_df = raw_df.drop(columns=list(IDENTIFIER_COLUMNS), errors="ignore")
    raw_df.insert(0, "record_id", [f"row_{index}" for index in range(len(raw_df))])
    try:
        ranked = service.rank(raw_df.to_dict(orient="records"), top_k=int(top_k))
    except (KeyError, TypeError, ValueError) as exc:
        st.error(str(exc))
        return
    st.dataframe(pd.DataFrame(ranked), use_container_width=True)


def main() -> None:
    st.title("AI4I Maintenance Risk Triage")
    st.caption(
        "Failure-risk classification của operating snapshot hiện tại — không phải RUL, "
        "time-to-failure hoặc future forecasting."
    )
    service = RiskInferenceService.get_instance()
    if not service.is_ready:
        st.error("Artifact chưa sẵn sàng. Chạy `python -m src.train` trước.")
        return

    render_summary(load_test_report())
    single_tab, batch_tab = st.tabs(["Single Snapshot", "Batch Ranking"])
    with single_tab:
        render_single(service)
    with batch_tab:
        render_batch(service)


if __name__ == "__main__":
    main()
