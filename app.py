"""Giao diện Dashboard trực quan hoá Hệ thống Phân loại Rủi ro & Quyết định Bảo trì Máy móc (AI4I Machine Failure Risk Decision System).

Ứng dụng hỗ trợ:
1. Điều chỉnh thông số cảm biến vận hành thời gian thực.
2. Gọi Inference Engine chuẩn hóa (Shared Feature Builder, Calibrated Model, Reliability Gate, Decision Policy).
3. Đánh giá Reliability Gate (Distribution Range Guardrail: P0.5 - P99.5 của tập Train).
4. Phân định minh bạch giữa 4 Khối:
   - Dự báo Rủi ro Xác suất (Prediction Block)
   - Trạng thái Tin cậy / Phân bố (Reliability Gate)
   - Quyết định Vận hành (Decision Action: NO_ALERT / REVIEW_REQUIRED / PRIORITY_REVIEW)
   - Ngữ cảnh Vận hành & Đặc trưng (Reason Codes & Feature Context)
5. Trực quan hóa Threshold Ablation Study, Capacity Constraints và Phân tích lát cắt Failure-Mode Recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import RiskInferenceService

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="AI4I Machine Failure Risk Decision System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tùy chỉnh giao diện chuyên nghiệp
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .action-priority {
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 1.2rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .action-review {
        background-color: #FFFBEB;
        border-left: 6px solid #D97706;
        padding: 1.2rem;
        border-radius: 8px;
        color: #92400E;
    }
    .action-noalert {
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A;
        padding: 1.2rem;
        border-radius: 8px;
        color: #166534;
    }
    .guardrail-box {
        background-color: #FFF7ED;
        border: 1px solid #FDBA74;
        border-radius: 6px;
        padding: 0.8rem;
        color: #C2410C;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.markdown(
        '<div class="main-header">⚙️ AI4I Machine Failure Risk Decision System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Leakage-safe machine-failure risk scoring with calibrated probabilities, '
        'cost-sensitive maintenance decisions, reliability guardrails, operational reason codes and production-oriented serving.</div>',
        unsafe_allow_html=True,
    )

    # Sơ đồ Kiến trúc Canonical 7 Giai đoạn
    with st.expander("📌 Sơ đồ Kiến trúc Canonical 7 Giai đoạn (Offline & Online Architecture)", expanded=False):
        st.code(
            """
OFFLINE ML PIPELINE:
Raw AI4I Observations -> Data Contract & Audit (SHA256, Schema, Prevalence)
      ↓
Leakage Quarantine (Drop Target, UDI, Product ID, isolate TWF/HDF/PWF/OSF/RNF for Eval only)
      ↓
Shared Feature Contract (Raw Sensors + temperature_delta_k + mechanical_power_w + wear_load_interaction)
      ↓
Stratified Split Registry (Train 64% / Validation 16% / Locked Test 20%)
      ↓
Model Zoo Benchmark (Logistic, Random Forest, HistGB + Sigmoid Calibration) -> Select Champion
      ↓
Decision Policy Optimization (Validation Probabilities + FN=5x, FP=1x + Capacity Constraints) -> Freeze Policy
      ↓
Locked Hold-out Test Evaluation (Zero-leakage metrics, Failure-Mode Slices, Error Analysis)

ONLINE SERVING PIPELINE:
Sensor Snapshot -> Shared Feature Builder -> Champion Model -> Calibrated Failure Risk
      ↓
Reliability Gate (Distribution Range Guardrail) -> Frozen Decision Policy -> Action (NO_ALERT / REVIEW / PRIORITY)
      ↓
Operational Reason Codes + Feature Context -> 4-Block API Response
            """,
            language="text",
        )

    # Khởi tạo Inference Service
    try:
        service = RiskInferenceService.get_instance()
        service_ready = service.is_loaded
    except Exception as exc:
        service_ready = False
        st.error(f"⚠️ Lỗi khởi tạo Inference Service: {exc}")

    # Nạp báo cáo test metrics và failure mode analysis nếu có
    test_metrics = {}
    test_metrics_path = Path("reports/final_test_metrics.json")
    if not test_metrics_path.exists():
        test_metrics_path = Path("reports/test_metrics.json")

    if test_metrics_path.exists():
        try:
            test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    failure_modes_path = Path("reports/failure_mode_analysis.json")
    failure_modes_data = {}
    if failure_modes_path.exists():
        try:
            failure_modes_data = json.loads(failure_modes_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Sidebar: Nhập thông số cảm biến thời gian thực
    st.sidebar.header("🎛️ Thông số cảm biến thời gian thực")

    machine_type = st.sidebar.selectbox(
        "Chất lượng sản phẩm / Máy (Type)",
        options=["L", "M", "H"],
        index=1,
        help="L: Low quality (50% dataset), M: Medium quality (30%), H: High quality (20%)",
    )

    air_temp = st.sidebar.slider(
        "Nhiệt độ không khí buồng máy (Air Temp - K)",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1,
    )

    proc_temp = st.sidebar.slider(
        "Nhiệt độ quá trình gia công (Process Temp - K)",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1,
    )

    speed_rpm = st.sidebar.slider(
        "Tốc độ quay trục chính (Rotational Speed - RPM)",
        min_value=1100,
        max_value=2900,
        value=1500,
        step=10,
    )

    torque_nm = st.sidebar.slider(
        "Mô-men xoắn hoạt động (Torque - Nm)",
        min_value=3.0,
        max_value=90.0,
        value=40.0,
        step=0.5,
    )

    tool_wear = st.sidebar.slider(
        "Độ mòn dụng cụ tích lũy (Tool Wear - phút)",
        min_value=0,
        max_value=260,
        value=120,
        step=1,
    )

    # Headline KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        model_name = service.manifest.get("model_type", "N/A") if service_ready else "N/A"
        st.metric("Champion Model", model_name)
    with col2:
        perf = test_metrics.get("test_performance", test_metrics)
        st.metric("PR-AUC (Locked Test)", f"{perf.get('pr_auc', 0.0):.4f}" if perf else "N/A")
    with col3:
        st.metric("Brier / ECE", f"{perf.get('brier_score', perf.get('brier', 0.0)):.4f} / {perf.get('ece', 0.0):.4f}" if perf else "N/A")
    with col4:
        alert_thresh = service.policy.get("primary_alert_threshold", 0.3574) if service_ready else 0.3574
        st.metric("Alert Threshold (θ*)", f"{alert_thresh:.4f}")
    with col5:
        cost_1k = perf.get("cost_units_per_1000_observations", perf.get("cost_per_1000_machines", 0.0)) if perf else 0.0
        st.metric("Cost Units / 1k obs", f"{cost_1k:.2f}")

    st.markdown("---")

    if service_ready:
        raw_payload = {
            "Type": machine_type,
            "air_temperature_k": air_temp,
            "process_temperature_k": proc_temp,
            "rotational_speed_rpm": speed_rpm,
            "torque_nm": torque_nm,
            "tool_wear_min": tool_wear,
        }

        # Gọi Inference Service đồng bộ hoàn toàn với API
        result = service.predict(raw_payload)

        pred_block = result["prediction"]
        rel_block = result["reliability"]
        dec_block = result["decision"]
        ops_block = result["operational_context"]

        failure_risk = pred_block["failure_risk"]
        action = dec_block["action"]
        is_alert = dec_block["maintenance_alert"]
        has_warning = rel_block["distribution_warning"]

        # Chọn CSS class theo action
        if action == "PRIORITY_REVIEW":
            box_class = "action-priority"
            action_text = "🚨 PRIORITY REVIEW: RỦI RO CỰC CAO - ĐƯA VÀO HÀNG ĐỢI BẢO TRÌ KHẨN CẤP"
        elif action == "REVIEW_REQUIRED":
            box_class = "action-review"
            action_text = "⚠️ REVIEW REQUIRED: RỦI RO VƯỢT NGƯỠNG - ĐƯA VÀO HÀNG ĐỢI KIỂM TRA ĐỊNH KỲ"
        else:
            box_class = "action-noalert"
            action_text = "🟢 NO ALERT: THIẾT BỊ HOẠT ĐỘNG BÌNH THƯỜNG TRONG NGƯỠNG AN TOÀN"

        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("📊 Quyết định & Rủi ro Vận hành")

            # Cảnh báo Reliability Gate
            if has_warning:
                st.markdown(
                    f"""
                    <div class="guardrail-box">
                        <b>⚠️ DISTRIBUTION RANGE GUARDRAIL:</b> Trạng thái tin cậy: <b>{rel_block['status']}</b>.<br/>
                        Cảm biến vận hành ngoài dải phân vị P0.5 - P99.5 của tập huấn luyện:
                        <ul>{"".join([f"<li>{f}</li>" for f in rel_block['warning_features']])}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Banner Quyết định
            st.markdown(
                f"""
                <div class="{box_class}">
                    <h3>Hành động Đề xuất: {action}</h3>
                    <p style="font-size: 1.7rem; font-weight: bold; margin-bottom: 0.2rem;">
                        Xác suất rủi ro hỏng máy: {failure_risk * 100:.2f}%
                    </p>
                    <p style="font-size: 1.05rem;"><b>Khuyến nghị Vận hành:</b> {action_text}</p>
                    <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.85;">
                        Chính sách: {dec_block['policy_version']} | Kịch bản chi phí: {dec_block['cost_scenario']} |
                        Alert Threshold: {dec_block['alert_threshold']:.4f} | Critical Threshold: {dec_block['critical_threshold']:.4f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(failure_risk)

            st.markdown("#### 🚨 Mã Lý do Vận hành Chuyên gia (Operational Reason Codes):")
            st.caption("Các quy tắc heuristic độc lập dùng cho kỹ sư nhà máy nhanh chóng xác định nguyên nhân bất thường:")
            if ops_block["reason_codes"]:
                for code in ops_block["reason_codes"]:
                    st.warning(f"• **{code}**")
            else:
                st.success("• Không phát hiện bất thường cơ học/nhiệt theo quy tắc chuyên gia.")

            st.markdown("#### 🔍 Ngữ cảnh Đặc trưng Quan sát (Feature Context):")
            st.caption("Các giá trị cảm biến và đặc trưng phái sinh tại thời điểm snapshot:")
            df_ctx = pd.DataFrame(ops_block["feature_context"])
            st.dataframe(df_ctx, use_container_width=True)

        with right_col:
            st.subheader("📐 Đặc trưng Dẫn xuất & Phân tích Nghiệp vụ")

            temp_delta = proc_temp - air_temp
            mech_power = torque_nm * (speed_rpm * 2.0 * 3.14159265 / 60.0)
            wear_load = tool_wear * torque_nm

            df_features = pd.DataFrame(
                {
                    "Đặc trưng dẫn xuất": [
                        "temperature_delta_k",
                        "mechanical_power_w",
                        "wear_load_interaction",
                    ],
                    "Giá trị tính toán": [
                        f"{temp_delta:.2f} K",
                        f"{mech_power:.2f} Watts",
                        f"{wear_load:.2f} min·Nm",
                    ],
                    "Ý nghĩa kỹ thuật": [
                        "Thermal operating-state proxy (Độ chênh nhiệt gia công và buồng máy)",
                        "Mechanical Power thực tế của trục quay (P = Tau * Omega)",
                        "Engineering interaction proxy (Tương tác độ mòn dao và tải lực)",
                    ],
                }
            )
            st.table(df_features)

            st.markdown("#### 📊 Threshold Ablation Study (Đánh giá trên Locked Test):")
            ablation_data = test_metrics.get("threshold_ablation_study", {})
            if ablation_data:
                rows = []
                for strat_name, res in ablation_data.items():
                    rows.append(
                        {
                            "Chiến lược Ngưỡng": strat_name,
                            "Ngưỡng (θ)": f"{res['threshold']:.4f}",
                            "Recall": f"{res['recall']:.2%}",
                            "Precision": f"{res['precision']:.2%}",
                            "Alert Rate": f"{res['alert_rate']:.2%}",
                            "Cost Units / 1k": f"{res.get('cost_units_per_1000', res.get('cost_per_1000_machines', 0.0)):.2f}",
                        }
                    )
                st.table(pd.DataFrame(rows))

            if failure_modes_data:
                st.markdown("#### ⚙️ Phân tích Lát cắt Chế độ Hỏng hóc (Failure-Mode Recall):")
                mode_rows = []
                for m_col, m_info in failure_modes_data.items():
                    mode_rows.append(
                        {
                            "Cơ chế hỏng": m_info.get("description", m_col),
                            "Tổng số ca test": m_info.get("total_test_failures", 0),
                            "Phát hiện": m_info.get("detected_by_policy", 0),
                            "Tỷ lệ Recall": m_info.get("detection_percentage", "N/A"),
                        }
                    )
                st.table(pd.DataFrame(mode_rows))


if __name__ == "__main__":
    main()
