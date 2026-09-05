"""Giao diện Dashboard trực quan hoá Hệ thống Dự báo Rủi ro & Quyết định Bảo trì (Machine Failure Risk System).

Ứng dụng hỗ trợ:
- Điều chỉnh thông số cảm biến vận hành thời gian thực.
- Gọi mô hình ML Pipeline để tính xác suất rủi ro hỏng máy (Snapshot Risk Classification).
- Kiểm tra Out-Of-Distribution (OOD) so với khoảng phân bố tập huấn luyện.
- Hiển thị phân cấp rủi ro (HIGH, MEDIUM, LOW), phát cảnh báo bảo trì tối ưu chi phí.
- Phân định minh bạch giữa Mã lý do vận hành chuyên gia (Operational Reason Codes) và Giải thích mô hình ML (Model Explanations).
- Trực quan hóa Kiến trúc Canonical 7 Giai đoạn và Bảng so sánh chiến lược ngưỡng (Threshold Ablation Study).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from src.api import (
    SensorPayload,
    build_feature_row,
    check_out_of_distribution,
    generate_model_explanations,
    generate_operational_reason_codes,
    get_model_config_and_ranges,
)

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Machine Failure Risk & Maintenance Decision System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tùy chỉnh giao diện hiện đại
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
    .risk-high {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-medium {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        color: #92400E;
    }
    .risk-low {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        color: #166534;
    }
    .ood-box {
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


def main():
    st.markdown(
        '<div class="main-header">⚙️ Machine Failure Risk & Maintenance Decision Platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Leakage-safe machine-failure risk scoring system with calibrated probabilities, cost-sensitive maintenance decisions, operational reason codes and production-oriented serving.</div>',
        unsafe_allow_html=True,
    )

    # Collapsible 7-Stage Canonical Pipeline Visual
    with st.expander("📌 Sơ đồ Kiến trúc Canonical 7 Giai đoạn (Canonical 7-Stage Pipeline)", expanded=False):
        st.code(
            """
1. DATA INGESTION
   AI4I sensor observations
        ↓
2. ANTI-LEAKAGE & FEATURE ENGINEERING
   Drop: Machine failure target, TWF/HDF/PWF/OSF/RNF, UDI, Product ID
   Engineer: temperature_delta, mechanical_power (Watts), wear_load_interaction
        ↓
3. DATA PARTITION
   Stratified random split (Train 64% / Validation 16% / Test 20%)
        ↓
4. MODEL DEVELOPMENT & CALIBRATION
   Benchmark Model Zoo (Logistic, Random Forest, HistGB) + Sigmoid Calibration (PR-AUC / Brier trade-off)
        ↓
5. DECISION OPTIMIZATION
   Validation probabilities + Cost matrix (FN=5x, FP=1x) + Maintenance capacity sweep -> Minimum expected cost threshold
        ↓
6. FINAL TEST EVALUATION
   Untouched Test evaluation: PR-AUC, ROC-AUC, Precision, Recall, F1, Brier, ECE, Alert Rate, Test Expected Cost & Ablation
        ↓
7. SERVING & API
   Sensor Request -> Shared preprocessing -> Risk Probability + Model Explanation -> Decision Policy -> Reason Codes -> FastAPI / Streamlit
            """,
            language="text",
        )

    # Nạp mô hình, cấu hình và feature ranges
    try:
        model, config, feature_ranges = get_model_config_and_ranges()
        model_ready = True
    except Exception as exc:  # noqa: BLE001 - ranh giới UI phải hiển thị mọi lỗi artifact
        model_ready = False
        st.error(
            f"⚠️ Chưa tìm thấy mô hình huấn luyện! Vui lòng thực thi `python -m src.train` trước. Chi tiết: {exc}"
        )

    # Nạp báo cáo test metrics nếu có
    test_metrics = {}
    test_metrics_path = Path("reports/test_metrics.json")
    if test_metrics_path.exists():
        try:
            test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Không thể đọc báo cáo đánh giá: {exc}")

    # Sidebar: Nhập dữ liệu cảm biến
    st.sidebar.header("🎛️ Thông số cảm biến thời gian thực")

    machine_type = st.sidebar.selectbox(
        "Loại chất lượng máy (Type)",
        options=["L", "M", "H"],
        index=1,
        help="L: Low quality, M: Medium quality, H: High quality",
    )

    air_temp = st.sidebar.slider(
        "Nhiệt độ không khí (Air Temp - K)",
        min_value=290.0,
        max_value=310.0,
        value=300.0,
        step=0.1,
    )

    proc_temp = st.sidebar.slider(
        "Nhiệt độ vận hành (Process Temp - K)",
        min_value=300.0,
        max_value=320.0,
        value=310.0,
        step=0.1,
    )

    speed_rpm = st.sidebar.slider(
        "Tốc độ quay (Rotational Speed - RPM)",
        min_value=1100,
        max_value=2900,
        value=1500,
        step=10,
    )

    torque_nm = st.sidebar.slider(
        "Mô-men xoắn (Torque - Nm)",
        min_value=3.0,
        max_value=90.0,
        value=40.0,
        step=0.5,
    )

    tool_wear = st.sidebar.slider(
        "Độ mòn công cụ tích lũy (Tool Wear - phút)",
        min_value=0,
        max_value=260,
        value=120,
        step=1,
    )

    # Hiển thị tổng quan metrics hệ thống
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        champion_name = config.get("selected_model", "N/A") if model_ready else "N/A"
        st.metric("Champion Model", champion_name)
    with col2:
        st.metric("PR-AUC (Test)", f"{test_metrics.get('pr_auc', 0.0):.4f}" if test_metrics else "N/A")
    with col3:
        st.metric("Brier / ECE", f"{test_metrics.get('brier', 0.0):.4f} / {test_metrics.get('ece', 0.0):.4f}" if test_metrics else "N/A")
    with col4:
        threshold_val = config.get("threshold", 0.5) if model_ready else 0.5
        st.metric("Ngưỡng tối ưu (θ*)", f"{threshold_val:.4f}")
    with col5:
        cost_1k = test_metrics.get("cost_per_1000_machines", 0.0) if test_metrics else 0.0
        st.metric("Cost / 1,000 máy", f"${cost_1k:.2f}")

    st.markdown("---")

    if model_ready:
        # Tạo SensorPayload
        payload = SensorPayload(
            Type=machine_type,
            air_temperature_k=air_temp,
            process_temperature_k=proc_temp,
            rotational_speed_rpm=speed_rpm,
            torque_nm=torque_nm,
            tool_wear_min=tool_wear,
        )

        # Tính toán đặc trưng vật lý chuẩn
        temp_delta = proc_temp - air_temp
        angular_vel = speed_rpm * (2.0 * math.pi / 60.0)
        mech_power = torque_nm * angular_vel
        wear_load = tool_wear * torque_nm

        # Ánh xạ feature row
        feature_df = build_feature_row(payload, config["features"])

        # Dự báo xác suất & kiểm tra OOD
        failure_prob = float(model.predict_proba(feature_df)[0, 1])
        is_ood, ood_features = check_out_of_distribution(feature_df, feature_ranges)
        is_alert = failure_prob >= threshold_val

        if failure_prob >= max(threshold_val, 0.7):
            risk_tier = "HIGH"
            card_class = "risk-high"
        elif failure_prob >= threshold_val:
            risk_tier = "MEDIUM"
            card_class = "risk-medium"
        else:
            risk_tier = "LOW"
            card_class = "risk-low"

        reason_codes = generate_operational_reason_codes(payload)
        explanations = generate_model_explanations(model, feature_df)

        # Layout hiển thị kết quả
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("📊 Kết quả Dự báo & Quyết định Bảo trì")

            if is_ood:
                st.markdown(
                    f"""
                    <div class="ood-box">
                        <b>⚠️ CẢNH BÁO OUT-OF-DISTRIBUTION (OOD):</b> Các thông số nằm ngoài dải phân bố tập huấn luyện (P0.5-P99.5):
                        <ul>{"".join([f"<li>{item}</li>" for item in ood_features])}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="{card_class}">
                    <h3>Mức độ Rủi ro: {risk_tier}</h3>
                    <p style="font-size: 1.6rem; font-weight: bold; margin-bottom: 0.2rem;">
                        Xác suất hỏng máy: {failure_prob * 100:.2f}%
                    </p>
                    <p style="font-size: 1.1rem;">
                        <b>Quyết định Bảo trì:</b> {"🔴 CẢNH BÁO DỪNG MÁY / KIỂM TRA BẢO TRÌ" if is_alert else "🟢 AN TOÀN: CHO PHÉP MÁY VẬN HÀNH BÌNH THƯỜNG"}
                    </p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.85;">
                        Giả định ma trận chi phí: FN={config.get('false_negative_cost', 5.0)}x, FP={config.get('false_positive_cost', 1.0)}x | Ngưỡng quyết định θ*={threshold_val:.4f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(failure_prob)

            st.markdown("#### 🚨 Mã Lý do Vận hành (Operational Reason Codes):")
            st.caption("Các mã cảnh báo độc lập dựa trên quy tắc ngưỡng chuyên gia vận hành nhà máy:")
            if reason_codes:
                for code in reason_codes:
                    st.warning(f"• **{code}**")
            else:
                st.success("• Không phát hiện bất thường vận hành theo quy tắc chuyên gia.")

            st.markdown("#### 🔍 Giải thích Mô hình ML (Model Feature Contributions):")
            st.caption("Mức độ đóng góp của từng đặc trưng đến xác suất rủi ro dự báo:")
            if explanations and "contribution" in explanations[0]:
                df_exp = pd.DataFrame(explanations)
                st.dataframe(df_exp, use_container_width=True)
            else:
                st.info("Mô hình dạng cây/ensemble hiện không tính trực tiếp tuyến tính contribution.")

        with right_col:
            st.subheader("📐 Đặc trưng Vật lý Tính toán (Physics-Based Features)")

            df_features = pd.DataFrame(
                {
                    "Đặc trưng vật lý": [
                        "Chênh lệch nhiệt độ (temperature_delta)",
                        "Công suất cơ học thực (mechanical_power)",
                        "Tương tác mòn-tải (wear_load_interaction)",
                    ],
                    "Giá trị tính toán": [
                        f"{temp_delta:.2f} K",
                        f"{mech_power:.2f} Watts (J/s)",
                        f"{wear_load:.2f} min·Nm",
                    ],
                    "Cơ sở vật lý chuẩn": [
                        "ΔT = T_process - T_air (Nhiệt sinh ra do ma sát)",
                        "P = Torque × Angular Velocity (Watts thực tế)",
                        "Tương tác tích lũy độ mòn công cụ và mô-men xoắn",
                    ],
                }
            )
            st.table(df_features)

            st.markdown("#### 📊 Threshold Strategy Ablation Study (Đánh giá trên tập Test độc lập):")
            ablation_data = test_metrics.get("threshold_ablation_study", {})
            if ablation_data:
                rows = []
                for strat_name, res in ablation_data.items():
                    rows.append(
                        {
                            "Chiến lược Ngưỡng": strat_name,
                            "Ngưỡng (θ)": f"{res['threshold']:.4f}",
                            "Recall": f"{res['recall']:.2%}",
                            "Alert Rate": f"{res['alert_rate']:.2%}",
                            "Cost / 1,000 máy": f"${res['cost_per_1000_machines']:.2f}",
                        }
                    )
                st.table(pd.DataFrame(rows))
            else:
                st.info("Chưa có báo cáo Ablation study trên tập Test.")


if __name__ == "__main__":
    main()

