"""Giao diện Dashboard trực quan hoá Dự báo Rủi ro Hỏng hóc Máy móc (Predictive Maintenance App).

Ứng dụng hỗ trợ:
- Điều chỉnh các chỉ số cảm biến vận hành thời gian thực.
- Gọi trực tiếp hàm dự báo từ mô hình AI4I ML Pipeline.
- Hiển thị xác suất rủi ro, phân cấp mức độ (HIGH, MEDIUM, LOW), phát cảnh báo dừng máy và sinh các mã lý do vận hành.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from src.api import (
    SensorPayload,
    build_feature_row,
    generate_operational_reason_codes,
    get_model_and_config,
)

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Predictive Maintenance AI Service",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        color: #991B1B;
    }
    .risk-medium {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        color: #92400E;
    }
    .risk-low {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.markdown(
        '<div class="main-header">⚙️ Predictive Maintenance & Failure Risk Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Hệ thống AI dự báo xác suất hỏng máy móc sản xuất và đưa ra đề xuất bảo trì theo chi phí nghiệp vụ</div>',
        unsafe_allow_html=True,
    )

    # Nạp mô hình và cấu hình
    try:
        model, config = get_model_and_config()
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Phiên bản Model", config.get("version", "v2") if model_ready else "N/A"
        )
    with col2:
        st.metric(
            "PR-AUC (Test)",
            f"{test_metrics.get('pr_auc', 0.0):.4f}" if test_metrics else "N/A",
        )
    with col3:
        st.metric(
            "Brier Score",
            f"{test_metrics.get('brier', 0.0):.4f}" if test_metrics else "N/A",
        )
    with col4:
        threshold_val = config.get("threshold", 0.5) if model_ready else 0.5
        st.metric("Ngưỡng cảnh báo (Cost-tuned)", f"{threshold_val:.4f}")

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

        # Tính toán đặc trưng phụ
        temp_delta = proc_temp - air_temp
        power_proxy = speed_rpm * torque_nm
        strain_proxy = tool_wear * torque_nm

        # Ánh xạ feature row
        feature_df = build_feature_row(payload, config["features"])

        # Dự báo xác suất
        failure_prob = float(model.predict_proba(feature_df)[0, 1])
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

        # Layout hiển thị kết quả
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("📊 Kết quả dự báo AI")

            st.markdown(
                f"""
                <div class="{card_class}">
                    <h3>Mức độ rủi ro: {risk_tier}</h3>
                    <p style="font-size: 1.5rem; font-weight: bold;">Xác suất hỏng máy: {failure_prob * 100:.2f}%</p>
                    <p><b>Trạng thái hệ thống:</b> {"🔴 CẢNH BÁO: CẦN BẢO TRÌ/KIỂM TRA" if is_alert else "🟢 AN TOÀN: MÁY HOẠT ĐỘNG BÌNH THƯỜNG"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(failure_prob)

            st.markdown("#### 🚨 Mã lý do vận hành (Reason Codes):")
            if reason_codes:
                for code in reason_codes:
                    st.warning(f"• **{code}**")
            else:
                st.success(
                    "• Không phát hiện bất thường vận hành theo quy tắc chuyên gia."
                )

        with right_col:
            st.subheader("📐 Đặc trưng vật lý tính toán (Feature Engineering)")

            df_features = pd.DataFrame(
                {
                    "Đặc trưng vật lý": [
                        "Nhiệt độ delta (Process - Air)",
                        "Công suất cơ học xấp xỉ (Speed x Torque)",
                        "Ứng suất mòn tích lũy (Tool Wear x Torque)",
                    ],
                    "Giá trị": [
                        f"{temp_delta:.2f} K",
                        f"{power_proxy:.2f} RPM·Nm",
                        f"{strain_proxy:.2f} min·Nm",
                    ],
                    "Giải thích kỹ thuật": [
                        "Biểu diễn mức tích nhiệt do ma sát vận hành",
                        "Chỉ số đại diện cho công suất hoạt động",
                        "Tải trọng lực cơ học tích lũy lên công cụ",
                    ],
                }
            )
            st.table(df_features)

            st.markdown("#### 💡 Giả định chi phí quyết định (Business Cost Matrix):")
            st.info(
                f"• Chi phí bỏ sót 1 sự cố (False Negative): **{config.get('false_negative_cost', 5.0)}x**\n"
                f"• Chi phí kiểm tra nhầm (False Positive): **{config.get('false_positive_cost', 1.0)}x**\n"
                f"• Ngưỡng cảnh báo tối ưu hóa tổng chi phí: **{threshold_val:.4f}**"
            )


if __name__ == "__main__":
    main()
