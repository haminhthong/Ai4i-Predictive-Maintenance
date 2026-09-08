# =============================================================================
# Stage 1: Builder - Huấn luyện mô hình và sinh artifact
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài đặt dependencies cho cả train và serve
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Copy source code để train
COPY . .

# Bước 1: Tải dữ liệu thô
RUN python -m scripts.download_data

# Bước 2: Huấn luyện mô hình -> sinh release bundle và mirror legacy
RUN python -m src.train

# =============================================================================
# Stage 2: Runtime - Image cuối cùng (nhẹ, không chứa source code không cần thiết)
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Chỉ cài dependency cần cho API và release inference.
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "fastapi>=0.95.0" \
        "uvicorn[standard]>=0.20.0" \
        "pydantic>=2.0.0" \
        "scikit-learn>=1.2.0" \
        "pandas>=2.0.0" \
        "numpy>=1.24.0" \
        "joblib>=1.2.0"

# Copy mã nguồn và artifact đã train từ builder
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/releases ./releases
COPY --from=builder /app/models ./models
COPY --from=builder /app/artifacts ./artifacts

# Tạo user không phải root để tăng cường bảo mật
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

# CMD mặc định: chạy API service; dashboard chạy ngoài runtime image này.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
