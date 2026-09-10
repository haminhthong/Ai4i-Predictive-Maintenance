FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "fastapi>=0.95.0" \
        "uvicorn[standard]>=0.20.0" \
        "pydantic>=2.0.0" \
        "scikit-learn==1.7.1" \
        "pandas>=2.0.0" \
        "numpy>=1.24.0" \
        "joblib>=1.2.0"

COPY src ./src
COPY artifacts ./artifacts

RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
