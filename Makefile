# =============================================================================
# Machine Failure Risk & Maintenance Decision System
# Makefile chuẩn cho Linux/macOS/Windows (Git Bash / WSL)
# =============================================================================

.PHONY: help setup download train evaluate serve dashboard test clean full-pipeline docker-build docker-run lint

# Mặc định: in trợ giúp
help:
	@echo "================== Makefile Targets =================="
	@echo "  make setup            - Cài đặt thư viện Python từ requirements.txt"
	@echo "  make download         - Tải dữ liệu AI4I 2020 từ UCI Repository"
	@echo "  make train            - Huấn luyện mô hình và sinh artifact"
	@echo "  make evaluate         - Đánh giá mô hình trên tập Test độc lập"
	@echo "  make full-pipeline    - Chạy trọn vẹn: download + train + evaluate"
	@echo "  make serve            - Khởi chạy FastAPI service (port 8000)"
	@echo "  make dashboard        - Khởi chạy Streamlit dashboard (port 8501)"
	@echo "  make test             - Chạy bộ test tự động với pytest"
	@echo "  make clean            - Xóa artifact (model, reports) - KHÔNG xóa data thô"
	@echo "  make docker-build     - Build Docker image (multi-stage)"
	@echo "  make docker-run       - Chạy Docker container"
	@echo "  make lint             - Kiểm tra cú pháp (cần cài ruff: pip install ruff)"
	@echo "====================================================="

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

download:
	python -m scripts.download_data

train:
	python -m src.train

evaluate:
	python -m src.evaluate

# Chạy trọn vẹn pipeline từ đầu đến cuối
full-pipeline: download train evaluate
	@echo ""
	@echo "[OK] Full pipeline hoàn tất!"
	@echo "     - Dữ liệu: data/raw/ai4i2020.csv"
	@echo "     - Model:   models/model.joblib"
	@echo "     - Reports: reports/validation_metrics.json + reports/final_test_metrics.json"

serve:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

test:
	python -m pytest -v

# Dọn dẹp artifact (giữ lại data thô)
clean:
	@echo "[CLEAN] Đang xóa artifact..."
	rm -rf models/*.joblib models/*.pkl models/*.bin
	rm -rf artifacts/champion/*.joblib artifacts/champion/*.pkl
	rm -rf reports/*.json
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "[OK] Đã xóa model artifacts, reports và cache."

# Docker
docker-build:
	docker build -t machine-failure-risk-system:latest .

docker-run:
	docker run -d -p 8000:8000 --name risk-service machine-failure-risk-system:latest

docker-stop:
	docker stop risk-service && docker rm risk-service

# Lint (tùy chọn)
lint:
	@command -v ruff >/dev/null 2>&1 || { echo "Cài ruff: pip install ruff"; exit 1; }
	ruff check src/ tests/ scripts/
