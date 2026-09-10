.PHONY: help setup download train evaluate full-pipeline serve dashboard test lint docker-build docker-run docker-stop clean

help:
	@echo "make setup         - Cai dependencies"
	@echo "make download      - Tai dataset AI4I"
	@echo "make train         - Chon, calibrate va luu model"
	@echo "make evaluate      - Danh gia tren Test hold-out"
	@echo "make full-pipeline - Download + train + evaluate"
	@echo "make serve         - Chay FastAPI tai cong 8000"
	@echo "make dashboard     - Chay Streamlit tai cong 8501"
	@echo "make test          - Chay pytest"
	@echo "make lint          - Ruff check va format check"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install ruff==0.9.7

download:
	python -m scripts.download_data

train:
	python -m src.train

evaluate:
	python -m src.evaluate

full-pipeline: download train evaluate
	@echo "Full pipeline hoan tat."

serve:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app.py --server.port 8501 --server.address 0.0.0.0

test:
	python -B -m pytest -q

lint:
	python -m ruff check --no-cache src app.py tests scripts
	python -m ruff format --check --no-cache src app.py tests scripts

docker-build:
	docker build -t ai4i-maintenance-risk-triage:latest .

docker-run:
	docker run --rm -p 8000:8000 --name ai4i-risk-triage ai4i-maintenance-risk-triage:latest

docker-stop:
	docker stop ai4i-risk-triage

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('artifacts').glob('*.joblib')]"
	python -c "from pathlib import Path; [p.unlink() for p in Path('artifacts').glob('*.json')]"
	rm -rf .pytest_cache .ruff_cache .mypy_cache
