.PHONY: setup download train evaluate serve dashboard test

setup:
	python -m pip install -r requirements.txt

download:
	python scripts/download_data.py

train:
	python -m src.train

evaluate:
	python -m src.evaluate

serve:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

dashboard:
	streamlit run app.py

test:
	python -m pytest -v
