.PHONY: install train serve test lint clean pipeline

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

train:
	python -m src.pipeline

serve:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

pipeline:
	python -m src.pipeline

clean:
	rm -rf models/*.joblib data/processed/*.csv reports/*.html reports/*.json mlruns/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
