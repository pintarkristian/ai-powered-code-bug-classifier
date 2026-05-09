SHELL := /bin/bash

.PHONY: install test lint format train-tf clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests -v

lint:
	ruff check src app tests

format:
	black src app tests

train-tf:
	python -m src.train_tensorflow_baseline --train data/processed/train.csv --valid data/processed/valid.csv --output models/tensorflow_baseline.keras --epochs 3 --batch-size 16

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
