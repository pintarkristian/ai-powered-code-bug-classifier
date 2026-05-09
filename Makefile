SHELL := /bin/bash

.PHONY: install test lint format clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests -v

lint:
	ruff check src app tests

format:
	black src app tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
