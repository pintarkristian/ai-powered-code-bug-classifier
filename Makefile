.PHONY: install preprocess test lint format train-tf train-transformer evaluate api clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

preprocess:
	python -m src.data_preprocessing --hf-dataset google/code_x_glue_cc_defect_detection --output-dir data/processed

test:
	python -m pytest tests -q

lint:
	ruff check src app tests

format:
	black src app tests

train-tf:
	python -m src.train_tensorflow_baseline --train data/processed/train.csv --valid data/processed/valid.csv --output models/tensorflow_baseline.keras --epochs 3 --batch-size 16

train-transformer:
	python -m src.train_pytorch_transformer --train data/processed/train.csv --valid data/processed/valid.csv --model-name microsoft/codebert-base --output-dir models/codebert-bug-classifier --epochs 1 --batch-size 8 --max-length 256 --learning-rate 2e-5 --seed 42

evaluate:
	python -m src.evaluate --test data/processed/test.csv --model-dir models/codebert-bug-classifier --model-type transformer

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'dist', 'build']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
