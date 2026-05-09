"""Train a PyTorch/Hugging Face Transformer classifier for code-risk detection.

The training entrypoint fine-tunes a sequence-classification model, such as
``microsoft/codebert-base``, on processed CSV files containing ``code`` and
``label`` columns. It intentionally keeps defaults small enough for laptop
experiments while still exposing the most important training knobs through CLI
arguments.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


class CodeDataset:
    """Simple PyTorch-compatible dataset for tokenized code snippets."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Transformer training."""
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer code bug classifier.")
    parser.add_argument("--train", type=Path, required=True, help="Path to processed train.csv.")
    parser.add_argument("--valid", type=Path, required=True, help="Path to processed valid.csv.")
    parser.add_argument("--model-name", default="microsoft/codebert-base", help="HF model checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/codebert-bug-classifier"),
        help="Directory where the model and tokenizer are saved.",
    )
    parser.add_argument("--epochs", type=float, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train/eval batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum tokenized sequence length.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/codebert_metrics.json"),
        help="Path for evaluation metrics JSON output.",
    )
    return parser


def configure_logging() -> None:
    """Configure clear console logging for training progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_split(path: Path) -> pd.DataFrame:
    """Load a processed split and validate required columns."""
    LOGGER.info("Loading split from %s", path)
    dataframe = pd.read_csv(path)
    required_columns = {"code", "label"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    dataframe = dataframe.loc[:, ["code", "label"]].dropna(subset=["code", "label"]).copy()
    dataframe["code"] = dataframe["code"].astype(str)
    dataframe["label"] = dataframe["label"].astype(int)

    invalid_labels = sorted(set(dataframe["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{path} contains non-binary labels: {invalid_labels}")
    if dataframe.empty:
        raise ValueError(f"{path} contains no usable rows.")

    LOGGER.info("Loaded %d rows from %s", len(dataframe), path)
    return dataframe


def tokenize_dataframe(dataframe: pd.DataFrame, tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Tokenize code snippets for Transformer sequence classification."""
    return tokenizer(
        dataframe["code"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax over class logits."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute classification metrics for Hugging Face Trainer."""
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = softmax(np.asarray(logits))
    predictions = np.argmax(probabilities, axis=1)
    labels = np.asarray(labels)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }

    try:
        if probabilities.shape[1] == 2 and len(np.unique(labels)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        LOGGER.warning("ROC-AUC could not be computed for this evaluation split.")

    return metrics


def build_training_arguments(args: argparse.Namespace) -> Any:
    """Build TrainingArguments while handling minor Transformers API differences."""
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "logging_steps": 25,
        "save_total_limit": 2,
        "report_to": "none",
        "load_best_model_at_end": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters

    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in parameters:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in parameters:
        kwargs["logging_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def save_metrics(metrics: dict[str, Any], metrics_output: Path) -> Path:
    """Save metrics as JSON using stable, simple keys."""
    cleaned = {
        key.replace("eval_", ""): float(value) if isinstance(value, (int, float, np.floating)) else value
        for key, value in metrics.items()
        if not key.endswith("_runtime")
        and not key.endswith("_samples_per_second")
        and not key.endswith("_steps_per_second")
        and key != "epoch"
    }
    return write_json(cleaned, metrics_output)


def main() -> None:
    """Run Transformer fine-tuning from the command line."""
    configure_logging()
    args = build_arg_parser().parse_args()

    # Lazy imports keep this module importable in lightweight test environments.
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Hugging Face Transformers are required for transformer training. "
            "Install project dependencies first with: pip install -r requirements.txt"
        ) from exc

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.metrics_output.parent)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device selected by Trainer: %s", device_name)
    LOGGER.info("Loading tokenizer and model: %s", args.model_name)

    train_df = load_split(args.train)
    valid_df = load_split(args.valid)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    LOGGER.info("Tokenizing training and validation data with max_length=%d", args.max_length)
    train_encodings = tokenize_dataframe(train_df, tokenizer, args.max_length)
    valid_encodings = tokenize_dataframe(valid_df, tokenizer, args.max_length)

    train_dataset = CodeDataset(train_encodings, train_df["label"].tolist())
    valid_dataset = CodeDataset(valid_encodings, valid_df["label"].tolist())

    training_args = build_training_arguments(args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    LOGGER.info(
        "Starting training: epochs=%s batch_size=%d learning_rate=%s",
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )
    trainer.train()

    LOGGER.info("Evaluating on validation split")
    eval_metrics = trainer.evaluate(eval_dataset=valid_dataset)

    LOGGER.info("Saving model and tokenizer to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics_path = save_metrics(eval_metrics, args.metrics_output)
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
