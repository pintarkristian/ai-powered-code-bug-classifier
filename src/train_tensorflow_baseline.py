"""Train a lightweight TensorFlow/Keras baseline for code-risk classification.

The model is intentionally small enough to run on a laptop. It uses Keras'
``TextVectorization`` layer to tokenize code snippets, followed by an embedding
layer, a bidirectional LSTM, dense layers, and a sigmoid output for binary
classification.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import write_json

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"code", "label"}
DEFAULT_METRICS_PATH = Path("reports/tensorflow_metrics.json")


def configure_logging() -> None:
    """Configure consistent command-line logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_reproducibility_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""

    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow is imported lazily inside this function so importing the module
    # remains lightweight and does not require TensorFlow in non-training tests.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover - depends on TF/platform support.
        LOGGER.debug("TensorFlow deterministic ops were not enabled: %s", exc)


def load_training_frame(path: Path) -> pd.DataFrame:
    """Load and validate a processed CSV file with ``code`` and ``label`` columns.

    Args:
        path: Path to a CSV file.

    Returns:
        A clean DataFrame containing only the required columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or labels are invalid.
    """

    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.loc[:, ["code", "label"]].dropna(subset=["code", "label"]).copy()
    df["code"] = df["code"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    invalid_labels = sorted(set(df["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Labels must be binary 0/1. Found: {invalid_labels}")

    if df.empty:
        raise ValueError(f"{path} does not contain any usable rows")

    return df


def build_model(
    *,
    max_tokens: int,
    sequence_length: int,
    embedding_dim: int = 64,
    lstm_units: int = 32,
    dense_units: int = 32,
) -> Any:
    """Build a compact TensorFlow/Keras binary text classifier.

    Args:
        max_tokens: Maximum vocabulary size for TextVectorization.
        sequence_length: Fixed token sequence length.
        embedding_dim: Embedding dimension.
        lstm_units: Number of units in the bidirectional LSTM.
        dense_units: Number of units in the hidden dense layer.

    Returns:
        A compiled ``tf.keras.Model``.
    """

    import tensorflow as tf

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=sequence_length,
        name="code_vectorizer",
    )

    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="code")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
        input_dim=max_tokens,
        output_dim=embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units),
        name="bidirectional_lstm",
    )(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="tensorflow_code_bug_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def summarize_history(history: Any) -> dict[str, Any]:
    """Convert a Keras history object into JSON-serializable final metrics."""

    final_metrics: dict[str, float] = {}
    best_validation_metrics: dict[str, float] = {}

    for metric_name, values in history.history.items():
        if not values:
            continue
        numeric_values = [float(value) for value in values]
        final_metrics[metric_name] = numeric_values[-1]
        if metric_name.startswith("val_"):
            if "loss" in metric_name:
                best_validation_metrics[f"best_{metric_name}"] = min(numeric_values)
            else:
                best_validation_metrics[f"best_{metric_name}"] = max(numeric_values)

    return {
        "final": final_metrics,
        "best_validation": best_validation_metrics,
        "epochs_ran": len(history.epoch),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train the TensorFlow baseline and persist the model plus metrics."""

    import tensorflow as tf

    LOGGER.info("Setting reproducibility seed to %s", args.seed)
    set_reproducibility_seed(args.seed)

    LOGGER.info("Loading training data from %s", args.train)
    train_df = load_training_frame(args.train)
    LOGGER.info("Loaded %s training rows", len(train_df))

    LOGGER.info("Loading validation data from %s", args.valid)
    valid_df = load_training_frame(args.valid)
    LOGGER.info("Loaded %s validation rows", len(valid_df))

    train_text = train_df["code"].astype(str).to_numpy()
    train_labels = train_df["label"].astype("float32").to_numpy()
    valid_text = valid_df["code"].astype(str).to_numpy()
    valid_labels = valid_df["label"].astype("float32").to_numpy()

    LOGGER.info(
        "Building model with max_tokens=%s and sequence_length=%s",
        args.max_tokens,
        args.sequence_length,
    )
    model = build_model(
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
    )

    vectorizer = model.get_layer("code_vectorizer")
    LOGGER.info("Adapting TextVectorization vocabulary")
    vectorizer.adapt(train_text)

    LOGGER.info(
        "Starting training for %s epoch(s), batch_size=%s",
        args.epochs,
        args.batch_size,
    )
    history = model.fit(
        train_text,
        train_labels,
        validation_data=(valid_text, valid_labels),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving TensorFlow baseline model to %s", args.output)
    model.save(args.output)

    metrics = summarize_history(history)
    metrics.update(
        {
            "status": "trained",
            "model_path": str(args.output),
            "train_path": str(args.train),
            "valid_path": str(args.valid),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "parameters": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "max_tokens": int(args.max_tokens),
                "sequence_length": int(args.sequence_length),
                "seed": int(args.seed),
                "tensorflow_version": tf.__version__,
            },
        }
    )

    metrics_path = args.metrics_output
    LOGGER.info("Saving TensorFlow metrics to %s", metrics_path)
    write_json(metrics, metrics_path)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for baseline training."""

    parser = argparse.ArgumentParser(
        description="Train TensorFlow/Keras baseline classifier."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to processed training CSV with code and label columns.",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path("data/processed/valid.csv"),
        help="Path to processed validation CSV with code and label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tensorflow_baseline.keras"),
        help="Path where the trained Keras model will be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path where training metrics JSON will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20_000,
        help="Maximum TextVectorization vocabulary size.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Fixed token sequence length for code snippets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    return parser


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    args = build_arg_parser().parse_args()

    try:
        train(args)
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install dependencies first with: "
            "pip install -r requirements.txt"
        ) from exc


if __name__ == "__main__":
    main()
