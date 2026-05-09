"""TensorFlow/Keras baseline training entrypoint.

This file intentionally contains only a minimal placeholder implementation. Full
training code belongs in a later milestone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TensorFlow baseline placeholder.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/tensorflow_baseline.keras"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    # Import lazily so this module can be imported without TensorFlow installed.
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for training. Install requirements.txt first."
        ) from exc

    metrics_path = Path("reports/tensorflow_metrics.json")
    write_json(
        {
            "status": "placeholder",
            "message": "TensorFlow baseline training is not implemented yet.",
            "train": str(args.train),
            "valid": str(args.valid),
            "output": str(args.output),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        metrics_path,
    )
    print(f"Wrote placeholder metrics to {metrics_path}")


if __name__ == "__main__":
    main()
