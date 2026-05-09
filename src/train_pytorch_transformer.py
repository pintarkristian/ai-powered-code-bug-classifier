"""PyTorch + Hugging Face Transformer training entrypoint.

This file intentionally contains only a minimal placeholder implementation. Full
fine-tuning code belongs in a later milestone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import ensure_dir, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Transformer placeholder.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--output-dir", type=Path, default=Path("models/codebert-bug-classifier"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    # Import lazily so this module can be imported without optional ML packages installed.
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Hugging Face Transformers are required for transformer training. "
            "Install requirements.txt first."
        ) from exc

    ensure_dir(args.output_dir)
    metrics_path = Path("reports/codebert_metrics.json")
    write_json(
        {
            "status": "placeholder",
            "message": "Transformer fine-tuning is not implemented yet.",
            "train": str(args.train),
            "valid": str(args.valid),
            "model_name": args.model_name,
            "output_dir": str(args.output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        metrics_path,
    )
    print(f"Wrote placeholder metrics to {metrics_path}")


if __name__ == "__main__":
    main()
