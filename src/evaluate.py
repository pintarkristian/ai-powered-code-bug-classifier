"""Model evaluation entrypoint placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained model placeholder.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["tensorflow", "transformer"], required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/test_metrics.json"))
    return parser


def evaluate_placeholder(test_path: Path, model_dir: Path, model_type: str) -> dict[str, object]:
    """Return placeholder evaluation metrics."""
    return {
        "status": "placeholder",
        "message": "Evaluation is not implemented yet.",
        "test": str(test_path),
        "model_dir": str(model_dir),
        "model_type": model_type,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "pr_auc": None,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = evaluate_placeholder(args.test, args.model_dir, args.model_type)
    write_json(metrics, args.output)
    print(f"Wrote placeholder metrics to {args.output}")


if __name__ == "__main__":
    main()
