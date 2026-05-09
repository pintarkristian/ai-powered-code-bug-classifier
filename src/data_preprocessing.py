"""Pandas-based preprocessing utilities for code classification data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.utils import ensure_dir

MIN_CODE_LENGTH = 5
MAX_CODE_LENGTH = 4096


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(path)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a JSON Lines dataset."""
    return pd.read_json(path, lines=True)


def normalize_columns(
    dataframe: pd.DataFrame,
    code_column: str = "code",
    label_column: str = "label",
) -> pd.DataFrame:
    """Normalize arbitrary code and label columns to `code` and `label`."""
    missing = [column for column in (code_column, label_column) if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    normalized = dataframe[[code_column, label_column]].rename(
        columns={code_column: "code", label_column: "label"}
    )
    return normalized


def clean_dataset(
    dataframe: pd.DataFrame,
    min_code_length: int = MIN_CODE_LENGTH,
    max_code_length: int = MAX_CODE_LENGTH,
) -> pd.DataFrame:
    """Clean and normalize a dataset with `code` and `label` columns."""
    required = {"code", "label"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    cleaned = dataframe.copy()
    cleaned["code"] = cleaned["code"].astype("string")
    cleaned = cleaned.dropna(subset=["code", "label"])
    cleaned["code"] = cleaned["code"].str.strip()
    cleaned = cleaned[cleaned["code"].str.len() >= min_code_length]
    cleaned["code"] = cleaned["code"].str.slice(0, max_code_length)
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned[cleaned["label"].isin([0, 1])]
    cleaned = cleaned.drop_duplicates(subset=["code"])
    return cleaned.reset_index(drop=True)


def split_dataset(
    dataframe: pd.DataFrame,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test dataframes."""
    if not 0 < valid_size < 1 or not 0 < test_size < 1:
        raise ValueError("valid_size and test_size must be between 0 and 1")
    if valid_size + test_size >= 1:
        raise ValueError("valid_size + test_size must be less than 1")

    stratify = dataframe["label"] if dataframe["label"].nunique() == 2 else None
    train_valid, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    relative_valid_size = valid_size / (1 - test_size)
    stratify_train_valid = (
        train_valid["label"] if train_valid["label"].nunique() == 2 else None
    )
    train, valid = train_test_split(
        train_valid,
        test_size=relative_valid_size,
        random_state=random_state,
        stratify=stratify_train_valid,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save train, validation, and test CSV files."""
    directory = ensure_dir(output_dir)
    paths = {
        "train": directory / "train.csv",
        "valid": directory / "valid.csv",
        "test": directory / "test.csv",
    }
    train.to_csv(paths["train"], index=False)
    valid.to_csv(paths["valid"], index=False)
    test.to_csv(paths["test"], index=False)
    return paths


def preprocess_file(
    input_path: str | Path,
    output_dir: str | Path,
    code_column: str = "code",
    label_column: str = "label",
    random_state: int = 42,
) -> dict[str, Path]:
    """Load, clean, split, and save a local CSV or JSONL dataset."""
    path = Path(input_path)
    if path.suffix.lower() == ".jsonl":
        raw = load_jsonl(path)
    else:
        raw = load_csv(path)

    normalized = normalize_columns(raw, code_column=code_column, label_column=label_column)
    cleaned = clean_dataset(normalized)
    train, valid, test = split_dataset(cleaned, random_state=random_state)
    return save_splits(train, valid, test, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Preprocess code classification data.")
    parser.add_argument("--input", type=Path, help="Local CSV or JSONL input file.")
    parser.add_argument("--output-dir", type=Path, default=get_settings().processed_data_dir)
    parser.add_argument("--code-column", default="code")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Placeholder for future Hugging Face dataset loading.",
    )
    return parser


def main() -> None:
    """Run preprocessing from the command line."""
    args = build_arg_parser().parse_args()
    if args.hf_dataset and not args.input:
        raise NotImplementedError(
            "Hugging Face dataset loading will be implemented in a later milestone. "
            "Use --input for the current skeleton."
        )
    if not args.input:
        raise ValueError("Provide --input for the current skeleton implementation.")

    paths = preprocess_file(
        input_path=args.input,
        output_dir=args.output_dir,
        code_column=args.code_column,
        label_column=args.label_column,
    )
    for split, path in paths.items():
        print(f"Saved {split}: {path}")


if __name__ == "__main__":
    main()
