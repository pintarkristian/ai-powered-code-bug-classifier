"""Tests for preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing import clean_dataset, normalize_columns, split_dataset


def test_normalize_columns_renames_inputs() -> None:
    raw = pd.DataFrame({"snippet": ["print('ok')"], "target": [0]})
    normalized = normalize_columns(raw, code_column="snippet", label_column="target")
    assert list(normalized.columns) == ["code", "label"]


def test_clean_dataset_drops_missing_short_and_duplicate_rows() -> None:
    raw = pd.DataFrame(
        {
            "code": ["print('ok')", "print('ok')", None, "x", "eval(user_input)"],
            "label": [0, 0, 1, 0, 1],
        }
    )
    cleaned = clean_dataset(raw)
    assert len(cleaned) == 2
    assert cleaned["code"].tolist() == ["print('ok')", "eval(user_input)"]
    assert cleaned["label"].tolist() == [0, 1]


def test_split_dataset_returns_three_non_empty_splits() -> None:
    dataframe = pd.DataFrame(
        {
            "code": [f"def f_{i}(): return {i}" for i in range(40)],
            "label": [0, 1] * 20,
        }
    )
    train, valid, test = split_dataset(dataframe, valid_size=0.1, test_size=0.1)
    assert len(train) + len(valid) + len(test) == len(dataframe)
    assert len(train) > len(valid) > 0
    assert len(test) > 0
