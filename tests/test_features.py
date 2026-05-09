"""Tests for static feature extraction."""

from __future__ import annotations

import pandas as pd

from src.features import extract_features_dataframe, extract_features_from_code


def test_empty_code_does_not_crash() -> None:
    features = extract_features_from_code("")
    assert features["char_count"] == 0
    assert features["line_count"] == 0


def test_suspicious_keyword_detection() -> None:
    features = extract_features_from_code("def run(x): return eval(x)")
    assert features["suspicious_keyword_count"] >= 1
    assert features["function_call_count"] >= 2


def test_feature_dataframe_has_expected_columns() -> None:
    dataframe = pd.DataFrame({"code": ["for i in range(3): print(i)"]})
    features = extract_features_dataframe(dataframe)
    assert "loop_count" in features.columns
    assert features.loc[0, "loop_count"] == 1
