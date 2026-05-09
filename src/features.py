"""Static feature extraction for source-code snippets."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "eval",
    "exec",
    "pickle",
    "subprocess",
    "os.system",
    "strcpy",
    "sprintf",
    "gets",
    "input",
    "shell=True",
)


def _safe_code(code: str | None) -> str:
    return code or ""


def count_suspicious_keywords(code: str) -> int:
    """Count occurrences of known risky API or code patterns."""
    lowered = code.lower()
    return sum(lowered.count(keyword.lower()) for keyword in SUSPICIOUS_KEYWORDS)


def extract_features_from_code(code: str | None) -> dict[str, Any]:
    """Extract lightweight static features from a single code snippet."""
    text = _safe_code(code)
    lines = text.splitlines() or ([text] if text else [])
    non_empty_lines = [line for line in lines if line.strip()]
    char_count = len(text)
    line_count = len(lines)

    return {
        "char_count": char_count,
        "line_count": line_count,
        "avg_line_length": char_count / line_count if line_count else 0.0,
        "function_call_count": len(re.findall(r"\b[A-Za-z_]\w*\s*\(", text)),
        "loop_count": len(re.findall(r"\b(for|while)\b", text)),
        "conditional_count": len(re.findall(r"\b(if|elif|else|switch|case)\b", text)),
        "try_except_count": len(re.findall(r"\b(try|except|catch|finally)\b", text)),
        "comment_line_count": sum(
            1 for line in non_empty_lines if line.strip().startswith(("#", "//", "/*", "*"))
        ),
        "suspicious_keyword_count": count_suspicious_keywords(text),
    }


def extract_features_dataframe(
    dataframe: pd.DataFrame,
    code_column: str = "code",
) -> pd.DataFrame:
    """Return a dataframe of static features for each row of code."""
    if code_column not in dataframe.columns:
        raise KeyError(f"Missing required code column: {code_column}")

    features = dataframe[code_column].apply(extract_features_from_code)
    return pd.DataFrame(features.tolist(), index=dataframe.index)
