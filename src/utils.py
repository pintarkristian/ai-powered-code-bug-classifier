"""Shared utilities for files, JSON, and logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


LOGGER_NAME = "ai_code_bug_classifier"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a configured logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to a JSON file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path
