"""Project configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    project_name: str = "AI Code Bug Classifier"
    api_version: str = "0.1.0"
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(project_root / "models")))
    reports_dir: Path = project_root / "reports"
    default_model_name: str = os.getenv("DEFAULT_MODEL_NAME", "heuristic-placeholder")
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
