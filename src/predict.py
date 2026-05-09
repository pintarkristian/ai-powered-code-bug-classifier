"""Reusable prediction logic for code bug/risk classification."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from src.config import get_settings
from src.features import count_suspicious_keywords


class PredictionResult(TypedDict):
    """Dictionary returned by predictor implementations."""

    label: str
    risk_score: float
    model_name: str
    notes: list[str]


class CodeBugPredictor:
    """Placeholder predictor with simple heuristics.

    The class is intentionally lightweight so the API can run before a full model is
    trained. Later milestones can replace `_predict_with_heuristics` with TensorFlow
    or Transformer inference while preserving the public interface.
    """

    def __init__(self, model_path: str | Path | None = None, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_path = Path(model_path or settings.model_dir)
        self.model_name = model_name or settings.default_model_name
        self.is_loaded = self._detect_model_artifact()

    def _detect_model_artifact(self) -> bool:
        """Return true when a likely model artifact exists."""
        return self.model_path.exists() and any(self.model_path.iterdir()) if self.model_path.is_dir() else self.model_path.exists()

    def predict(self, code: str) -> PredictionResult:
        """Return a clean/buggy label, risk score, model name, and notes."""
        return self._predict_with_heuristics(code)

    def _predict_with_heuristics(self, code: str) -> PredictionResult:
        """Simple deterministic placeholder until trained model inference is added."""
        lowered = code.lower()
        notes: list[str] = []

        risky_patterns = {
            "eval(": "Use of eval detected",
            "exec(": "Use of exec detected",
            "shell=true": "shell=True subprocess risk detected",
            "os.system": "os.system usage detected",
            "pickle": "pickle usage detected",
            "strcpy": "Potential unsafe C string copy detected",
            "sprintf": "Potential unsafe C string formatting detected",
            "gets(": "Potential unsafe gets usage detected",
        }

        for pattern, note in risky_patterns.items():
            if pattern in lowered:
                notes.append(note)

        if "/" in code and "zero" not in lowered and "== 0" not in lowered and "!= 0" not in lowered:
            notes.append("Potential division-by-zero risk")

        suspicious_count = count_suspicious_keywords(code)
        base_score = 0.15
        risk_score = min(0.95, base_score + 0.18 * len(notes) + 0.08 * suspicious_count)
        label = "buggy" if risk_score >= 0.5 else "clean"

        return {
            "label": label,
            "risk_score": round(risk_score, 4),
            "model_name": self.model_name,
            "notes": notes,
        }
