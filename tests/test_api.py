"""Tests for the FastAPI app."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_predictor


class DummyPredictor:
    """Small test double that avoids loading a real Transformer model."""

    def __init__(self, is_loaded: bool = True) -> None:
        self.is_loaded = is_loaded

    def predict(self, code: str) -> dict:
        """Return a deterministic API-compatible prediction payload."""
        notes: list[str] = []
        risk_score = 0.82 if "/" in code else 0.12
        label = "buggy" if risk_score >= 0.5 else "clean"

        if "/" in code:
            notes.append("Potential division-by-zero risk")

        return {
            "label": label,
            "risk_score": risk_score,
            "model_name": "test-codebert",
            "notes": notes,
        }


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Create a test client with the predictor dependency mocked."""
    app.dependency_overrides[get_predictor] = lambda: DummyPredictor()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint_reports_ok_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_endpoint_reports_degraded_when_model_missing() -> None:
    app.dependency_overrides[get_predictor] = lambda: DummyPredictor(is_loaded=False)
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_predict_endpoint_accepts_valid_code(client: TestClient) -> None:
    response = client.post("/predict", json={"code": "def divide(a, b): return a / b"})

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "label": "buggy",
        "risk_score": 0.82,
        "model_name": "test-codebert",
        "notes": ["Potential division-by-zero risk"],
    }


def test_predict_endpoint_rejects_blank_code(client: TestClient) -> None:
    response = client.post("/predict", json={"code": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "code must not be empty"
