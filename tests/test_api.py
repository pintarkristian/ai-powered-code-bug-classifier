"""Tests for the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["model_loaded"], bool)


def test_predict_endpoint_accepts_valid_code() -> None:
    response = client.post("/predict", json={"code": "def divide(a, b): return a / b"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"clean", "buggy"}
    assert 0 <= payload["risk_score"] <= 1
    assert "model_name" in payload
    assert isinstance(payload["notes"], list)


def test_predict_endpoint_rejects_blank_code() -> None:
    response = client.post("/predict", json={"code": "   "})
    assert response.status_code == 422
