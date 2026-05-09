"""FastAPI application for code bug/risk classification."""

from __future__ import annotations

from fastapi import FastAPI

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config import get_settings
from src.predict import CodeBugPredictor

settings = get_settings()
predictor = CodeBugPredictor(model_path=settings.model_dir)

app = FastAPI(
    title=settings.project_name,
    version=settings.api_version,
    description="Classify source-code snippets as clean or potentially buggy/risky.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and whether a trained model was loaded."""
    return HealthResponse(model_loaded=predictor.is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Classify a source-code snippet."""
    result = predictor.predict(request.code)
    return PredictionResponse(**result)
