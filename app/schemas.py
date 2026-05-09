"""Pydantic schemas for the FastAPI inference service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request body for code-risk classification."""

    code: str = Field(..., min_length=1, description="Source-code snippet to classify.")

    @field_validator("code")
    @classmethod
    def code_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only snippets."""
        if not value.strip():
            raise ValueError("code must not be blank")
        return value


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    label: Literal["clean", "buggy"]
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["ok"] = "ok"
    model_loaded: bool
