"""Pydantic schemas for predict and drift endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequestSchema(BaseModel):
    """Request body for batch inference."""

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of input records (one dict per row, column → value).",
        examples=[[{"age": 35, "salary": 50000, "tenure": 3}]],
    )


class PredictResponseSchema(BaseModel):
    """Response body for a batch inference request."""

    predictions: list[float | int | str] = Field(
        ...,
        description="Predicted values (class label or regression value per row).",
    )
    probabilities: list[float] | None = Field(
        default=None,
        description="Probability of the positive class (classification only).",
    )
    drift_alert: bool = Field(
        default=False,
        description="True if data drift was detected vs. the training distribution.",
    )
    drift_summary: dict[str, Any] | None = Field(
        default=None,
        description="Drift report summary (PSI scores, KL divergence, severity).",
    )
