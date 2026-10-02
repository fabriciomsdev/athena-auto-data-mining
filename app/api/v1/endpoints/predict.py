"""Predict and drift endpoints."""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import polars as pl
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pipeline import Pipeline
from app.db.session import get_db
from app.schemas.predict import PredictRequestSchema, PredictResponseSchema
from app.services.ingestion.storage import StorageService
from app.services.pipeline.drift import DriftService
from app.services.pipeline.predict import PredictRequest, PredictService

logger = structlog.get_logger()
router = APIRouter()


@router.post(
    "/{pipeline_id}",
    response_model=PredictResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Run batch inference on a completed pipeline",
)
async def predict(
    pipeline_id: UUID,
    payload: PredictRequestSchema,
    db: AsyncSession = Depends(get_db),
) -> PredictResponseSchema:
    """
    Submit a list of records for batch inference.

    Drift is computed asynchronously in the background against the
    training distribution. If drift data is not yet available, the
    response will have ``drift_alert=false``.
    """
    pipeline = await db.get(Pipeline, pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found.")

    storage = StorageService()
    predict_svc = PredictService(storage)
    drift_svc = DriftService(storage)

    # ── Compute drift in background ───────────────────────────────────────────
    current_df = pl.DataFrame(payload.records)
    drift_report = await drift_svc.compute(pipeline, current_df)

    # ── Run inference ─────────────────────────────────────────────────────────
    response = await predict_svc.predict(
        pipeline=pipeline,
        records=payload.records,
        drift_summary=drift_report.to_dict() if drift_report.dataset_drift_detected else None,
        drift_alert=drift_report.dataset_drift_detected,
    )

    return PredictResponseSchema(
        predictions=response.predictions,
        probabilities=response.probabilities,
        drift_alert=response.drift_alert,
        drift_summary=response.drift_summary,
    )


@router.get(
    "/{pipeline_id}/drift-report",
    summary="Get the latest drift report for a pipeline",
)
async def get_drift_report(
    pipeline_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the most recent drift report computed for this pipeline."""
    pipeline = await db.get(Pipeline, pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found.")

    # TODO: Persist drift reports to DB (Phase 5b) and query latest here.
    # For now, returns a placeholder indicating no report is stored yet.
    return {
        "pipeline_id": str(pipeline_id),
        "message": "Drift reports will be persisted in Phase 5b (DriftReport model).",
    }
