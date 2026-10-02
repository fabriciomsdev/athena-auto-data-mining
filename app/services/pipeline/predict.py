"""Inference service — loads a trained model and generates predictions."""
from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import structlog

from app.db.models.pipeline import Pipeline, PipelineStatus, ProblemType
from app.services.ingestion.storage import StorageService

logger = structlog.get_logger()


@dataclass
class PredictRequest:
    """Input payload for a batch prediction request."""

    pipeline_id: str
    records: list[dict[str, Any]]


@dataclass
class PredictResponse:
    """Output of a batch prediction request."""

    predictions: list[float | int | str]
    probabilities: list[float] | None
    drift_alert: bool
    drift_summary: dict[str, Any] | None


class PredictService:
    """
    Generates batch predictions from a completed AthenaMining pipeline.

    Workflow:
    1. Validate pipeline is in COMPLETED status.
    2. Download the fitted sklearn preprocessing pipeline from MinIO.
    3. Download the trained model artifact from MinIO.
    4. Transform input records through the same preprocessing pipeline.
    5. Generate predictions (+ probabilities for classification).
    6. Return PredictResponse (drift computed separately by DriftService).
    """

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def predict(
        self,
        pipeline: Pipeline,
        records: list[dict[str, Any]],
        drift_summary: dict[str, Any] | None = None,
        drift_alert: bool = False,
    ) -> PredictResponse:
        """
        Run batch inference for the given pipeline and input records.

        Args:
            pipeline: A fully completed Pipeline ORM record.
            records: List of input row dicts (column → value).
            drift_summary: Pre-computed drift report summary (optional).
            drift_alert: Whether any drift was detected.

        Returns:
            PredictResponse with predictions, probabilities, and drift info.

        Raises:
            ValueError: If the pipeline is not in COMPLETED status.
        """
        if pipeline.status != PipelineStatus.COMPLETED:
            raise ValueError(
                f"Pipeline {pipeline.id} is not COMPLETED "
                f"(current status: {pipeline.status})."
            )
        if not records:
            raise ValueError("Prediction input records list is empty.")

        logger.info(
            "PredictService started",
            pipeline_id=str(pipeline.id),
            records=len(records),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ── Download preprocessing pipeline ─────────────────────────────
            assert pipeline.model_minio_path, "Pipeline has no model_minio_path."
            sk_pipeline = self._load_sklearn_pipeline(pipeline, tmp)

            # ── Download trained model ───────────────────────────────────────
            model = self._load_model(pipeline, tmp)

            # ── Transform input ──────────────────────────────────────────────
            import pandas as pd
            input_df = pd.DataFrame(records)
            X_transformed = sk_pipeline.transform(input_df)

            # ── Predict ──────────────────────────────────────────────────────
            predictions_raw = model.predict(X_transformed)
            predictions: list[float | int | str] = predictions_raw.tolist()

            # ── Probabilities (classification only) ──────────────────────────
            probabilities: list[float] | None = None
            if pipeline.problem_type == ProblemType.CLASSIFICATION:
                try:
                    proba_raw = model.predict_proba(X_transformed)
                    if proba_raw is not None:
                        # Return probability of the positive class for binary
                        if proba_raw.ndim == 2 and proba_raw.shape[1] == 2:
                            probabilities = proba_raw[:, 1].tolist()
                        else:
                            probabilities = proba_raw.max(axis=1).tolist()
                except Exception as exc:
                    logger.warning("Could not compute probabilities", error=str(exc))

        logger.info(
            "PredictService completed",
            pipeline_id=str(pipeline.id),
            predictions=len(predictions),
        )

        return PredictResponse(
            predictions=predictions,
            probabilities=probabilities,
            drift_alert=drift_alert,
            drift_summary=drift_summary,
        )

    def _load_sklearn_pipeline(self, pipeline: Pipeline, tmp: Path) -> Any:
        """Download and deserialise the sklearn preprocessing pipeline."""
        # The pipeline artifact path is stored alongside the model path
        pipeline_object_name = str(pipeline.model_minio_path).replace(
            "models/", "pipelines/"
        ).replace(".joblib", "_pipeline.joblib")

        try:
            local_path = self._storage.download_to_temp(pipeline_object_name, tmp)
        except Exception:
            # Fall back — if no separate pipeline artifact, passthrough
            logger.warning(
                "sklearn pipeline artifact not found, using passthrough",
                pipeline_id=str(pipeline.id),
            )
            from sklearn.pipeline import Pipeline as SKPipeline
            from sklearn.preprocessing import FunctionTransformer
            return SKPipeline([("passthrough", FunctionTransformer())])

        return joblib.load(local_path)

    def _load_model(self, pipeline: Pipeline, tmp: Path) -> Any:
        """Download and deserialise the trained model artifact."""
        local_path = self._storage.download_to_temp(
            str(pipeline.model_minio_path), tmp
        )
        return joblib.load(local_path)
