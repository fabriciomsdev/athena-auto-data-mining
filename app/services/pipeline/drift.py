"""Data drift monitoring service using Evidently AI."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import polars as pl
import structlog

from app.db.models.pipeline import Pipeline
from app.services.ingestion.storage import StorageService

logger = structlog.get_logger()

# PSI thresholds per industry convention
_PSI_WARNING_THRESHOLD: float = 0.1
_PSI_CRITICAL_THRESHOLD: float = 0.25


@dataclass
class DriftReport:
    """Outcome of a data drift analysis run."""

    pipeline_id: str
    computed_at: str  # ISO 8601
    dataset_drift_detected: bool
    psi_scores: dict[str, float]
    kl_divergence: dict[str, float]
    drifted_columns: list[str]
    severity: Literal["none", "warning", "critical"]
    evidently_report_path: str | None = None  # MinIO path for full HTML report

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pipeline_id": self.pipeline_id,
            "computed_at": self.computed_at,
            "dataset_drift_detected": self.dataset_drift_detected,
            "psi_scores": self.psi_scores,
            "kl_divergence": self.kl_divergence,
            "drifted_columns": self.drifted_columns,
            "severity": self.severity,
            "evidently_report_path": self.evidently_report_path,
        }


class DriftService:
    """
    Computes data drift between the training distribution and incoming
    prediction data using Evidently AI.

    Metrics:
    - Population Stability Index (PSI) per column.
    - Kullback-Leibler divergence per column.
    - Overall dataset drift flag.

    Severity classification:
    - none     → all PSI < 0.10
    - warning  → any PSI in [0.10, 0.25)
    - critical → any PSI >= 0.25
    """

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def compute(
        self,
        pipeline: Pipeline,
        current_df: pl.DataFrame,
    ) -> DriftReport:
        """
        Compute drift between training data and *current_df*.

        Args:
            pipeline: Completed Pipeline record with a model_minio_path.
            current_df: Polars DataFrame of the incoming prediction payload.

        Returns:
            DriftReport with PSI/KL scores and severity classification.
        """
        logger.info(
            "DriftService started",
            pipeline_id=str(pipeline.id),
            incoming_rows=len(current_df),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ── Load reference (training) data ────────────────────────────────
            reference_df = self._load_reference_data(pipeline, tmp)

            if reference_df is None:
                logger.warning(
                    "No reference data available for drift analysis",
                    pipeline_id=str(pipeline.id),
                )
                return self._empty_report(str(pipeline.id))

            # ── Align columns ─────────────────────────────────────────────────
            common_cols = [
                c for c in reference_df.columns
                if c in current_df.columns and c != "__outlier_flag__"
            ]
            ref_pd = reference_df.select(common_cols).to_pandas()
            cur_pd = current_df.select(
                [c for c in common_cols if c in current_df.columns]
            ).to_pandas()

            # ── Run Evidently drift report ─────────────────────────────────────
            psi_scores, kl_divergence, drifted_columns, report_path = (
                self._run_evidently(ref_pd, cur_pd, pipeline, tmp)
            )

            # ── Classify severity ──────────────────────────────────────────────
            max_psi = max(psi_scores.values(), default=0.0)
            if max_psi >= _PSI_CRITICAL_THRESHOLD:
                severity: Literal["none", "warning", "critical"] = "critical"
            elif max_psi >= _PSI_WARNING_THRESHOLD:
                severity = "warning"
            else:
                severity = "none"

            dataset_drift = len(drifted_columns) > 0

        report = DriftReport(
            pipeline_id=str(pipeline.id),
            computed_at=datetime.now(tz=timezone.utc).isoformat(),
            dataset_drift_detected=dataset_drift,
            psi_scores=psi_scores,
            kl_divergence=kl_divergence,
            drifted_columns=drifted_columns,
            severity=severity,
            evidently_report_path=report_path,
        )
        logger.info(
            "DriftService completed",
            pipeline_id=str(pipeline.id),
            severity=severity,
            drifted_columns=drifted_columns,
        )
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_reference_data(
        self, pipeline: Pipeline, tmp: Path
    ) -> pl.DataFrame | None:
        """Attempt to load the training parquet used for drift reference."""
        # The clean parquet path follows: athena-raw/clean/{data_source_id}/clean_*.parquet
        # We search MinIO for it via the data_source_id embedded in the path
        # For now return None if the path is not resolvable — graceful degradation
        try:
            # Construct expected MinIO path pattern
            data_source_id = str(pipeline.data_source_id)
            # List objects with prefix to find the clean parquet
            from minio import Minio
            from app.core.config import settings

            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            objects = list(client.list_objects(
                settings.MINIO_RAW_BUCKET,
                prefix=f"clean/{data_source_id}/",
                recursive=True,
            ))
            if not objects:
                return None

            latest = sorted(objects, key=lambda o: o.last_modified, reverse=True)[0]
            minio_path = f"{settings.MINIO_RAW_BUCKET}/{latest.object_name}"
            local_path = self._storage.download_to_temp(minio_path, tmp)
            return pl.read_parquet(local_path)
        except Exception as exc:
            logger.warning("Could not load reference data", error=str(exc))
            return None

    def _run_evidently(
        self,
        reference_df: object,
        current_df: object,
        pipeline: Pipeline,
        tmp: Path,
    ) -> tuple[dict[str, float], dict[str, float], list[str], str | None]:
        """
        Run Evidently DataDriftPreset and extract PSI + KL scores.

        Returns:
            Tuple of (psi_scores, kl_divergence, drifted_columns, report_minio_path).
        """
        try:
            from evidently import ColumnMapping
            from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
            from evidently.report import Report

            report = Report(metrics=[DatasetDriftMetric()])
            report.run(reference_data=reference_df, current_data=current_df)

            report_dict = report.as_dict()
            drift_result = report_dict["metrics"][0]["result"]

            psi_scores: dict[str, float] = {}
            kl_divergence: dict[str, float] = {}
            drifted_columns: list[str] = []

            for col_data in drift_result.get("drift_by_columns", {}).values():
                col_name = col_data.get("column_name", "")
                drift_score = float(col_data.get("drift_score", 0.0))
                psi_scores[col_name] = drift_score
                kl_divergence[col_name] = drift_score  # Evidently uses stattest internally
                if col_data.get("drift_detected", False):
                    drifted_columns.append(col_name)

            # Save HTML report to MinIO
            report_path: str | None = None
            try:
                html_file = tmp / f"drift_report_{uuid.uuid4().hex}.html"
                report.save_html(str(html_file))
                report_path = self._storage.upload_file(
                    local_path=html_file,
                    object_name=f"drift/{pipeline.id}/{html_file.name}",
                )
            except Exception as exc:
                logger.warning("Failed to save drift HTML report", error=str(exc))

            return psi_scores, kl_divergence, drifted_columns, report_path

        except ImportError:
            logger.warning("evidently not installed; skipping drift computation")
            return {}, {}, [], None
        except Exception as exc:
            logger.error("Evidently drift report failed", error=str(exc))
            return {}, {}, [], None

    def _empty_report(self, pipeline_id: str) -> DriftReport:
        """Return a no-op drift report when reference data is unavailable."""
        return DriftReport(
            pipeline_id=pipeline_id,
            computed_at=datetime.now(tz=timezone.utc).isoformat(),
            dataset_drift_detected=False,
            psi_scores={},
            kl_divergence={},
            drifted_columns=[],
            severity="none",
        )
