"""Pipeline orchestrator — coordinates all phases sequentially for a single run."""
from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pipeline import Pipeline, PipelineStatus
from app.db.repositories.data_source_repository import DataSourceRepository
from app.db.repositories.pipeline_repository import PipelineRepository
from app.db.session import AsyncSessionLocal
from app.services.ingestion.storage import StorageService
from app.services.pipeline.data_prep import DataPrepConfig, DataPrepService
from app.services.pipeline.feature_engineering import (
    FeatureEngineeringConfig,
    FeatureEngineeringService,
)
from app.services.pipeline.training_session import TrainingConfig, TrainingSessionService
from app.services.registry.mlflow_registry import MLflowRegistry

logger = structlog.get_logger()


class PipelineOrchestrator:
    """
    Central coordinator called by the Celery pipeline task.

    Sequentially executes:
      Phase 2 → DataPrepService
      Phase 3 → FeatureEngineeringService
      Phase 4 → TrainingSessionService

    Updates Pipeline.status between each phase and captures errors.
    All DB access is done via a dedicated sync session opened per-call
    (Celery workers run in a non-async context).
    """

    def __init__(
        self,
        pipeline_id: str,
        data_source_id: str,
        problem_type: str,
        target_column: str,
        config: dict[str, Any],
    ) -> None:
        self._pipeline_id = uuid.UUID(pipeline_id)
        self._data_source_id = uuid.UUID(data_source_id)
        self._problem_type = problem_type
        self._target_column = target_column
        self._config = config
        self._storage = StorageService()
        self._mlflow = MLflowRegistry()

    def run(self) -> dict[str, Any]:
        """
        Synchronous entry point called from the Celery task.

        Opens its own asyncio event loop to run the async orchestration.

        Returns:
            A dict summary of the completed run for Celery result storage.
        """
        import asyncio

        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict[str, Any]:
        """Async orchestration of all pipeline phases."""
        async with AsyncSessionLocal() as session:
            pipeline_repo = PipelineRepository(session)
            ds_repo = DataSourceRepository(session)

            pipeline = await pipeline_repo.get_by_id_or_raise(self._pipeline_id)
            data_source = await ds_repo.get_by_id_or_raise(self._data_source_id)

            try:
                # ── Phase 2: data_prep ─────────────────────────────────────────
                await pipeline_repo.update_status(
                    self._pipeline_id, PipelineStatus.DATA_PREP
                )
                await session.commit()

                data_prep_config = DataPrepConfig(
                    **{k: v for k, v in self._config.items()
                       if k in DataPrepConfig.__dataclass_fields__}
                )
                data_prep_svc = DataPrepService(
                    storage=self._storage,
                    ds_repo=ds_repo,
                    config=data_prep_config,
                )
                prep_result = await data_prep_svc.run(data_source, self._target_column)
                await session.commit()

                logger.info(
                    "Phase 2 complete",
                    pipeline_id=str(self._pipeline_id),
                    rows=prep_result.row_count,
                )

                # ── Phase 3: feature_engineering ──────────────────────────────
                await pipeline_repo.update_status(
                    self._pipeline_id, PipelineStatus.FEATURE_ENGINEERING
                )
                await session.commit()

                fe_config = FeatureEngineeringConfig(
                    **{k: v for k, v in self._config.items()
                       if k in FeatureEngineeringConfig.__dataclass_fields__}
                )
                fe_svc = FeatureEngineeringService(
                    storage=self._storage,
                    config=fe_config,
                )
                fe_result = await fe_svc.run(
                    clean_parquet_path=prep_result.clean_parquet_path,
                    schema=prep_result.schema,
                    target_column=self._target_column,
                    problem_type=pipeline.problem_type,
                )
                await session.commit()

                logger.info(
                    "Phase 3 complete",
                    pipeline_id=str(self._pipeline_id),
                    features=len(fe_result.feature_names_out),
                    leakage=fe_result.leakage_alerts,
                )

                # ── Phase 4: training_session ─────────────────────────────────
                await pipeline_repo.update_status(
                    self._pipeline_id, PipelineStatus.TRAINING
                )
                await session.commit()

                training_config = TrainingConfig(
                    **{k: v for k, v in self._config.items()
                       if k in TrainingConfig.__dataclass_fields__}
                )
                training_svc = TrainingSessionService(
                    storage=self._storage,
                    pipeline_repo=pipeline_repo,
                    mlflow_registry=self._mlflow,
                    config=training_config,
                )
                training_result = await training_svc.run(
                    pipeline=pipeline,
                    processed_parquet_path=fe_result.processed_parquet_path,
                    pipeline_artifact_path=fe_result.pipeline_artifact_path,
                    feature_names=fe_result.feature_names_out,
                )
                await session.commit()

                logger.info(
                    "Phase 4 complete",
                    pipeline_id=str(self._pipeline_id),
                    winner=training_result.winning_model,
                    metrics=training_result.metrics,
                )

                return {
                    "pipeline_id": str(self._pipeline_id),
                    "status": PipelineStatus.COMPLETED,
                    "winning_model": training_result.winning_model,
                    "metrics": training_result.metrics,
                    "model_minio_path": training_result.model_minio_path,
                    "mlflow_run_id": training_result.mlflow_run_id,
                    "leakage_alerts": fe_result.leakage_alerts,
                    "correlation_matrix": fe_result.correlation_matrix,
                    "feature_importance": training_result.feature_importance,
                }

            except Exception as exc:
                await session.rollback()
                error_msg = str(exc)
                logger.error(
                    "Pipeline failed",
                    pipeline_id=str(self._pipeline_id),
                    error=error_msg,
                )
                async with AsyncSessionLocal() as err_session:
                    err_repo = PipelineRepository(err_session)
                    await err_repo.update_status(
                        self._pipeline_id,
                        PipelineStatus.FAILED,
                        error_message=error_msg,
                    )
                    await err_session.commit()
                raise
