"""Repository pattern for Pipeline ORM operations."""
from __future__ import annotations

import uuid
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pipeline import Pipeline, PipelineStatus


class PipelineRepository:
    """Encapsulates all DB operations for Pipeline entities."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, pipeline_id: uuid.UUID) -> Pipeline | None:
        """Return a Pipeline by primary key, or None if not found."""
        return await self._session.get(Pipeline, pipeline_id)

    async def get_by_id_or_raise(self, pipeline_id: uuid.UUID) -> Pipeline:
        """Return a Pipeline by primary key, raising ValueError if not found."""
        pipeline = await self.get_by_id(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline {pipeline_id} not found.")
        return pipeline

    async def get_by_celery_task_id(self, task_id: str) -> Pipeline | None:
        """Look up a Pipeline by its associated Celery task ID."""
        result = await self._session.execute(
            select(Pipeline).where(Pipeline.celery_task_id == task_id)
        )
        return result.scalar_one_or_none()

    async def list_by_status(self, status: PipelineStatus) -> Sequence[Pipeline]:
        """Return all pipelines in a given status."""
        result = await self._session.execute(
            select(Pipeline).where(Pipeline.status == status)
        )
        return result.scalars().all()

    async def create(self, pipeline: Pipeline) -> Pipeline:
        """Persist a new Pipeline and return the refreshed instance."""
        self._session.add(pipeline)
        await self._session.flush()
        await self._session.refresh(pipeline)
        return pipeline

    async def update_status(
        self,
        pipeline_id: uuid.UUID,
        status: PipelineStatus,
        error_message: str | None = None,
    ) -> Pipeline:
        """Transition a pipeline's status, optionally recording an error."""
        pipeline = await self.get_by_id_or_raise(pipeline_id)
        pipeline.status = status
        if error_message is not None:
            pipeline.error_message = error_message
        await self._session.flush()
        return pipeline

    async def update_training_result(
        self,
        pipeline_id: uuid.UUID,
        mlflow_run_id: str,
        mlflow_experiment_id: str,
        model_minio_path: str,
        metrics_json: str,
    ) -> Pipeline:
        """Persist training outcomes once training_session completes."""
        pipeline = await self.get_by_id_or_raise(pipeline_id)
        pipeline.mlflow_run_id = mlflow_run_id
        pipeline.mlflow_experiment_id = mlflow_experiment_id
        pipeline.model_minio_path = model_minio_path
        pipeline.metrics_json = metrics_json
        pipeline.status = PipelineStatus.COMPLETED
        await self._session.flush()
        return pipeline
