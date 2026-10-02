from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pipeline import Pipeline, PipelineStatus
from app.db.session import get_db
from app.schemas.data_source import PipelineCreate, PipelineOut
from app.tasks.pipeline_tasks import run_pipeline

logger = structlog.get_logger()
router = APIRouter()


@router.post(
    "",
    response_model=PipelineOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create and enqueue a new AutoML pipeline",
)
async def create_pipeline(
    payload: PipelineCreate,
    db: AsyncSession = Depends(get_db),
) -> PipelineOut:
    import json

    pipeline = Pipeline(
        name=payload.name,
        problem_type=payload.problem_type,
        target_column=payload.target_column,
        data_source_id=payload.data_source_id,
        config_json=json.dumps(payload.config or {}),
        status=PipelineStatus.PENDING,
    )
    db.add(pipeline)
    await db.flush()
    await db.refresh(pipeline)

    # Enqueue the Celery task
    task = run_pipeline.apply_async(
        kwargs={
            "pipeline_id": str(pipeline.id),
            "data_source_id": str(payload.data_source_id),
            "problem_type": payload.problem_type,
            "target_column": payload.target_column,
            "config": payload.config or {},
        },
        queue="pipeline",
    )

    pipeline.celery_task_id = task.id
    await db.flush()

    logger.info("Pipeline enqueued", id=str(pipeline.id), task_id=task.id)
    return PipelineOut.model_validate(pipeline)


@router.get(
    "/{pipeline_id}",
    response_model=PipelineOut,
    summary="Get pipeline status and results",
)
async def get_pipeline(
    pipeline_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> PipelineOut:
    pipeline = await db.get(Pipeline, pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return PipelineOut.model_validate(pipeline)
