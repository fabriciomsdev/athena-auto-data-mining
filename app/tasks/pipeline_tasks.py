from __future__ import annotations

import structlog

from app.core.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(
    name="app.tasks.pipeline_tasks.run_pipeline",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def run_pipeline(
    self,  # noqa: ANN001
    pipeline_id: str,
    data_source_id: str,
    problem_type: str,
    target_column: str,
    config: dict,
) -> dict:
    """
    Orchestrates the full AutoML pipeline:
      1. data_prep
      2. feature_engineering
      3. training_session
      4. model persistence
    """
    from app.services.pipeline.orchestrator import PipelineOrchestrator

    logger.info(
        "Pipeline task started",
        pipeline_id=pipeline_id,
        problem_type=problem_type,
        target=target_column,
    )

    try:
        orchestrator = PipelineOrchestrator(
            pipeline_id=pipeline_id,
            data_source_id=data_source_id,
            problem_type=problem_type,
            target_column=target_column,
            config=config,
        )
        result = orchestrator.run()
        logger.info("Pipeline task completed", pipeline_id=pipeline_id, result=result)
        return result

    except Exception as exc:
        logger.error("Pipeline task failed", pipeline_id=pipeline_id, error=str(exc))
        raise self.retry(exc=exc) from exc
