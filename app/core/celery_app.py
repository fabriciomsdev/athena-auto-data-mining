from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "athena",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ingestion_tasks",
        "app.tasks.pipeline_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task per worker at a time (ML jobs are heavy)
    result_expires=3600 * 24,  # keep results for 24h
    task_routes={
        "app.tasks.ingestion_tasks.*": {"queue": "ingestion"},
        "app.tasks.pipeline_tasks.*": {"queue": "pipeline"},
    },
)
