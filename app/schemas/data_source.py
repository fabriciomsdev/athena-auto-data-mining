from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DataSourceCreate(BaseModel):
    name: str
    external_uri: str | None = None
    config: dict | None = None


class DataSourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    source_type: str
    minio_path: str | None
    external_uri: str | None
    row_count: int | None
    created_at: datetime


class PipelineCreate(BaseModel):
    name: str
    data_source_id: uuid.UUID
    problem_type: str  # "classification" | "forecasting"
    target_column: str
    config: dict | None = None


class PipelineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    problem_type: str
    target_column: str
    status: str
    celery_task_id: str | None
    mlflow_run_id: str | None
    model_minio_path: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
