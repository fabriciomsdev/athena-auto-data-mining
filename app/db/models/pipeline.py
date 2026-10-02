from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class DataSourceType(StrEnum):
    CSV = "csv"
    EXCEL = "excel"
    GOOGLE_SHEETS = "google_sheets"
    BIGQUERY = "bigquery"


class PipelineStatus(StrEnum):
    PENDING = "pending"
    INGESTING = "ingesting"
    DATA_PREP = "data_prep"
    FEATURE_ENGINEERING = "feature_engineering"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class ProblemType(StrEnum):
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"


# ── DataSource ────────────────────────────────────────────────────────────────

class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[DataSourceType] = mapped_column(String(50), nullable=False)
    minio_path: Mapped[str | None] = mapped_column(String(1024))
    external_uri: Mapped[str | None] = mapped_column(Text)
    schema_json: Mapped[str | None] = mapped_column(Text)  # inferred schema as JSON
    row_count: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    pipelines: Mapped[list[Pipeline]] = relationship(back_populates="data_source")


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline(Base):
    __tablename__ = "pipelines"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    problem_type: Mapped[ProblemType] = mapped_column(String(50), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[PipelineStatus] = mapped_column(
        String(50), default=PipelineStatus.PENDING
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(255))
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(255))
    model_minio_path: Mapped[str | None] = mapped_column(String(1024))
    error_message: Mapped[str | None] = mapped_column(Text)
    config_json: Mapped[str | None] = mapped_column(Text)  # user config overrides
    metrics_json: Mapped[str | None] = mapped_column(Text)  # final metrics snapshot

    data_source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("data_sources.id"), nullable=False
    )
    data_source: Mapped[DataSource] = relationship(back_populates="pipelines")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
