"""002 create pipelines table

Revision ID: 002
Revises: 001
Create Date: 2026-10-02
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "002"
down_revision: str = "001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "pipelines",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("problem_type", sa.String(50), nullable=False),
        sa.Column("target_column", sa.String(255), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("mlflow_run_id", sa.String(255), nullable=True),
        sa.Column("mlflow_experiment_id", sa.String(255), nullable=True),
        sa.Column("model_minio_path", sa.String(1024), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column(
            "data_source_id",
            UUID(as_uuid=True),
            sa.ForeignKey("data_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_pipelines_status", "pipelines", ["status"])
    op.create_index("ix_pipelines_data_source_id", "pipelines", ["data_source_id"])


def downgrade() -> None:
    op.drop_index("ix_pipelines_data_source_id", table_name="pipelines")
    op.drop_index("ix_pipelines_status", table_name="pipelines")
    op.drop_table("pipelines")
