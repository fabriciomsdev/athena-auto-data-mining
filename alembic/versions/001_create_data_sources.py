"""001 create data_sources table

Revision ID: 001
Revises:
Create Date: 2026-10-02
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "data_sources",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("minio_path", sa.String(1024), nullable=True),
        sa.Column("external_uri", sa.Text(), nullable=True),
        sa.Column("schema_json", sa.Text(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_data_sources_source_type", "data_sources", ["source_type"])


def downgrade() -> None:
    op.drop_index("ix_data_sources_source_type", table_name="data_sources")
    op.drop_table("data_sources")
