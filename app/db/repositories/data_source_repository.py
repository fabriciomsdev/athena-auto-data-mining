"""Repository pattern for DataSource ORM operations."""
from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pipeline import DataSource


class DataSourceRepository:
    """Encapsulates all DB operations for DataSource entities."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, data_source_id: uuid.UUID) -> DataSource | None:
        """Return a DataSource by primary key, or None if not found."""
        return await self._session.get(DataSource, data_source_id)

    async def get_by_id_or_raise(self, data_source_id: uuid.UUID) -> DataSource:
        """Return a DataSource by primary key, raising ValueError if not found."""
        ds = await self.get_by_id(data_source_id)
        if ds is None:
            raise ValueError(f"DataSource {data_source_id} not found.")
        return ds

    async def list_all(self) -> Sequence[DataSource]:
        """Return all DataSource records."""
        result = await self._session.execute(select(DataSource))
        return result.scalars().all()

    async def create(self, data_source: DataSource) -> DataSource:
        """Persist a new DataSource and return the refreshed instance."""
        self._session.add(data_source)
        await self._session.flush()
        await self._session.refresh(data_source)
        return data_source

    async def update_schema(
        self,
        data_source_id: uuid.UUID,
        schema_json: str,
        row_count: int,
    ) -> DataSource:
        """Update schema_json and row_count after data_prep completes."""
        ds = await self.get_by_id_or_raise(data_source_id)
        ds.schema_json = schema_json
        ds.row_count = row_count
        await self._session.flush()
        return ds
