"""Data preparation service — cleans raw data and persists processed parquet to MinIO."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import structlog
from sklearn.ensemble import IsolationForest

from app.db.models.pipeline import DataSource, DataSourceType
from app.db.repositories.data_source_repository import DataSourceRepository
from app.services.ingestion.connectors import ConnectorFactory
from app.services.ingestion.storage import StorageService
from app.services.pipeline.schema_inferrer import ColumnSchema, ColumnType, SchemaInferrer

logger = structlog.get_logger()

# Polars numeric dtype group for convenience
_NUMERIC_DTYPES: tuple[type, ...] = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


@dataclass
class DataPrepConfig:
    """User-overridable configuration for the data preparation step."""

    numeric_impute_strategy: Literal["mean", "median", "drop"] = "median"
    categorical_impute_value: str = "missing"
    outlier_detection: bool = True
    outlier_contamination: float = 0.05  # expected proportion of outliers
    drop_text_columns: bool = True        # drop ColumnType.TEXT columns automatically


@dataclass
class DataPrepResult:
    """Outcome of the data preparation step."""

    clean_parquet_path: str          # MinIO object path
    schema: list[ColumnSchema]
    row_count: int
    outlier_count: int
    null_report: dict[str, float]    # column_name → null percentage before imputation
    dropped_columns: list[str]       # TEXT columns removed


class DataPrepService:
    """
    Prepares raw ingested data for feature engineering.

    Responsibilities (in order):
    1. Download raw file from MinIO via StorageService.
    2. Parse into a Polars DataFrame using the appropriate connector.
    3. Run SchemaInferrer to classify every column.
    4. Drop free-text columns (ColumnType.TEXT) if configured.
    5. Impute nulls:
       - numeric  → mean / median / row-drop (configurable)
       - category → fill with ``config.categorical_impute_value``
       - boolean  → fill with mode
       - datetime → forward-fill
    6. Detect outliers on numeric columns via IsolationForest.
       Adds a boolean column ``__outlier_flag__`` to the DataFrame.
    7. Validate that target_column exists post-cleaning.
    8. Save cleaned DataFrame as a `.parquet` file back to MinIO.
    9. Update DataSource.schema_json and DataSource.row_count in DB.
    10. Return a DataPrepResult.
    """

    _OUTLIER_FLAG_COLUMN: str = "__outlier_flag__"

    def __init__(
        self,
        storage: StorageService,
        ds_repo: DataSourceRepository,
        config: DataPrepConfig | None = None,
    ) -> None:
        self._storage = storage
        self._ds_repo = ds_repo
        self._config = config or DataPrepConfig()
        self._inferrer = SchemaInferrer()

    async def run(
        self,
        data_source: DataSource,
        target_column: str,
    ) -> DataPrepResult:
        """
        Execute the full data preparation workflow.

        Args:
            data_source: The DataSource ORM record containing the raw file location.
            target_column: Name of the prediction target column.

        Returns:
            DataPrepResult with the MinIO path of the cleaned parquet and quality stats.

        Raises:
            ValueError: If the target column is absent after cleaning.
        """
        logger.info(
            "DataPrepService started",
            data_source_id=str(data_source.id),
            target_column=target_column,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # ── 1. Download raw file from MinIO ───────────────────────────────
            raw_path = self._download_raw(data_source, Path(tmp_dir))

            # ── 2. Parse into Polars DataFrame ────────────────────────────────
            df = await self._parse(data_source, raw_path)
            logger.info("Raw data loaded", rows=len(df), columns=len(df.columns))

            # ── 3. Infer schema ───────────────────────────────────────────────
            schema = self._inferrer.infer(df, target_column)

            # ── 4. Build null report before any imputation ────────────────────
            null_report = {s.name: s.null_pct for s in schema}

            # ── 5. Drop TEXT columns ──────────────────────────────────────────
            dropped_columns = self._drop_text_columns(df, schema)
            if dropped_columns:
                df = df.drop(dropped_columns)
                schema = [s for s in schema if s.name not in dropped_columns]

            # ── 6. Impute nulls ───────────────────────────────────────────────
            df = self._impute(df, schema)

            # ── 7. Detect outliers ────────────────────────────────────────────
            outlier_count = 0
            if self._config.outlier_detection:
                df, outlier_count = self._flag_outliers(df, schema)

            # ── 8. Validate target column ─────────────────────────────────────
            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found after cleaning. "
                    f"Available columns: {df.columns}"
                )

            # ── 9. Save cleaned parquet to MinIO ──────────────────────────────
            parquet_filename = f"clean_{uuid.uuid4().hex}.parquet"
            local_parquet = Path(tmp_dir) / parquet_filename
            df.write_parquet(local_parquet)

            object_name = f"clean/{data_source.id}/{parquet_filename}"
            clean_path = self._storage.upload_file(
                local_path=local_parquet,
                object_name=object_name,
            )

            # ── 10. Update DataSource in DB ────────────────────────────────────
            schema_json = json.dumps([s.to_dict() for s in schema])
            await self._ds_repo.update_schema(
                data_source_id=data_source.id,
                schema_json=schema_json,
                row_count=len(df),
            )

        result = DataPrepResult(
            clean_parquet_path=clean_path,
            schema=schema,
            row_count=len(df),
            outlier_count=outlier_count,
            null_report=null_report,
            dropped_columns=dropped_columns,
        )
        logger.info(
            "DataPrepService completed",
            rows=result.row_count,
            outliers=result.outlier_count,
            dropped=result.dropped_columns,
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _download_raw(self, data_source: DataSource, dest_dir: Path) -> Path:
        """Download the raw file from MinIO if it is a file-backed source."""
        if data_source.minio_path:
            return self._storage.download_to_temp(data_source.minio_path, dest_dir)
        raise ValueError(
            f"DataSource {data_source.id} has no minio_path. "
            "External sources (Sheets/BigQuery) must be materialised first."
        )

    async def _parse(self, data_source: DataSource, raw_path: Path) -> pl.DataFrame:
        """Load the raw file into a Polars DataFrame using the ConnectorFactory."""
        connector_kwargs: dict[str, object] = {}

        if data_source.source_type == DataSourceType.CSV:
            connector_kwargs = {"file_path": raw_path}
        elif data_source.source_type == DataSourceType.EXCEL:
            connector_kwargs = {"file_path": raw_path}
        else:
            raise ValueError(
                f"Source type '{data_source.source_type}' cannot be parsed from a local file."
            )

        connector = ConnectorFactory.from_source_type(
            data_source.source_type, **connector_kwargs
        )
        return await connector.load()

    def _drop_text_columns(
        self, df: pl.DataFrame, schema: list[ColumnSchema]
    ) -> list[str]:
        """Return a list of TEXT column names to drop (targets are never dropped)."""
        if not self._config.drop_text_columns:
            return []
        dropped = [
            s.name
            for s in schema
            if s.dtype == ColumnType.TEXT and not s.is_target
        ]
        if dropped:
            logger.warning("Dropping free-text columns", columns=dropped)
        return dropped

    def _impute(self, df: pl.DataFrame, schema: list[ColumnSchema]) -> pl.DataFrame:
        """Apply null-filling strategies per column type."""
        expressions: list[pl.Expr] = []

        for col_schema in schema:
            col = col_schema.name
            series = df[col]

            if col_schema.dtype == ColumnType.NUMERIC:
                if self._config.numeric_impute_strategy == "median":
                    fill_val = series.drop_nulls().median()
                elif self._config.numeric_impute_strategy == "mean":
                    fill_val = series.drop_nulls().mean()
                else:
                    # "drop" strategy — handled at DataFrame level below
                    continue
                expressions.append(pl.col(col).fill_null(fill_val))

            elif col_schema.dtype in (
                ColumnType.CATEGORICAL_LOW,
                ColumnType.CATEGORICAL_HIGH,
            ):
                expressions.append(
                    pl.col(col).fill_null(self._config.categorical_impute_value)
                )

            elif col_schema.dtype == ColumnType.BOOLEAN:
                mode_vals = series.drop_nulls().mode()
                mode_val = mode_vals[0] if len(mode_vals) > 0 else False
                expressions.append(pl.col(col).fill_null(mode_val))

            elif col_schema.dtype == ColumnType.DATETIME:
                expressions.append(pl.col(col).forward_fill())

        if expressions:
            df = df.with_columns(expressions)

        # Handle "drop" strategy — remove rows where any numeric col is null
        if self._config.numeric_impute_strategy == "drop":
            numeric_cols = [
                s.name for s in schema if s.dtype == ColumnType.NUMERIC
            ]
            if numeric_cols:
                df = df.drop_nulls(subset=numeric_cols)

        return df

    def _flag_outliers(
        self, df: pl.DataFrame, schema: list[ColumnSchema]
    ) -> tuple[pl.DataFrame, int]:
        """
        Run IsolationForest on numeric columns and add __outlier_flag__ column.

        Returns:
            Tuple of (modified DataFrame, count of detected outliers).
        """
        numeric_cols = [
            s.name
            for s in schema
            if s.dtype == ColumnType.NUMERIC and not s.is_target
        ]

        if not numeric_cols:
            logger.debug("No numeric columns available for outlier detection")
            df = df.with_columns(pl.lit(False).alias(self._OUTLIER_FLAG_COLUMN))
            return df, 0

        X = df.select(numeric_cols).to_numpy()
        clf = IsolationForest(
            contamination=self._config.outlier_contamination,
            random_state=42,
            n_jobs=-1,
        )
        predictions = clf.fit_predict(X)  # 1 = inlier, -1 = outlier
        outlier_flags = (predictions == -1).tolist()
        outlier_count = sum(outlier_flags)

        df = df.with_columns(
            pl.Series(name=self._OUTLIER_FLAG_COLUMN, values=outlier_flags)
        )
        logger.info(
            "Outlier detection complete",
            total=len(df),
            outliers=outlier_count,
            pct=round(outlier_count / len(df) * 100, 2),
        )
        return df, outlier_count
