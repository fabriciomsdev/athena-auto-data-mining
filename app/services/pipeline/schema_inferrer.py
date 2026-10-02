"""Column type classification for incoming datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import polars as pl
import structlog

logger = structlog.get_logger()


class ColumnType(StrEnum):
    """Strict taxonomy of column data types used across the pipeline."""

    NUMERIC = "numeric"
    CATEGORICAL_LOW = "categorical_low"    # cardinality < HIGH_CARDINALITY_THRESHOLD
    CATEGORICAL_HIGH = "categorical_high"  # cardinality >= HIGH_CARDINALITY_THRESHOLD
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"  # high-cardinality free text, drop candidate
    UNKNOWN = "unknown"


@dataclass
class ColumnSchema:
    """Schema metadata inferred for a single DataFrame column."""

    name: str
    dtype: ColumnType
    null_pct: float
    cardinality: int
    is_target: bool
    sample_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "dtype": str(self.dtype),
            "null_pct": round(self.null_pct, 4),
            "cardinality": self.cardinality,
            "is_target": self.is_target,
            "sample_values": self.sample_values,
        }


class SchemaInferrer:
    """
    Scan a Polars DataFrame and classify every column into a ColumnType.

    Classification rules (evaluated in order):
    1. BOOLEAN  — dtype is Boolean, or cardinality <= 2 with values in {0,1,True,False}
    2. DATETIME — dtype is Date/Datetime, or column name contains date/time keywords
    3. NUMERIC  — Polars numeric dtypes (Int*, UInt*, Float*)
    4. CATEGORICAL_LOW  — String/Categorical with cardinality < HIGH_CARDINALITY_THRESHOLD
    5. CATEGORICAL_HIGH — String/Categorical with HIGH_CARDINALITY_THRESHOLD <= cardinality
                          < TEXT_CARDINALITY_THRESHOLD
    6. TEXT     — String with cardinality >= TEXT_CARDINALITY_THRESHOLD (free text)
    7. UNKNOWN  — anything else
    """

    HIGH_CARDINALITY_THRESHOLD: ClassVar[int] = 50
    TEXT_CARDINALITY_THRESHOLD: ClassVar[int] = 500
    SAMPLE_SIZE: ClassVar[int] = 5
    DATETIME_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {"date", "time", "timestamp", "dt", "datetime", "created", "updated", "at"}
    )

    def infer(self, df: pl.DataFrame, target_column: str) -> list[ColumnSchema]:
        """
        Infer a ColumnSchema for every column in *df*.

        Args:
            df: The source Polars DataFrame.
            target_column: Name of the prediction target column.

        Returns:
            A list of ColumnSchema, one per column in df.
        """
        schemas: list[ColumnSchema] = []
        n_rows = len(df)

        for col_name in df.columns:
            series = df[col_name]
            null_count = series.null_count()
            null_pct = null_count / n_rows if n_rows > 0 else 0.0
            non_null = series.drop_nulls()
            cardinality = non_null.n_unique()
            sample_values = [str(v) for v in non_null.head(self.SAMPLE_SIZE).to_list()]
            dtype = self._classify(series, col_name, cardinality)
            is_target = col_name == target_column

            schemas.append(
                ColumnSchema(
                    name=col_name,
                    dtype=dtype,
                    null_pct=null_pct,
                    cardinality=cardinality,
                    is_target=is_target,
                    sample_values=sample_values,
                )
            )
            logger.debug(
                "Column inferred",
                column=col_name,
                dtype=dtype,
                cardinality=cardinality,
                null_pct=round(null_pct, 3),
            )

        logger.info(
            "Schema inference complete",
            total_columns=len(schemas),
            target=target_column,
        )
        return schemas

    def _classify(
        self, series: pl.Series, col_name: str, cardinality: int
    ) -> ColumnType:
        """Apply classification rules and return the appropriate ColumnType."""
        polars_dtype = series.dtype

        # 1. BOOLEAN
        if polars_dtype == pl.Boolean:
            return ColumnType.BOOLEAN
        if cardinality <= 2:
            non_null_vals = set(str(v).lower() for v in series.drop_nulls().to_list())
            bool_sets: list[set[str]] = [
                {"0", "1"},
                {"true", "false"},
                {"yes", "no"},
                {"y", "n"},
            ]
            if any(non_null_vals.issubset(s) for s in bool_sets):
                return ColumnType.BOOLEAN

        # 2. DATETIME
        if polars_dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
            return ColumnType.DATETIME
        lower_name = col_name.lower()
        if any(kw in lower_name for kw in self.DATETIME_KEYWORDS):
            return ColumnType.DATETIME

        # 3. NUMERIC
        numeric_types = (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        )
        if polars_dtype in numeric_types:
            return ColumnType.NUMERIC

        # 4-6. STRING / CATEGORICAL
        if polars_dtype in (pl.String, pl.Utf8, pl.Categorical):
            if cardinality < self.HIGH_CARDINALITY_THRESHOLD:
                return ColumnType.CATEGORICAL_LOW
            if cardinality < self.TEXT_CARDINALITY_THRESHOLD:
                return ColumnType.CATEGORICAL_HIGH
            return ColumnType.TEXT

        return ColumnType.UNKNOWN
