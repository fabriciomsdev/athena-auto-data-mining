"""Unit tests for SchemaInferrer — all happy and edge-case paths."""
from __future__ import annotations

import polars as pl
import pytest

from app.services.pipeline.schema_inferrer import ColumnType, SchemaInferrer


@pytest.fixture
def inferrer() -> SchemaInferrer:
    return SchemaInferrer()


class TestSchemaInferrerNumeric:
    def test_int_column_classified_as_numeric(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"age": [25, 30, 35], "target": [0, 1, 0]})
        schemas = inferrer.infer(df, "target")
        age = next(s for s in schemas if s.name == "age")
        assert age.dtype == ColumnType.NUMERIC

    def test_float_column_classified_as_numeric(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"salary": [1000.5, 2000.0, 3000.75], "target": [0, 1, 0]})
        schemas = inferrer.infer(df, "target")
        salary = next(s for s in schemas if s.name == "salary")
        assert salary.dtype == ColumnType.NUMERIC


class TestSchemaInferrerBoolean:
    def test_bool_dtype_classified_as_boolean(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"active": [True, False, True], "target": [0, 1, 0]})
        schemas = inferrer.infer(df, "target")
        active = next(s for s in schemas if s.name == "active")
        assert active.dtype == ColumnType.BOOLEAN

    def test_binary_int_column_classified_as_boolean(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"flag": [0, 1, 0, 1], "target": [0, 1, 0, 1]})
        schemas = inferrer.infer(df, "target")
        flag = next(s for s in schemas if s.name == "flag")
        assert flag.dtype == ColumnType.BOOLEAN

    def test_yes_no_column_classified_as_boolean(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"subscribed": ["yes", "no", "yes"], "target": ["a", "b", "a"]})
        schemas = inferrer.infer(df, "target")
        subscribed = next(s for s in schemas if s.name == "subscribed")
        assert subscribed.dtype == ColumnType.BOOLEAN


class TestSchemaInferrerDatetime:
    def test_date_dtype_classified_as_datetime(self, inferrer: SchemaInferrer) -> None:
        from datetime import date
        df = pl.DataFrame({
            "signup_date": [date(2024, 1, 1), date(2024, 6, 15)],
            "target": [0, 1],
        })
        schemas = inferrer.infer(df, "target")
        dt = next(s for s in schemas if s.name == "signup_date")
        assert dt.dtype == ColumnType.DATETIME

    def test_column_named_with_date_keyword(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({
            "created_at": ["2024-01-01", "2024-06-15"],
            "target": ["a", "b"],
        })
        schemas = inferrer.infer(df, "target")
        col = next(s for s in schemas if s.name == "created_at")
        assert col.dtype == ColumnType.DATETIME


class TestSchemaInferrerCategorical:
    def test_low_cardinality_string_classified_as_categorical_low(
        self, inferrer: SchemaInferrer
    ) -> None:
        df = pl.DataFrame({
            "city": ["NY", "LA", "NY", "LA", "SF"] * 5,
            "target": [0] * 25,
        })
        schemas = inferrer.infer(df, "target")
        city = next(s for s in schemas if s.name == "city")
        assert city.dtype == ColumnType.CATEGORICAL_LOW

    def test_high_cardinality_string_classified_as_categorical_high(
        self, inferrer: SchemaInferrer
    ) -> None:
        unique_vals = [f"prod_{i}" for i in range(100)]
        df = pl.DataFrame({"product_id": unique_vals, "target": [0] * 100})
        schemas = inferrer.infer(df, "target")
        prod = next(s for s in schemas if s.name == "product_id")
        assert prod.dtype == ColumnType.CATEGORICAL_HIGH

    def test_very_high_cardinality_classified_as_text(
        self, inferrer: SchemaInferrer
    ) -> None:
        unique_vals = [f"review text number {i} " * 5 for i in range(600)]
        df = pl.DataFrame({"comment": unique_vals, "target": [0] * 600})
        schemas = inferrer.infer(df, "target")
        comment = next(s for s in schemas if s.name == "comment")
        assert comment.dtype == ColumnType.TEXT


class TestSchemaInferrerTarget:
    def test_target_column_marked_as_is_target(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"age": [25, 30], "churn": [0, 1]})
        schemas = inferrer.infer(df, "churn")
        churn = next(s for s in schemas if s.name == "churn")
        assert churn.is_target is True

    def test_non_target_columns_not_marked(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"age": [25, 30], "churn": [0, 1]})
        schemas = inferrer.infer(df, "churn")
        age = next(s for s in schemas if s.name == "age")
        assert age.is_target is False


class TestSchemaInferrerNullPct:
    def test_null_pct_computed_correctly(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"val": [1, None, 3, None], "target": [0, 1, 0, 1]})
        schemas = inferrer.infer(df, "target")
        val = next(s for s in schemas if s.name == "val")
        assert abs(val.null_pct - 0.5) < 1e-6

    def test_zero_null_pct_for_complete_column(self, inferrer: SchemaInferrer) -> None:
        df = pl.DataFrame({"val": [1, 2, 3], "target": [0, 1, 0]})
        schemas = inferrer.infer(df, "target")
        val = next(s for s in schemas if s.name == "val")
        assert val.null_pct == 0.0
