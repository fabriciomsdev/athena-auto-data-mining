"""Unit tests for CorrelationService — leakage detection and matrix computation."""
from __future__ import annotations

import polars as pl
import pytest

from app.services.pipeline.correlation import CorrelationService


@pytest.fixture
def service() -> CorrelationService:
    return CorrelationService()


class TestCorrelationMatrix:
    def test_perfect_self_correlation(self, service: CorrelationService) -> None:
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [1.0, 2.0, 3.0, 4.0]})
        matrix = service.compute(df, "target")
        assert abs(matrix["x"]["x"] - 1.0) < 1e-6

    def test_positive_correlation_detected(self, service: CorrelationService) -> None:
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        matrix = service.compute(df, "target")
        assert matrix["x"]["target"] > 0.9

    def test_negative_correlation_detected(self, service: CorrelationService) -> None:
        df = pl.DataFrame({
            "x": [5.0, 4.0, 3.0, 2.0, 1.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        matrix = service.compute(df, "target")
        assert matrix["x"]["target"] < -0.9

    def test_empty_matrix_when_no_numeric_columns(
        self, service: CorrelationService
    ) -> None:
        df = pl.DataFrame({"city": ["NY", "LA", "SF"], "target_str": ["a", "b", "c"]})
        matrix = service.compute(df, "target_str")
        assert matrix == {}

    def test_matrix_is_symmetric(self, service: CorrelationService) -> None:
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "target": [1.0, 0.0, 1.0, 0.0],
        })
        matrix = service.compute(df, "target")
        assert abs(matrix["a"]["b"] - matrix["b"]["a"]) < 1e-6


class TestTargetLeakageDetection:
    def test_leakage_detected_above_threshold(
        self, service: CorrelationService
    ) -> None:
        matrix = {
            "target": {"target": 1.0, "leaking_col": 0.97},
            "leaking_col": {"leaking_col": 1.0, "target": 0.97},
        }
        alerts = service.check_target_leakage(matrix, "target", threshold=0.95)
        assert "leaking_col" in alerts

    def test_no_leakage_below_threshold(self, service: CorrelationService) -> None:
        matrix = {
            "target": {"target": 1.0, "safe_col": 0.80},
            "safe_col": {"safe_col": 1.0, "target": 0.80},
        }
        alerts = service.check_target_leakage(matrix, "target", threshold=0.95)
        assert alerts == []

    def test_target_itself_not_flagged_as_leakage(
        self, service: CorrelationService
    ) -> None:
        matrix = {"target": {"target": 1.0}}
        alerts = service.check_target_leakage(matrix, "target", threshold=0.95)
        assert "target" not in alerts

    def test_negative_high_correlation_detected_as_leakage(
        self, service: CorrelationService
    ) -> None:
        matrix = {
            "target": {"target": 1.0, "inverse_col": -0.98},
            "inverse_col": {"inverse_col": 1.0, "target": -0.98},
        }
        alerts = service.check_target_leakage(matrix, "target", threshold=0.95)
        assert "inverse_col" in alerts
