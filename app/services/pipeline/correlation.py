"""Correlation matrix computation and target leakage detection."""
from __future__ import annotations

from typing import Literal

import polars as pl
import structlog

logger = structlog.get_logger()


class CorrelationService:
    """
    Compute a Spearman or Pearson correlation matrix between all numeric
    features and the target column using Polars.

    Target Leakage is flagged when a feature's absolute correlation with
    the target exceeds a configurable threshold.
    """

    def compute(
        self,
        df: pl.DataFrame,
        target_column: str,
        method: Literal["spearman", "pearson"] = "spearman",
    ) -> dict[str, dict[str, float]]:
        """
        Compute the correlation matrix for all numeric columns in *df*.

        Args:
            df: The processed Polars DataFrame (post feature engineering).
            target_column: Column to highlight in leakage checks.
            method: Statistical correlation method ("spearman" or "pearson").

        Returns:
            Nested dict ``{col_a: {col_b: corr_value, ...}, ...}`` suitable
            for JSON serialisation and frontend rendering.
        """
        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            )
        ]

        if not numeric_cols:
            logger.warning("No numeric columns found for correlation computation")
            return {}

        matrix: dict[str, dict[str, float]] = {}

        for col_a in numeric_cols:
            matrix[col_a] = {}
            for col_b in numeric_cols:
                if col_a == col_b:
                    matrix[col_a][col_b] = 1.0
                    continue
                try:
                    corr = df.select(
                        pl.corr(col_a, col_b, method=method)
                    ).item()
                    matrix[col_a][col_b] = round(float(corr) if corr is not None else 0.0, 4)
                except Exception:
                    matrix[col_a][col_b] = 0.0

        logger.info(
            "Correlation matrix computed",
            method=method,
            columns=len(numeric_cols),
        )
        return matrix

    def check_target_leakage(
        self,
        correlation_matrix: dict[str, dict[str, float]],
        target_column: str,
        threshold: float = 0.95,
    ) -> list[str]:
        """
        Return a list of feature columns whose absolute correlation with
        *target_column* exceeds *threshold*.

        Args:
            correlation_matrix: Output from :meth:`compute`.
            target_column: The prediction target column name.
            threshold: Absolute correlation value above which leakage is flagged.

        Returns:
            List of column names (excluding the target itself) that exceed
            the leakage threshold.
        """
        target_corrs = correlation_matrix.get(target_column, {})
        leaking = [
            col
            for col, corr in target_corrs.items()
            if col != target_column and abs(corr) >= threshold
        ]
        if leaking:
            logger.warning(
                "Target leakage detected",
                leaking_columns=leaking,
                threshold=threshold,
            )
        return leaking
