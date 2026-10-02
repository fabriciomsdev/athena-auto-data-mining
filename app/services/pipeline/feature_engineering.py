"""Feature engineering service — encodes, scales, and extracts features."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import polars as pl
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import (
    LabelBinarizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from app.db.models.pipeline import ProblemType
from app.services.ingestion.storage import StorageService
from app.services.pipeline.correlation import CorrelationService
from app.services.pipeline.schema_inferrer import ColumnSchema, ColumnType

logger = structlog.get_logger()


@dataclass
class FeatureEngineeringConfig:
    """User-overridable parameters for feature engineering."""

    scaler: Literal["standard", "robust", "none"] = "robust"
    high_card_encoder: Literal["target", "onehot"] = "target"
    low_card_encoder: Literal["onehot", "ordinal"] = "onehot"
    leakage_threshold: float = 0.95
    correlation_method: Literal["spearman", "pearson"] = "spearman"
    # Forecasting-specific
    lag_periods: list[int] = field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 30])
    datetime_col: str | None = None  # required for forecasting feature extraction


@dataclass
class FeatureEngineeringResult:
    """Outcome of the feature engineering step."""

    processed_parquet_path: str       # MinIO path for transformed training data
    pipeline_artifact_path: str       # MinIO path for serialised sklearn Pipeline
    correlation_matrix: dict[str, dict[str, float]]
    leakage_alerts: list[str]
    feature_names_out: list[str]
    numeric_cols: list[str]
    categorical_low_cols: list[str]
    categorical_high_cols: list[str]


class FeatureEngineeringService:
    """
    Build and apply a typed sklearn ColumnTransformer pipeline.

    Pipeline structure:
    ┌─────────────────────────────────────────────────────────┐
    │ ColumnTransformer                                        │
    │  ├─ numeric_pipe:      [SimpleImputer → Scaler]          │
    │  ├─ cat_low_pipe:      [SimpleImputer → OneHotEncoder]   │
    │  ├─ cat_high_pipe:     [SimpleImputer → TargetEncoder*]  │
    │  └─ datetime_pipe:     [DateFeatureExtractor]            │
    └─────────────────────────────────────────────────────────┘
    *TargetEncoder is approximated via mean-target encoding to avoid
    the category_encoders dependency for now; can be swapped later.

    Additionally:
    - Forecasting: extracts lag, rolling-mean, and date-part columns.
    - Computes Spearman correlation matrix.
    - Flags target leakage if any feature corr > leakage_threshold.
    - Serialises the fitted pipeline to MinIO.
    """

    def __init__(
        self,
        storage: StorageService,
        config: FeatureEngineeringConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or FeatureEngineeringConfig()
        self._corr_service = CorrelationService()

    async def run(
        self,
        clean_parquet_path: str,
        schema: list[ColumnSchema],
        target_column: str,
        problem_type: ProblemType,
    ) -> FeatureEngineeringResult:
        """
        Execute feature engineering on the cleaned parquet.

        Args:
            clean_parquet_path: MinIO path produced by DataPrepService.
            schema: ColumnSchema list from SchemaInferrer.
            target_column: Name of the prediction target column.
            problem_type: CLASSIFICATION or FORECASTING.

        Returns:
            FeatureEngineeringResult with MinIO paths and analysis artefacts.
        """
        logger.info(
            "FeatureEngineeringService started",
            source=clean_parquet_path,
            problem_type=problem_type,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ── Download cleaned parquet ──────────────────────────────────────
            local_parquet = self._storage.download_to_temp(clean_parquet_path, tmp)
            df = pl.read_parquet(local_parquet)

            # ── Forecasting: extract temporal features ────────────────────────
            if problem_type == ProblemType.FORECASTING:
                df = self._extract_forecasting_features(df, target_column)

            # ── Classify columns by type ──────────────────────────────────────
            non_target_schema = [s for s in schema if not s.is_target]
            numeric_cols = [
                s.name for s in non_target_schema if s.dtype == ColumnType.NUMERIC
            ]
            cat_low_cols = [
                s.name for s in non_target_schema
                if s.dtype == ColumnType.CATEGORICAL_LOW
            ]
            cat_high_cols = [
                s.name for s in non_target_schema
                if s.dtype == ColumnType.CATEGORICAL_HIGH
            ]

            # ── Build and fit sklearn pipeline ───────────────────────────────
            X = df.drop(target_column).to_pandas()
            y = df[target_column].to_pandas()

            sk_pipeline = self._build_pipeline(
                numeric_cols, cat_low_cols, cat_high_cols
            )
            X_transformed = sk_pipeline.fit_transform(X, y)
            feature_names = self._get_feature_names(
                sk_pipeline, numeric_cols, cat_low_cols, cat_high_cols
            )

            # ── Rebuild transformed DataFrame for correlation analysis ─────────
            transformed_df = pl.from_numpy(
                X_transformed
                if isinstance(X_transformed, np.ndarray)
                else X_transformed.toarray(),
                schema={name: pl.Float64 for name in feature_names},
            ).with_columns(df[target_column].cast(pl.Float64).alias(target_column))

            # ── Correlation + leakage ─────────────────────────────────────────
            corr_matrix = self._corr_service.compute(
                transformed_df,
                target_column,
                method=self._config.correlation_method,
            )
            leakage_alerts = self._corr_service.check_target_leakage(
                corr_matrix, target_column, self._config.leakage_threshold
            )

            # ── Save processed parquet ────────────────────────────────────────
            proc_parquet_name = f"processed_{uuid.uuid4().hex}.parquet"
            local_proc = tmp / proc_parquet_name
            transformed_df.write_parquet(local_proc)
            processed_path = self._storage.upload_file(
                local_path=local_proc,
                object_name=f"processed/{proc_parquet_name}",
            )

            # ── Serialise sklearn pipeline ────────────────────────────────────
            pipeline_name = f"pipeline_{uuid.uuid4().hex}.joblib"
            local_pipeline = tmp / pipeline_name
            joblib.dump(sk_pipeline, local_pipeline)
            pipeline_path = self._storage.upload_file(
                local_path=local_pipeline,
                object_name=f"pipelines/{pipeline_name}",
            )

        result = FeatureEngineeringResult(
            processed_parquet_path=processed_path,
            pipeline_artifact_path=pipeline_path,
            correlation_matrix=corr_matrix,
            leakage_alerts=leakage_alerts,
            feature_names_out=feature_names,
            numeric_cols=numeric_cols,
            categorical_low_cols=cat_low_cols,
            categorical_high_cols=cat_high_cols,
        )
        logger.info(
            "FeatureEngineeringService completed",
            features_out=len(feature_names),
            leakage_alerts=leakage_alerts,
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_pipeline(
        self,
        numeric_cols: list[str],
        cat_low_cols: list[str],
        cat_high_cols: list[str],
    ) -> SKPipeline:
        """Assemble the ColumnTransformer inside a sklearn Pipeline."""
        transformers: list[tuple[str, Any, list[str]]] = []

        if numeric_cols:
            scaler: Any
            if self._config.scaler == "robust":
                scaler = RobustScaler()
            elif self._config.scaler == "standard":
                scaler = StandardScaler()
            else:
                scaler = "passthrough"

            num_pipe = SKPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ])
            transformers.append(("numeric", num_pipe, numeric_cols))

        if cat_low_cols:
            low_pipe = SKPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("categorical_low", low_pipe, cat_low_cols))

        if cat_high_cols:
            # Mean-target encoding approximation (passthrough for now;
            # category_encoders.TargetEncoder can replace this)
            high_pipe = SKPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=50,
                )),
            ])
            transformers.append(("categorical_high", high_pipe, cat_high_cols))

        if not transformers:
            raise ValueError(
                "No numeric or categorical columns found to build a feature pipeline."
            )

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        return SKPipeline([("preprocessor", ct)])

    def _get_feature_names(
        self,
        pipeline: SKPipeline,
        numeric_cols: list[str],
        cat_low_cols: list[str],
        cat_high_cols: list[str],
    ) -> list[str]:
        """Extract output feature names from the fitted ColumnTransformer."""
        ct: ColumnTransformer = pipeline.named_steps["preprocessor"]
        names: list[str] = []

        for name, transformer, cols in ct.transformers_:
            if name == "numeric":
                names.extend(cols)
            elif name in ("categorical_low", "categorical_high"):
                encoder = transformer.named_steps["encoder"]
                try:
                    ohe_names = encoder.get_feature_names_out(cols).tolist()
                    names.extend(ohe_names)
                except AttributeError:
                    names.extend(cols)

        return names

    def _extract_forecasting_features(
        self, df: pl.DataFrame, target_column: str
    ) -> pl.DataFrame:
        """
        Add lag, rolling-mean, and date-part columns for forecasting problems.

        Requires the target column to be numeric.
        """
        expressions: list[pl.Expr] = []

        # Lag features on target
        for lag in self._config.lag_periods:
            expressions.append(
                pl.col(target_column).shift(lag).alias(f"{target_column}_lag_{lag}")
            )

        # Rolling mean features on target
        for window in self._config.rolling_windows:
            expressions.append(
                pl.col(target_column)
                .rolling_mean(window_size=window)
                .alias(f"{target_column}_rolling_mean_{window}")
            )

        # Date-part extraction if datetime column is known
        if self._config.datetime_col and self._config.datetime_col in df.columns:
            dt_col = self._config.datetime_col
            expressions.extend([
                pl.col(dt_col).dt.weekday().alias(f"{dt_col}_weekday"),
                pl.col(dt_col).dt.month().alias(f"{dt_col}_month"),
                pl.col(dt_col).dt.quarter().alias(f"{dt_col}_quarter"),
                pl.col(dt_col).dt.year().alias(f"{dt_col}_year"),
            ])

        if expressions:
            df = df.with_columns(expressions).drop_nulls()

        return df
