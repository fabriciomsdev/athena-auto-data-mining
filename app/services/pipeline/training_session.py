"""Training session service — competitive model training with Optuna hyperparameter search."""
from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import polars as pl
import structlog
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

from app.db.models.pipeline import Pipeline, PipelineStatus, ProblemType
from app.db.repositories.pipeline_repository import PipelineRepository
from app.services.ingestion.storage import StorageService
from app.services.pipeline.models.base_model import BaseMLModel
from app.services.pipeline.models.lgbm_model import LightGBMModel
from app.services.pipeline.models.xgboost_model import XGBoostModel
from app.services.registry.mlflow_registry import MLflowRegistry

logger = structlog.get_logger()
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrainingConfig:
    """Configurable parameters for the training session."""

    n_trials: int = 30
    cv_folds: int = 5
    primary_metric: str = "f1"          # metric used to select the winner
    primary_metric_ascending: bool = False  # True for error metrics (RMSE, MAE)
    test_size: float = 0.2
    random_state: int = 42
    time_budget_seconds: int = 600


@dataclass
class TrainingResult:
    """Outcome of a completed training session."""

    winning_model: str
    best_params: dict[str, Any]
    metrics: dict[str, float]
    feature_importance: dict[str, float]
    mlflow_run_id: str
    model_minio_path: str


# ── Optuna search spaces ──────────────────────────────────────────────────────

def _lgbm_classification_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _xgb_classification_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ── Training session service ──────────────────────────────────────────────────

class TrainingSessionService:
    """
    Runs competitive multi-model training with Optuna hyperparameter search.

    Workflow:
    1. Load processed parquet from MinIO.
    2. Split data (StratifiedKFold for classification, TimeSeriesSplit for forecasting).
    3. For each model adapter, create an Optuna study and run n_trials.
       Each trial is logged to MLflow.
    4. Select the overall winner by the primary_metric.
    5. Retrain winner on the full training set.
    6. Compute final metrics on the held-out test set.
    7. Persist the fitted model to MinIO via joblib.
    8. Update the Pipeline DB record to COMPLETED.
    """

    _MODEL_REGISTRY: dict[str, type[BaseMLModel]] = {
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
    }

    def __init__(
        self,
        storage: StorageService,
        pipeline_repo: PipelineRepository,
        mlflow_registry: MLflowRegistry,
        config: TrainingConfig | None = None,
    ) -> None:
        self._storage = storage
        self._pipeline_repo = pipeline_repo
        self._mlflow = mlflow_registry
        self._config = config or TrainingConfig()

    async def run(
        self,
        pipeline: Pipeline,
        processed_parquet_path: str,
        pipeline_artifact_path: str,
        feature_names: list[str],
    ) -> TrainingResult:
        """
        Execute the full training competition.

        Args:
            pipeline: The Pipeline ORM record to update.
            processed_parquet_path: MinIO path of the feature-engineered parquet.
            pipeline_artifact_path: MinIO path of the fitted sklearn preprocessing pipeline.
            feature_names: Output feature names from FeatureEngineeringService.

        Returns:
            TrainingResult with winner details, metrics, and MinIO model path.
        """
        logger.info(
            "TrainingSessionService started",
            pipeline_id=str(pipeline.id),
            problem_type=pipeline.problem_type,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ── Load data ─────────────────────────────────────────────────────
            local_parquet = self._storage.download_to_temp(processed_parquet_path, tmp)
            df = pl.read_parquet(local_parquet)

            target = pipeline.target_column
            X = df.drop(target).to_numpy()
            y = df[target].to_numpy()

            # ── Train / test split ─────────────────────────────────────────────
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self._config.test_size,
                random_state=self._config.random_state,
                stratify=y if pipeline.problem_type == ProblemType.CLASSIFICATION else None,
            )

            # ── Competition ────────────────────────────────────────────────────
            problem_type = ProblemType(pipeline.problem_type)
            best_score = float("-inf") if not self._config.primary_metric_ascending else float("inf")
            winner_name: str = ""
            winner_params: dict[str, Any] = {}

            for model_name, model_cls in self._MODEL_REGISTRY.items():
                run_id = self._mlflow.start_run(str(pipeline.id), model_name)
                score, best_params = self._run_optuna_study(
                    model_name=model_name,
                    model_cls=model_cls,
                    X_train=X_train,
                    y_train=y_train,
                    problem_type=problem_type,
                    mlflow_run_id=run_id,
                )
                self._mlflow.end_run(run_id)

                logger.info(
                    "Model trial complete",
                    model=model_name,
                    score=score,
                    metric=self._config.primary_metric,
                )

                is_better = (
                    score < best_score
                    if self._config.primary_metric_ascending
                    else score > best_score
                )
                if is_better:
                    best_score = score
                    winner_name = model_name
                    winner_params = best_params

            # ── Retrain winner on full train set ───────────────────────────────
            logger.info("Retraining winner", model=winner_name)
            X_val_split, X_retrain, y_val_split, y_retrain = train_test_split(
                X_train, y_train, test_size=0.8, random_state=self._config.random_state
            )
            winner_model = self._MODEL_REGISTRY[winner_name]()
            winner_model.train(X_retrain, y_retrain, X_val_split, y_val_split, dict(winner_params))

            # ── Final evaluation on held-out test set ──────────────────────────
            metrics = winner_model.evaluate(X_test, y_test, problem_type)
            feature_importance = winner_model.get_feature_importance()

            # ── Log winner to MLflow ───────────────────────────────────────────
            final_run_id = self._mlflow.start_run(str(pipeline.id), f"{winner_name}_final")
            self._mlflow.log_params(final_run_id, winner_params)
            self._mlflow.log_metrics(final_run_id, metrics)
            model_uri = self._mlflow.log_model(final_run_id, winner_model.get_fitted_model())
            self._mlflow.end_run(final_run_id)

            # ── Persist model to MinIO ─────────────────────────────────────────
            model_filename = f"model_{uuid.uuid4().hex}.joblib"
            local_model = tmp / model_filename
            joblib.dump(winner_model.get_fitted_model(), local_model)
            model_path = self._storage.upload_file(
                local_path=local_model,
                object_name=f"models/{pipeline.id}/{model_filename}",
                bucket="athena-models",
            )

            # ── Update Pipeline DB record ──────────────────────────────────────
            await self._pipeline_repo.update_training_result(
                pipeline_id=pipeline.id,
                mlflow_run_id=final_run_id,
                mlflow_experiment_id=self._mlflow._get_or_create_experiment(),
                model_minio_path=model_path,
                metrics_json=json.dumps(metrics),
            )

        result = TrainingResult(
            winning_model=winner_name,
            best_params=winner_params,
            metrics=metrics,
            feature_importance=feature_importance,
            mlflow_run_id=final_run_id,
            model_minio_path=model_path,
        )
        logger.info(
            "TrainingSessionService completed",
            winner=winner_name,
            metrics=metrics,
        )
        return result

    def _run_optuna_study(
        self,
        model_name: str,
        model_cls: type[BaseMLModel],
        X_train: np.ndarray,
        y_train: np.ndarray,
        problem_type: ProblemType,
        mlflow_run_id: str,
    ) -> tuple[float, dict[str, Any]]:
        """
        Run an Optuna study for one model and return (best_score, best_params).
        """
        search_space_fn = (
            _lgbm_classification_space
            if model_name == "lightgbm"
            else _xgb_classification_space
        )
        cv = (
            StratifiedKFold(
                n_splits=self._config.cv_folds,
                shuffle=True,
                random_state=self._config.random_state,
            )
            if problem_type == ProblemType.CLASSIFICATION
            else TimeSeriesSplit(n_splits=self._config.cv_folds)
        )

        def objective(trial: optuna.Trial) -> float:
            params = search_space_fn(trial)
            scores: list[float] = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = model_cls()
                model.train(X_tr, y_tr, X_val, y_val, dict(params))
                fold_metrics = model.evaluate(X_val, y_val, problem_type)
                scores.append(fold_metrics.get(self._config.primary_metric, 0.0))

            mean_score = float(np.mean(scores))
            self._mlflow.log_metrics(
                mlflow_run_id,
                {f"trial_{self._config.primary_metric}": mean_score},
                step=trial.number,
            )
            return mean_score

        direction = "minimize" if self._config.primary_metric_ascending else "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(
            objective,
            n_trials=self._config.n_trials,
            timeout=self._config.time_budget_seconds,
            show_progress_bar=False,
        )
        self._mlflow.log_params(mlflow_run_id, study.best_params)
        return study.best_value, study.best_params
