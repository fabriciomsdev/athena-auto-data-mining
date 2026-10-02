"""MLflow experiment and run management for AthenaMining."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class MLflowRegistry:
    """
    Thin wrapper over the MLflow client following AthenaMining conventions.

    All experiments are namespaced under settings.MLFLOW_EXPERIMENT_NAME.
    Runs are tagged with pipeline_id for cross-referencing with the DB.
    """

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self._experiment_name = settings.MLFLOW_EXPERIMENT_NAME

    def _get_or_create_experiment(self) -> str:
        """Return the experiment ID, creating the experiment if it doesn't exist."""
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self._experiment_name)
            logger.info("MLflow experiment created", name=self._experiment_name)
        else:
            experiment_id = experiment.experiment_id
        return experiment_id

    def start_run(self, pipeline_id: str, model_name: str) -> str:
        """
        Start a new MLflow run and return its run_id.

        Args:
            pipeline_id: AthenaMining Pipeline UUID (used as tag for traceability).
            model_name: Model being trained (e.g. 'lightgbm').

        Returns:
            The MLflow run ID string.
        """
        experiment_id = self._get_or_create_experiment()
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{model_name}_{pipeline_id[:8]}",
            tags={
                "pipeline_id": pipeline_id,
                "model": model_name,
                "platform": "athena-mining",
            },
        )
        logger.info(
            "MLflow run started",
            run_id=run.info.run_id,
            model=model_name,
        )
        return run.info.run_id

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log hyperparameters to an existing run."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(params)

    def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        step: int = 0,
    ) -> None:
        """Log evaluation metrics to an existing run."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, run_id: str, local_path: str | Path) -> None:
        """Upload a local file as an artifact in the given run."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(local_path))
        logger.info("Artifact logged", run_id=run_id, path=str(local_path))

    def log_model(self, run_id: str, model: Any, artifact_path: str = "model") -> str:
        """
        Log a sklearn-compatible model and return the model URI.

        Args:
            run_id: The active MLflow run ID.
            model: A fitted sklearn-compatible model.
            artifact_path: Sub-folder name under the run artifacts.

        Returns:
            The MLflow model URI string.
        """
        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.log_model(model, artifact_path)
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info("Model logged to MLflow", run_id=run_id, uri=model_uri)
        return model_uri

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Explicitly end an MLflow run."""
        with mlflow.start_run(run_id=run_id):
            mlflow.end_run(status=status)

    def get_best_run_id(self, metric: str, ascending: bool = False) -> str | None:
        """
        Return the run_id of the best run in the experiment by a given metric.

        Args:
            metric: MLflow metric name (e.g. 'f1', 'rmse').
            ascending: If True, lower metric value is better (e.g. RMSE, MAE).

        Returns:
            Best run_id or None if no runs exist.
        """
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        if runs.empty:
            return None
        return str(runs.iloc[0]["run_id"])
