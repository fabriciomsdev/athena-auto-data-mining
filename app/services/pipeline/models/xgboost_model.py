"""XGBoost model adapter."""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.db.models.pipeline import ProblemType
from app.services.pipeline.models.base_model import BaseMLModel

logger = structlog.get_logger()


class XGBoostModel(BaseMLModel):
    """
    XGBoost adapter for classification and regression.

    Uses early stopping rounds on the eval set to prevent overfitting.
    """

    def __init__(self) -> None:
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return "xgboost"

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> "XGBoostModel":
        from xgboost import XGBClassifier, XGBRegressor

        n_estimators = params.pop("n_estimators", 500)
        problem_type = params.pop("_problem_type", ProblemType.CLASSIFICATION)

        if problem_type == ProblemType.CLASSIFICATION:
            self._model = XGBClassifier(
                n_estimators=n_estimators,
                early_stopping_rounds=50,
                eval_metric="logloss",
                verbosity=0,
                **params,
            )
        else:
            self._model = XGBRegressor(
                n_estimators=n_estimators,
                early_stopping_rounds=50,
                eval_metric="rmse",
                verbosity=0,
                **params,
            )

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        logger.info("XGBoost training complete", best_iteration=self._model.best_iteration)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        assert self._model is not None, "Model has not been trained yet."
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        problem_type: ProblemType,
    ) -> dict[str, float]:
        assert self._model is not None, "Model has not been trained yet."

        if problem_type == ProblemType.CLASSIFICATION:
            y_pred = self.predict(X_val)
            y_proba = self.predict_proba(X_val)
            metrics: dict[str, float] = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "f1": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
                "precision": float(
                    precision_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_val, y_pred, average="weighted", zero_division=0)
                ),
            }
            if y_proba is not None:
                try:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")
                    )
                except Exception:
                    pass
            return metrics

        y_pred_raw = self.predict(X_val).astype(float)
        mse = mean_squared_error(y_val, y_pred_raw)
        return {
            "mae": float(mean_absolute_error(y_val, y_pred_raw)),
            "rmse": float(np.sqrt(mse)),
            "mape": float(
                np.mean(np.abs((y_val - y_pred_raw) / np.where(y_val == 0, 1, y_val))) * 100
            ),
        }

    def get_feature_importance(self) -> dict[str, float]:
        assert self._model is not None, "Model has not been trained yet."
        scores = self._model.feature_importances_
        total = scores.sum() or 1.0
        names = [f"f{i}" for i in range(len(scores))]
        return {name: float(score / total) for name, score in zip(names, scores)}

    def get_fitted_model(self) -> Any:
        return self._model
