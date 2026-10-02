"""Abstract base class for all ML model adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.db.models.pipeline import ProblemType


class BaseMLModel(ABC):
    """
    Defines the contract every model adapter must implement.

    All subclasses must be fully typed and must not hold mutable global state.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for this model (e.g. 'lightgbm')."""
        ...

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any],
    ) -> "BaseMLModel":
        """
        Fit the model and return self for chaining.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation feature matrix (used for early stopping).
            y_val: Validation target vector.
            params: Hyperparameters provided by the Optuna trial.

        Returns:
            The fitted model instance.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """
        Return class probabilities for *X* (classification only).
        Returns None for regression/forecasting models.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        problem_type: ProblemType,
    ) -> dict[str, float]:
        """
        Compute evaluation metrics appropriate for *problem_type*.

        Classification metrics: accuracy, f1, precision, recall, roc_auc.
        Forecasting metrics:    mae, rmse, mape.

        Returns:
            Dict mapping metric name to float value.
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """
        Return a mapping of feature name → importance score.
        Values are normalised to sum to 1.0.
        """
        ...

    @abstractmethod
    def get_fitted_model(self) -> Any:
        """Return the underlying fitted model object for serialisation."""
        ...
