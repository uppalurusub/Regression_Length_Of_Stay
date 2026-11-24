
"""
Prediction Monitoring Module
----------------------------
Project : Hospital Length of Stay (LOS) Prediction
Purpose : Monitor single and batch prediction operations.
Logging : Loguru
Tracking: MLflow

Monitors:
    - Prediction execution time
    - Prediction count
    - Batch size
    - Predicted LOS statistics
    - Negative prediction count
    - Prediction failures
    - MLflow operational metrics

Privacy:
    - Does not log patient identifiers.
    - Does not log patient input features.
    - Does not log raw prediction payloads.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import mlflow
import numpy as np
from loguru import logger
from mlflow.tracking import MlflowClient


class PredictionMonitor:
    """
    Monitors operational metrics for Hospital LOS predictions.

    This class records inference metrics only. Actual model quality
    metrics require ground-truth LOS values and should be calculated
    separately in an evaluation workflow.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "hospital-los-inference",
        enable_mlflow: bool = True,
    ) -> None:

        self.experiment_name = experiment_name
        self.enable_mlflow = enable_mlflow
        self.client: Optional[MlflowClient] = None

        if self.enable_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            self.client = MlflowClient()

            mlflow.set_experiment(self.experiment_name)

        logger.info(
            "PredictionMonitor initialized | "
            "experiment={} | mlflow_enabled={}",
            self.experiment_name,
            self.enable_mlflow,
        )

    @staticmethod
    def _utc_timestamp() -> str:
        """Return timezone-aware UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_predictions(
        predictions: Sequence[Any] | np.ndarray,
    ) -> np.ndarray:
        """
        Convert predictions into a validated numeric NumPy array.
        """
        values = np.asarray(predictions, dtype=float).reshape(-1)

        if values.size == 0:
            raise ValueError("Prediction output cannot be empty.")

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Predictions contain NaN or infinite values."
            )

        return values

    @staticmethod
    def _prediction_statistics(
        predictions: np.ndarray,
    ) -> dict[str, float | int]:
        """
        Calculate descriptive prediction statistics.

        These are not model accuracy metrics.
        """
        return {
            "prediction_count": int(predictions.size),
            "predicted_los_mean": float(np.mean(predictions)),
            "predicted_los_min": float(np.min(predictions)),
            "predicted_los_max": float(np.max(predictions)),
            "predicted_los_std": float(np.std(predictions)),
            "negative_prediction_count": int(
                np.sum(predictions < 0)
            ),
        }

    

    def monitor_single_prediction(
        self,
        prediction: float,
        execution_time_seconds: float,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Monitor a single LOS prediction.

        Args:
            prediction:
                Predicted LOS value.

            execution_time_seconds:
                Time taken by the prediction operation.

            correlation_id:
                Non-identifying request correlation ID.

        Returns:
            Monitoring result dictionary.
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        if execution_time_seconds < 0:
            raise ValueError(
                "Execution time cannot be negative."
            )

        predictions = self._normalize_predictions([prediction])

        metrics = self._prediction_statistics(predictions)
        metrics["execution_time_seconds"] = float(
            execution_time_seconds
        )
        metrics["throughput_predictions_per_second"] = (
            1.0 / execution_time_seconds
            if execution_time_seconds > 0
            else 0.0
        )

        result = {
            "correlation_id": correlation_id,
            "timestamp_utc": self._utc_timestamp(),
            "monitoring_type": "single_prediction",
            "metrics": metrics,
        }

        logger.info(
            "Single prediction monitored | "
            "correlation_id={} | prediction_count={} | "
            "execution_time_seconds={:.6f} | "
            "predicted_los={:.4f}",
            correlation_id,
            metrics["prediction_count"],
            execution_time_seconds,
            prediction,
        )

        if metrics["negative_prediction_count"] > 0:
            logger.warning(
                "Negative LOS prediction detected | "
                "correlation_id={}",
                correlation_id,
            )

        

        return result

    def monitor_batch_prediction(
        self,
        predictions: Sequence[Any] | np.ndarray,
        execution_time_seconds: float,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Monitor batch LOS predictions.

        Args:
            predictions:
                Sequence of predicted LOS values.

            execution_time_seconds:
                Total batch inference execution time.

            correlation_id:
                Non-identifying request correlation ID.

        Returns:
            Monitoring result dictionary.
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        if execution_time_seconds < 0:
            raise ValueError(
                "Execution time cannot be negative."
            )

        values = self._normalize_predictions(predictions)

        metrics = self._prediction_statistics(values)
        metrics["batch_size"] = int(values.size)
        metrics["execution_time_seconds"] = float(
            execution_time_seconds
        )
        metrics["throughput_predictions_per_second"] = (
            float(values.size / execution_time_seconds)
            if execution_time_seconds > 0
            else 0.0
        )

        result = {
            "correlation_id": correlation_id,
            "timestamp_utc": self._utc_timestamp(),
            "monitoring_type": "batch_prediction",
            "metrics": metrics,
        }

        logger.info(
            "Batch prediction monitored | "
            "correlation_id={} | batch_size={} | "
            "execution_time_seconds={:.6f} | "
            "throughput={:.2f} predictions/sec",
            correlation_id,
            metrics["batch_size"],
            execution_time_seconds,
            metrics["throughput_predictions_per_second"],
        )

        logger.info(
            "Prediction distribution | "
            "mean={:.4f} | min={:.4f} | max={:.4f} | "
            "std={:.4f}",
            metrics["predicted_los_mean"],
            metrics["predicted_los_min"],
            metrics["predicted_los_max"],
            metrics["predicted_los_std"],
        )

        if metrics["negative_prediction_count"] > 0:
            logger.warning(
                "Negative LOS predictions detected | "
                "count={} | correlation_id={}",
                metrics["negative_prediction_count"],
                correlation_id,
            )

        

        return result

    def monitor_prediction_failure(
        self,
        error: Exception,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Record prediction failures without logging patient data.
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        logger.exception(
            "Prediction execution failed | "
            "correlation_id={} | error_type={}",
            correlation_id,
            type(error).__name__,
        )

        return {
            "correlation_id": correlation_id,
            "timestamp_utc": self._utc_timestamp(),
            "status": "FAILED",
            "error_type": type(error).__name__,
        }


class PredictionTimer:
    """
    Context manager for measuring prediction execution time.

    Example:
        with PredictionTimer() as timer:
            prediction = model.predict(input_data)

        execution_time = timer.elapsed_seconds
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "PredictionTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.end_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError(
                "PredictionTimer has not completed."
            )

        return self.end_time - self.start_time