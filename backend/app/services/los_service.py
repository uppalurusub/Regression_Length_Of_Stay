"""
Hospital Length of Stay (LOS) Service Layer.

Responsibilities:
    - Validate request-level feature payloads.
    - Orchestrate single and batch predictions.
    - Delegate all MLflow/model operations to the implementation layer.
    - Return API-independent result dictionaries.

Invocation:
    Router -> HospitalLOSService -> HospitalLOSImplementation
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd

from app.config.settings import settings
logger = settings.get_logger()

from app.implementations.los_implementation import (
    HospitalLOSImplementation,
    LOSImplementationError,
    LOSModelUnavailableError,
    LOSInferenceError,
    LOSInputError,
    los_implementation,
)
import mlflow
from datetime import datetime

from monitoring.prediction_monitor import (
    PredictionMonitor,
    PredictionTimer,
)
import uuid

prediction_monitor = PredictionMonitor(
    tracking_uri="http://localhost:5000",
    experiment_name="hospital-los-inference",
    enable_mlflow=True,
)



MAX_BATCH_SIZE = int(os.getenv("LOS_MAX_BATCH_SIZE", "500"))
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "hospital-los-regressor"
)
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://localhost:5000"
)


class LOSModelError(Exception):
    """Base exception for service-layer LOS errors."""


class LOSModelLoadError(LOSModelError):
    """Champion model loading or registry metadata failure."""


class LOSFeatureValidationError(LOSModelError):
    """Invalid single or batch feature payload."""


class LOSPredictionError(LOSModelError):
    """Model inference failure."""


class HospitalLOSService:
    """Business orchestration layer; no direct MLflow calls."""

    def __init__(
        self,
        implementation: HospitalLOSImplementation | None = None,
    ) -> None:
        self.implementation = implementation or los_implementation
        self.model_name = REGISTERED_MODEL_NAME
        self.model_alias = MODEL_ALIAS
        self.model_uri = MODEL_URI
        self.max_batch_size = MAX_BATCH_SIZE

        logger.info(
            "HospitalLOSService initialized | model={} | alias={}",
            self.model_name,
            self.model_alias,
        )

    @staticmethod
    def _validate_feature_record(features: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(features, dict) or not features:
            raise LOSFeatureValidationError(
                "Patient features cannot be empty."
            )
        if not all(
            isinstance(key, str) and key.strip()
            for key in features
        ):
            raise LOSFeatureValidationError(
                "Feature names must be non-empty strings."
            )
        return pd.DataFrame([features])

    def validate_features(self, features: dict[str, Any]) -> pd.DataFrame:
        """Validate payload shape; signature/type validation is delegated."""
        return self._validate_feature_record(features)

    def validate_batch_features(
        self,
        records: list[dict[str, Any]],
    ) -> pd.DataFrame:
        if not isinstance(records, list) or not records:
            raise LOSFeatureValidationError(
                "Batch records cannot be empty."
            )
        if len(records) > self.max_batch_size:
            raise LOSFeatureValidationError(
                f"Batch size exceeds maximum limit of {self.max_batch_size}."
            )
        if any(not isinstance(record, dict) or not record for record in records):
            raise LOSFeatureValidationError(
                "Every batch record must contain features."
            )
        if any(
            not all(isinstance(key, str) and key.strip() for key in record)
            for record in records
        ):
            raise LOSFeatureValidationError(
                "Feature names must be non-empty strings."
            )
        frame = pd.DataFrame(records)
        if frame.isna().all(axis=1).any():
            raise LOSFeatureValidationError(
                "A batch record cannot contain only missing values."
            )
        return frame

    def get_model(self):
        """Delegate model loading to implementation."""
        try:
            return self.implementation.load_model()
        except LOSImplementationError as exc:
            raise LOSModelLoadError(str(exc)) from exc

    def reload_model(self):
        """Delegate champion reload to implementation."""
        try:
            return self.implementation.reload_model()
        except LOSImplementationError as exc:
            raise LOSModelLoadError(str(exc)) from exc

    def get_model_metadata(self) -> dict[str, Any]:
        """Delegate registry metadata retrieval to implementation."""
        try:
            return self.implementation.get_model_metadata()
        except LOSImplementationError as exc:
            raise LOSModelLoadError(str(exc)) from exc

    def predict_single(self, features: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        mlflow.set_tracking_uri(
            "http://localhost:5000"
        )

        mlflow.set_experiment(
            "hospital-los-regression"
        )
        logger.info(
            "Service: single prediction orchestration started | features={}",
            len(features) if isinstance(features, dict) else 0,
        )
        correlation_id = str(uuid.uuid4())
        try:
            input_data = self.validate_features(features)
            with mlflow.start_run(
                run_name=
                f"hospital_los_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):

            # Service -> Implementation: implementation validates against
            # the MLflow signature, normalizes dtypes and runs model.predict.
                predictions = self.implementation.predict(
                    input_data=input_data,
                    operation_type="single",
                )
                predicted_los = float(predictions[0])
                elapsed = time.perf_counter() - started

                mlflow.log_param(
                    "input_text",
                    features
                )

                mlflow.log_param(
                    "prediction_result",
                    predicted_los
                )

                

                mlflow.set_tag(
                    "source",
                    "FastAPI"
                )

                mlflow.set_tag(
                    "endpoint",
                    "/predict_single"
                )

                with PredictionTimer() as timer:
                    prediction = self.implementation.predict(\
                        input_data=input_data,
                        operation_type="single"
                    )

                monitoring_result = (
                    prediction_monitor.monitor_single_prediction(
                        prediction=float(prediction[0]),
                        execution_time_seconds=timer.elapsed_seconds,
                        correlation_id=correlation_id,
                    )
                )


            result = {
                "predicted_los_days": predicted_los,
                "unit": "days",
                "model_name": self.model_name,
                "model_alias": self.model_alias,
                "execution_time_seconds": round(elapsed, 4),
                "monitoring": monitoring_result
            }
            logger.success(
                "Service: single prediction completed | duration={:.4f}s",
                elapsed,
            )
            return result

        except LOSFeatureValidationError:
            raise
        except LOSInputError as exc:
            raise LOSFeatureValidationError(str(exc)) from exc
        except LOSModelUnavailableError as exc:
            raise LOSModelLoadError(str(exc)) from exc
        except (LOSInferenceError, LOSImplementationError) as exc:
            raise LOSPredictionError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Service: single prediction failed.")
            raise LOSPredictionError(
                "Unable to generate LOS prediction."
            ) from exc

    def predict_batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        correlation_id = str(uuid.uuid4())

        mlflow.set_tracking_uri(
            "http://localhost:5000"
        )
        
        mlflow.set_experiment(
            "hospital-los-regression"
        )

        logger.info(
            "Service: batch prediction orchestration started | records={}",
            len(records) if isinstance(records, list) else 0,
        )
        try:
            input_data = self.validate_batch_features(records)

            with mlflow.start_run(
                run_name=
                f"hospital_los_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):

                # Service -> Implementation: shared inference path for batch.
                predictions = self.implementation.predict(
                    input_data=input_data,
                    operation_type="batch",
                )
                elapsed = time.perf_counter() - started

                mlflow.log_param(
                    "input_text",
                    input_data
                )

                mlflow.log_param(
                    "prediction_result",
                    predictions
                )

                

                mlflow.set_tag(
                    "source",
                    "FastAPI"
                )

                mlflow.set_tag(
                    "endpoint",
                    "/predict_batch"
                )

            with PredictionTimer() as timer:
                predictions = self.implementation.predict(
                    input_data=input_data,
                    operation_type="batch"
                )

            monitoring_result = (
                prediction_monitor.monitor_batch_prediction(
                    predictions=predictions,
                    execution_time_seconds=timer.elapsed_seconds,
                    correlation_id=correlation_id,
                )
            )


            result = {
                "predictions": predictions.tolist(),
                "unit": "days",
                "model_name": self.model_name,
                "model_alias": self.model_alias,
                "total_records": len(predictions),
                "execution_time_seconds": round(elapsed, 4),
                "monitoring": monitoring_result
            }
            logger.success(
                "Service: batch prediction completed | records={} | duration={:.4f}s",
                len(predictions),
                elapsed,
            )
            return result

        except LOSFeatureValidationError:
            raise
        except LOSInputError as exc:
            raise LOSFeatureValidationError(str(exc)) from exc
        except LOSModelUnavailableError as exc:
            raise LOSModelLoadError(str(exc)) from exc
        except (LOSInferenceError, LOSImplementationError) as exc:
            raise LOSPredictionError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Service: batch prediction failed.")
            raise LOSPredictionError(
                "Unable to generate batch LOS predictions."
            ) from exc

    def health_check(self) -> dict[str, Any]:
        """Delegate model and registry health checks to implementation."""
        try:
            return self.implementation.health_check()
        except LOSImplementationError as exc:
            raise LOSModelLoadError(str(exc)) from exc


# Singleton service used by the FastAPI router.
los_service = HospitalLOSService()
