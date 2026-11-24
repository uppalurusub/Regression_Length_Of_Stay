
"""
Hospital Length of Stay (LOS) - ML Implementation Layer.

Responsibilities:
    1. Connect to MLflow Tracking Server.
    2. Load the promoted champion model.
    3. Retrieve registered model metadata.
    4. Execute single and batch predictions.
    5. Validate prediction outputs.
    6. Expose model signature information.
    7. Support champion model reload.
    8. Log operations using Loguru.

MLflow:
    Registered Model : hospital-los-regressor
    Alias            : champion
    Model URI        : models:/hospital-los-regressor@champion

Model Framework:
    MLflow PyFunc
"""

from __future__ import annotations

import os
import time
import threading

from typing import Any

import mlflow
import mlflow.pyfunc

import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from app.config.settings import settings
logger = settings.get_logger()


# =========================================================
# CONFIGURATION
# =========================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "hospital-los-regressor",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)

MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
)


# =========================================================
# CUSTOM EXCEPTIONS
# =========================================================

class LOSImplementationError(Exception):
    """Base exception for LOS implementation errors."""


class LOSModelUnavailableError(LOSImplementationError):
    """Raised when the champion model is unavailable."""


class LOSInferenceError(LOSImplementationError):
    """Raised when model inference fails."""


class LOSInputError(LOSImplementationError):
    """Raised when prediction input is invalid."""


# =========================================================
# HOSPITAL LOS IMPLEMENTATION
# =========================================================

class HospitalLOSImplementation:
    """
    Low-level MLflow model integration and inference.

    This class does not depend on FastAPI or Pydantic.

    Example:
        implementation = HospitalLOSImplementation()

        model = implementation.load_model()

        predictions = implementation.predict(
            input_data
        )
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        model_name: str = REGISTERED_MODEL_NAME,
        model_alias: str = MODEL_ALIAS,
        client: MlflowClient | None = None,
    ) -> None:

        self.tracking_uri = (
            tracking_uri or MLFLOW_TRACKING_URI
        )

        self.model_name = model_name
        self.model_alias = model_alias

        self.model_uri = (
            f"models:/{self.model_name}@{self.model_alias}"
        )

        self.client = client

        self._model: mlflow.pyfunc.PyFuncModel | None = None

        self._model_lock = threading.RLock()

        # Configure MLflow tracking.
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

        self.client = self.client or MlflowClient(
            tracking_uri=self.tracking_uri
        )

        logger.info(
            "HospitalLOSImplementation initialized | "
            "tracking_uri={} | model_name={} | alias={}",
            self.tracking_uri,
            self.model_name,
            self.model_alias,
        )

    # =====================================================
    # MODEL LOADING
    # =====================================================

    def load_model(
        self,
        force_reload: bool = False,
    ) -> mlflow.pyfunc.PyFuncModel:
        """
        Load the model currently assigned to the champion alias.

        The loaded model is cached in this implementation
        instance to avoid repeated MLflow downloads.

        Args:
            force_reload:
                Force loading the current champion again.

        Returns:
            MLflow PyFunc model.
        """

        with self._model_lock:

            if self._model is not None and not force_reload:

                return self._model

            start_time = time.perf_counter()

            try:

                logger.info(
                    "Loading promoted Hospital LOS model | "
                    "model_uri={}",
                    self.model_uri,
                )

                model = mlflow.pyfunc.load_model(
                    model_uri=self.model_uri
                )

                execution_time = (
                    time.perf_counter() - start_time
                )

                self._model = model

                logger.success(
                    "Champion model loaded successfully | "
                    "model_name={} | alias={} | "
                    "execution_time_seconds={:.4f}",
                    self.model_name,
                    self.model_alias,
                    execution_time,
                )

                return self._model

            except Exception as exc:

                logger.exception(
                    "Failed to load Hospital LOS champion model | "
                    "model_uri={}",
                    self.model_uri,
                )

                raise LOSModelUnavailableError(
                    "Unable to load promoted Hospital LOS model."
                ) from exc

    # =====================================================
    # MODEL RELOAD
    # =====================================================

    def reload_model(
        self,
    ) -> mlflow.pyfunc.PyFuncModel:
        """
        Reload the currently promoted champion model.

        Use this after assigning a new champion alias.
        """

        logger.warning(
            "Reloading Hospital LOS champion model."
        )

        return self.load_model(
            force_reload=True
        )

    # =====================================================
    # MODEL METADATA
    # =====================================================

    def get_model_metadata(self) -> dict[str, Any]:
        """
        Retrieve metadata for the model version assigned
        to the champion alias.
        """

        try:

            model_version = (
                self.client.get_model_version_by_alias(
                    name=self.model_name,
                    alias=self.model_alias,
                )
            )

            metadata = {
                "model_name": self.model_name,
                "model_alias": self.model_alias,
                "model_version": str(
                    model_version.version
                ),
                "source_run_id": model_version.run_id,
                "model_uri": self.model_uri,
                "tracking_uri": self.tracking_uri,
                "status": model_version.status,
                "creation_timestamp": (
                    model_version.creation_timestamp
                ),
            }

            logger.info(
                "Champion model metadata retrieved | "
                "model_name={} | version={} | alias={}",
                self.model_name,
                model_version.version,
                self.model_alias,
            )

            return metadata

        except Exception as exc:

            logger.exception(
                "Unable to retrieve champion model metadata | "
                "model_name={} | alias={}",
                self.model_name,
                self.model_alias,
            )

            raise LOSModelUnavailableError(
                "Unable to retrieve champion model metadata."
            ) from exc

    # =====================================================
    # MODEL SIGNATURE
    # =====================================================

    def get_model_signature(self) -> dict[str, Any]:
        """
        Retrieve input and output signature information.

        The signature is generated during model training
        and logged with the MLflow model artifact.
        """

        try:

            model = self.load_model()

            signature = model.metadata.signature

            if signature is None:

                logger.warning(
                    "MLflow model does not contain a signature."
                )

                return {
                    "has_signature": False,
                    "input_features": [],
                    "output_schema": None,
                }

            input_features = []

            if signature.inputs is not None:

                input_features = (
                    signature.inputs.input_names()
                )

            output_schema = None

            if signature.outputs is not None:

                output_schema = str(
                    signature.outputs
                )

            return {
                "has_signature": True,
                "input_features": input_features,
                "output_schema": output_schema,
            }

        except Exception as exc:

            logger.exception(
                "Unable to retrieve Hospital LOS model signature."
            )

            raise LOSImplementationError(
                "Unable to retrieve model signature."
            ) from exc

    # =====================================================
    # INPUT DATA VALIDATION
    # =====================================================

    def validate_input(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Perform basic input validation.

        Feature names and order are checked against the
        MLflow model signature when available.

        Missing values and categorical encoding are handled
        by the fitted preprocessing pipeline, provided the
        training pipeline supports them.
        """

        if not isinstance(input_data, pd.DataFrame):

            raise LOSInputError(
                "Input must be a pandas DataFrame."
            )

        if input_data.empty:

            raise LOSInputError(
                "Input DataFrame cannot be empty."
            )

        if input_data.columns.duplicated().any():

            raise LOSInputError(
                "Duplicate feature columns are not allowed."
            )

        if not all(
            isinstance(column, str) and column.strip()
            for column in input_data.columns
        ):

            raise LOSInputError(
                "All feature names must be non-empty strings."
            )

        model = self.load_model()

        signature = model.metadata.signature

        if signature is None or signature.inputs is None:

            logger.warning(
                "Model signature unavailable. "
                "Skipping signature-based feature validation."
            )

            return input_data

        expected_features = (
            signature.inputs.input_names()
        )

        supplied_features = list(
            input_data.columns
        )

        missing_features = (
            set(expected_features) - set(supplied_features)
        )

        unexpected_features = (
            set(supplied_features) - set(expected_features)
        )

        if missing_features:

            raise LOSInputError(
                "Missing required model features: "
                f"{sorted(missing_features)}"
            )

        if unexpected_features:

            raise LOSInputError(
                "Unexpected model features: "
                f"{sorted(unexpected_features)}"
            )

        # Align columns with the exact training signature order.
        input_data = input_data[expected_features].copy()

        # Normalize primitive dtypes to the registered MLflow signature.
        # Integer columns reject fractional values instead of truncating them.
        for field in signature.inputs.inputs:
            column = field.name
            field_type = str(field.type).lower()

            try:
                if field_type in ("long", "integer"):
                    values = pd.to_numeric(input_data[column], errors="raise")
                    if values.isna().any() or not np.isfinite(values).all():
                        raise LOSInputError(
                            f"Null or non-finite values found in feature: {column}"
                        )
                    if not np.equal(values, np.floor(values)).all():
                        raise LOSInputError(
                            f"Non-integer values found in feature: {column}"
                        )
                    input_data[column] = values.astype("int64")

                elif field_type in ("double", "float"):
                    values = pd.to_numeric(input_data[column], errors="raise")
                    if values.isna().any() or not np.isfinite(values).all():
                        raise LOSInputError(
                            f"Null or non-finite values found in feature: {column}"
                        )
                    input_data[column] = values.astype("float64")

                elif field_type in ("string", "str"):
                    if input_data[column].isna().any():
                        raise LOSInputError(
                            f"Null values found in feature: {column}"
                        )
                    input_data[column] = input_data[column].astype(str)

            except LOSInputError:
                raise
            except (TypeError, ValueError) as exc:
                raise LOSInputError(
                    f"Invalid value type for feature: {column}"
                ) from exc

        return input_data

    # =====================================================
    # PREDICTION OUTPUT VALIDATION
    # =====================================================

    @staticmethod
    def validate_predictions(
        predictions: Any,
        expected_count: int,
    ) -> np.ndarray:
        """
        Validate prediction output shape and values.
        """

        try:

            predictions = np.asarray(
                predictions,
                dtype=float,
            ).reshape(-1)

        except (TypeError, ValueError) as exc:

            raise LOSInferenceError(
                "Model returned invalid prediction values."
            ) from exc

        if len(predictions) != expected_count:

            raise LOSInferenceError(
                "Prediction count does not match input count."
            )

        if not np.isfinite(predictions).all():

            raise LOSInferenceError(
                "Model returned non-finite prediction values."
            )

        return predictions

    # =====================================================
    # SINGLE / BATCH PREDICTION
    # =====================================================

    def predict(
        self,
        input_data: pd.DataFrame,
        operation_type: str = "inference",
    ) -> np.ndarray:
        """
        Validate and execute inference, recording operational metrics in MLflow.

        Patient feature values and individual predictions are deliberately
        excluded from MLflow parameters, tags and metrics.
        """
        operation_type = operation_type.lower().strip()
        if operation_type not in {"single", "batch", "inference"}:
            raise LOSInputError(
                "operation_type must be single, batch, or inference."
            )

        started = time.perf_counter()

        try:
            input_data = self.validate_input(input_data)
            model = self.load_model()
            predictions = model.predict(input_data)
            predictions = self.validate_predictions(
                predictions=predictions,
                expected_count=len(input_data),
            )

            
            
            
            
            logger.success(
                    "Implementation: inference and MLflow logging completed | "
                    "records={}",
                    
                    len(predictions),
                    
                )

            return predictions

        except LOSImplementationError:
            raise
        except Exception as exc:
            logger.exception(
                "Implementation: Hospital LOS inference failed | operation={}",
                operation_type,
            )
            raise LOSInferenceError(
                "Hospital LOS prediction execution or MLflow logging failed."
            ) from exc

    # =====================================================
    # HEALTH CHECK
    # =====================================================

    def health_check(self) -> dict[str, Any]:
        """
        Verify that the promoted model can be loaded
        and its registry metadata can be retrieved.
        """

        try:

            self.load_model()

            metadata = self.get_model_metadata()

            signature = self.get_model_signature()

            result = {
                "status": "HEALTHY",
                **metadata,
                "signature_available": (
                    signature["has_signature"]
                ),
            }

            logger.success(
                "Hospital LOS implementation health check passed."
            )

            return result

        except Exception as exc:

            logger.exception(
                "Hospital LOS implementation health check failed."
            )

            raise LOSModelUnavailableError(
                "Hospital LOS model health check failed."
            ) from exc


# =========================================================
# SINGLETON IMPLEMENTATION INSTANCE
# =========================================================

los_implementation = HospitalLOSImplementation()