
"""
Hospital LOS - MLflow Model Registry and Promotion

Responsibilities:
    1. Validate an MLflow model candidate.
    2. Register the model in the MLflow Model Registry.
    3. Attach version metadata and approval information.
    4. Promote an approved model using the 'champion' alias.
    5. Preserve the previous champion version for rollback.

MLflow 3.x:
    Uses model aliases instead of deprecated model stages.

Expected model URI:
    runs:/<run_id>/hospital_los_model

Safety:
    Registration does not automatically imply approval.
    Promotion requires explicit quality-gate approval.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import mlflow
import mlflow.pyfunc

from mlflow.tracking import MlflowClient


from app.config.settings import settings
logger = settings.get_logger()

from mlflow import MlflowClient
from mlflow.entities import ViewType


@dataclass
class ModelRegistrationResult:
    """Result of an MLflow model registration operation."""

    registered_model_name: str
    model_version: str
    run_id: str
    source_uri: str
    status: str


@dataclass
class ModelPromotionResult:
    """Result of a model alias promotion operation."""

    registered_model_name: str
    promoted_version: str
    alias: str
    previous_champion_version: Optional[str]
    status: str
    execution_time_seconds: float


class HospitalLOSModelRegistry:
    """
    Manage Hospital LOS model registration and promotion.

    Example:
        registry = HospitalLOSModelRegistry(
            tracking_uri="http://localhost:5000",
            model_name="hospital-los-regressor",
        )

        registration = registry.register_model(
            model_uri=best_candidate.model_uri,
            run_id=best_candidate.run_id,
        )

        promotion = registry.promote_model(
            version=registration.model_version,
            quality_gate_passed=True,
        )
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        model_name: str = "hospital-los-regressor",
        champion_alias: str = "champion",
        client: Optional[MlflowClient] = None,
        await_registration_for: int = 300,
    ) -> None:

        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.champion_alias = champion_alias
        self.await_registration_for = await_registration_for

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = client or MlflowClient(
            tracking_uri=mlflow.get_tracking_uri()
        )

        self._validate_configuration()

        logger.info(
            "Hospital LOS Model Registry initialized | "
            "model_name={} | alias={} | tracking_uri={}",
            self.model_name,
            self.champion_alias,
            mlflow.get_tracking_uri(),
        )

    # -----------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate registry configuration."""

        if not self.model_name or not self.model_name.strip():
            raise ValueError("Model name cannot be empty.")

        if not self.champion_alias or not self.champion_alias.strip():
            raise ValueError("Champion alias cannot be empty.")

        if self.await_registration_for < 0:
            raise ValueError(
                "await_registration_for cannot be negative."
            )

    # -----------------------------------------------------
    # Model URI validation
    # -----------------------------------------------------

    

    def _extract_run_id(self, model_uri: str) -> str:
        """
        Resolve the source run ID from either:
        - MLflow 2.x runs:/<run_id>/<artifact_path>
        - MLflow 3.x models:/<model_id>
        """

        if not model_uri:
            raise ValueError("Model URI cannot be empty.")

        # MLflow 2.x compatibility
        if model_uri.startswith("runs:/"):
            parts = model_uri.split("/")

            if len(parts) < 3 or not parts[1]:
                raise ValueError(
                    f"Invalid MLflow run URI: {model_uri}"
                )

            return parts[1]

        # MLflow 3.x logged model URI
        if model_uri.startswith("models:/"):
            model_identifier = model_uri[len("models:/"):].strip("/")

            # A logged-model URI has one identifier.
            # A registered model version URI has two components.
            if not model_identifier or "/" in model_identifier:
                raise ValueError(
                    f"Expected models:/<model_id>, "
                    f"received: {model_uri}"
                )

            client = MlflowClient()

            logged_model = client.get_logged_model(
                model_identifier
            )

            source_run_id = logged_model.source_run_id

            if not source_run_id:
                raise ValueError(
                    f"No source run ID found for logged model: "
                    f"{model_identifier}"
                )

            return source_run_id

        raise ValueError(
            "Unsupported MLflow model URI. Expected "
            "runs:/<run_id>/<artifact_path> or "
            "models:/<model_id>. "
            f"Received: {model_uri}"
        )

    # -----------------------------------------------------
    # Source run validation
    # -----------------------------------------------------

    def _validate_source_run(
        self,
        run_id: str,
        model_uri: str,
    ) -> None:
        """Ensure the source run exists and completed successfully."""

        uri_run_id = self._extract_run_id(model_uri)

        if uri_run_id != run_id:
            raise ValueError(
                "The supplied run_id does not match the model URI."
            )

        run = self.client.get_run(run_id)

        if run.info.status != "FINISHED":
            raise ValueError(
                f"Source MLflow run is not FINISHED. "
                f"Current status: {run.info.status}"
            )

        logger.info(
            "Source MLflow run validated | run_id={} | status={}",
            run_id,
            run.info.status,
        )

    # -----------------------------------------------------
    # Model loading validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_model_loadable(model_uri: str) -> None:
        """Verify that the logged model can be loaded."""

        try:
            mlflow.pyfunc.load_model(model_uri)

            logger.success(
                "Model artifact successfully loaded | model_uri={}",
                model_uri,
            )

        except Exception:
            logger.exception(
                "Model artifact validation failed | model_uri={}",
                model_uri,
            )
            raise

    # -----------------------------------------------------
    # Register model
    # -----------------------------------------------------

    def register_model(
        self,
        model_uri: str,
        run_id: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> ModelRegistrationResult:
        """
        Register a logged model in the MLflow Model Registry.

        Registration does not assign the champion alias.
        """

        start_time = time.perf_counter()

        try:
            logger.info(
                "Starting model registration | "
                "model_name={} | run_id={}",
                self.model_name,
                run_id,
            )

            self._validate_source_run(
                run_id=run_id,
                model_uri=model_uri,
            )

            self._validate_model_loadable(model_uri)

            registered_version = mlflow.register_model(
                model_uri=model_uri,
                name=self.model_name,
                #name=self.registered_model_name
                await_registration_for=self.await_registration_for,
            )

            version = str(registered_version.version)

            # Confirm the registered version is accessible.
            model_version = self.client.get_model_version(
                name=self.model_name,
                version=version,
            )

            if model_version.status != "READY":
                raise RuntimeError(
                    "Model version is not READY. "
                    f"Current status: {model_version.status}"
                )

            if description:
                self.client.update_model_version(
                    name=self.model_name,
                    version=version,
                    description=description,
                )

            version_tags = {
                "project": "hospital-los",
                "problem_type": "regression",
                "target": "los_days",
                "source_run_id": run_id,
                "registration_status": "registered",
            }

            if tags:
                version_tags.update(tags)

            for key, value in version_tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=version,
                    key=key,
                    value=str(value),
                )

            execution_time = time.perf_counter() - start_time

            logger.success(
                "Model registered successfully | "
                "model_name={} | version={} | "
                "run_id={} | execution_time={:.2f}s",
                self.model_name,
                version,
                run_id,
                execution_time,
            )

            return ModelRegistrationResult(
                registered_model_name=self.model_name,
                model_version=version,
                run_id=run_id,
                source_uri=model_uri,
                status="REGISTERED",
            )

        except Exception:
            logger.exception(
                "Model registration failed | "
                "model_name={} | run_id={}",
                self.model_name,
                run_id,
            )
            raise

    # -----------------------------------------------------
    # Read current champion
    # -----------------------------------------------------

    def get_champion_version(self) -> Optional[str]:
        """Return the version currently assigned to champion."""

        try:
            model_version = self.client.get_model_version_by_alias(
                name=self.model_name,
                alias=self.champion_alias,
            )

            logger.info(
                "Current champion found | model_name={} | version={}",
                self.model_name,
                model_version.version,
            )

            return str(model_version.version)

        except Exception as exc:
            # A missing alias is a normal first-deployment condition.
            # Other registry errors must not be silently ignored.
            from mlflow.exceptions import MlflowException

            if isinstance(exc, MlflowException):
                error_code = getattr(
                    exc,
                    "error_code",
                    None,
                )

                if error_code == "RESOURCE_DOES_NOT_EXIST":
                    logger.info(
                        "No existing champion alias found | model_name={}",
                        self.model_name,
                    )
                    return None

            logger.exception(
                "Unable to retrieve champion alias | model_name={}",
                self.model_name,
            )
            raise

    # -----------------------------------------------------
    # Version validation
    # -----------------------------------------------------

    def _validate_registered_version(
        self,
        version: str,
    ):
        """Verify the registered model version exists and is ready."""

        if not str(version).isdigit():
            raise ValueError(
                f"Invalid model version: {version}"
            )

        model_version = self.client.get_model_version(
            name=self.model_name,
            version=str(version),
        )

        if model_version.status != "READY":
            raise RuntimeError(
                f"Model version {version} is not READY. "
                f"Current status: {model_version.status}"
            )

        return model_version

    # -----------------------------------------------------
    # Promote model
    # -----------------------------------------------------

    def promote_model(
        self,
        version: str,
        quality_gate_passed: bool,
        approval_reference: Optional[str] = None,
        approval_note: Optional[str] = None,
    ) -> ModelPromotionResult:
        """
        Assign the champion alias to an approved model version.

        Promotion is blocked unless quality_gate_passed=True.

        approval_reference should identify an auditable approval
        record, such as a deployment ticket or release approval ID.
        """

        start_time = time.perf_counter()
        previous_champion = None

        try:
            logger.info(
                "Starting model promotion | "
                "model_name={} | candidate_version={}",
                self.model_name,
                version,
            )

            if quality_gate_passed is not True:
                raise PermissionError(
                    "Model promotion blocked: quality gate approval "
                    "was not supplied."
                )

            model_version = self._validate_registered_version(
                version
            )

            if not approval_reference or not approval_reference.strip():
                raise ValueError(
                    "An approval_reference is required for promotion."
                )

            # Verify that the version has a source run.
            if not model_version.run_id:
                raise ValueError(
                    "Registered model version has no source run ID."
                )

            previous_champion = self.get_champion_version()

            if previous_champion == str(version):
                logger.info(
                    "Candidate is already the champion | version={}",
                    version,
                )

                return ModelPromotionResult(
                    registered_model_name=self.model_name,
                    promoted_version=str(version),
                    alias=self.champion_alias,
                    previous_champion_version=previous_champion,
                    status="ALREADY_CHAMPION",
                    execution_time_seconds=(
                        time.perf_counter() - start_time
                    ),
                )

            # Record approval metadata before assigning the alias.
            approval_tags = {
                "quality_gate_status": "passed",
                "approval_reference": approval_reference,
                "promotion_status": "approved",
            }

            if approval_note:
                approval_tags["approval_note"] = approval_note

            for key, value in approval_tags.items():
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=str(version),
                    key=key,
                    value=str(value),
                )

            # Assign champion alias.
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias=self.champion_alias,
                version=str(version),
            )

            # Verify the alias points to the intended version.
            current_champion = self.get_champion_version()

            if current_champion != str(version):
                raise RuntimeError(
                    "Champion alias verification failed. "
                    f"Expected={version}, actual={current_champion}"
                )

            execution_time = time.perf_counter() - start_time

            logger.success(
                "Model promotion completed | "
                "model_name={} | promoted_version={} | "
                "previous_champion={} | alias={} | "
                "execution_time={:.2f}s",
                self.model_name,
                version,
                previous_champion,
                self.champion_alias,
                execution_time,
            )

            return ModelPromotionResult(
                registered_model_name=self.model_name,
                promoted_version=str(version),
                alias=self.champion_alias,
                previous_champion_version=previous_champion,
                status="PROMOTED",
                execution_time_seconds=execution_time,
            )

        except Exception:
            logger.exception(
                "Model promotion failed | "
                "model_name={} | candidate_version={}",
                self.model_name,
                version,
            )

            # Restore the previous alias if the candidate was assigned
            # but subsequent verification failed.
            try:
                current_champion = self.get_champion_version()

                if (
                    previous_champion is not None
                    and current_champion == str(version)
                ):
                    self.client.set_registered_model_alias(
                        name=self.model_name,
                        alias=self.champion_alias,
                        version=previous_champion,
                    )

                    logger.warning(
                        "Previous champion alias restored | "
                        "version={}",
                        previous_champion,
                    )

            except Exception:
                logger.exception(
                    "Champion rollback attempt failed | model_name={}",
                    self.model_name,
                )

            raise

    # -----------------------------------------------------
    # Explicit rollback
    # -----------------------------------------------------

    def rollback_champion(
        self,
        version: str,
        approval_reference: str,
    ) -> ModelPromotionResult:
        """
        Explicitly assign champion alias to a previous version.

        Rollback requires an auditable approval reference.
        """

        if not approval_reference or not approval_reference.strip():
            raise ValueError(
                "An approval_reference is required for rollback."
            )

        logger.warning(
            "Starting champion rollback | model_name={} | target_version={}",
            self.model_name,
            version,
        )

        return self.promote_model(
            version=version,
            quality_gate_passed=True,
            approval_reference=approval_reference,
            approval_note="Explicit rollback to a previously registered version.",
        )