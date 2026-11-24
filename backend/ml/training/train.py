
"""
Hospital LOS - End-to-End ML Training Orchestrator

Workflow:
    1. Load dataset
    2. Validate dataset
    3. Split train / validation / test
    4. Train candidate models
    5. Compare candidate models
    6. Select best candidate
    7. Evaluate on holdout test set
    8. Apply quality gate
    9. Register approved model
   10. Promote model using champion alias

Important:
    - Validation set is used for candidate selection.
    - Test set is used only for final independent evaluation.
    - Model promotion requires a release approval reference.
"""

from __future__ import annotations

from typing import Optional

import mlflow

from app.config.settings import settings
logger = settings.get_logger()

from ml.data.load_dataset import (
    HospitalLOSDatasetLoader,
)

from ml.training.experiment_runner import (
    HospitalLOSExperimentRunner,
)

from ml.evaluation.evaluate import (
    HospitalLOSEvaluator,
)

from ml.evaluation.quality_gate import (
    HospitalLOSQualityGate,
)

from ml.registry.promote_model import (
    HospitalLOSModelRegistry,
)

from mlflow.tracking import MlflowClient
from datetime import datetime, timezone

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRACKING_URI = "http://localhost:5000"

EXPERIMENT_NAME = "hospital-los-regression"

REGISTERED_MODEL_NAME = "hospital-los-regressor"

DATASET_PATH = "data/hospital_length_of_stay.csv"

TARGET_COLUMN = "length_of_stay_days"

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.16

CHAMPION_ALIAS = "champion"

# ---------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------

def train_hospital_los_model(
    approval_reference: Optional[str] = None,
):
    """
    Execute the complete Hospital LOS model lifecycle.

    Args:
        approval_reference:
            Auditable release approval reference.
            If omitted, the approved model is registered
            but is not promoted to champion.

    Returns:
        Dictionary containing model selection, evaluation,
        quality gate, registration and promotion results.
    """

    logger.info(
        "Starting Hospital LOS end-to-end ML workflow."
    )

    mlflow.set_tracking_uri(TRACKING_URI)

    # =====================================================
    # STEP 1: LOAD AND VALIDATE DATASET
    # =====================================================

    logger.info(
        "STEP 1: Loading Hospital LOS dataset."
    )

    loader = HospitalLOSDatasetLoader(
        dataset_path=DATASET_PATH,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        drop_duplicates=True,
        log_dataset_to_mlflow=False,
    )

    dataset = loader.load_and_split()

    logger.success(
        "Dataset loaded and split | "
        "total_rows={} | features={} | "
        "train={} | validation={} | test={}",
        dataset.total_rows,
        dataset.total_features,
        dataset.train_rows,
        dataset.validation_rows,
        dataset.test_rows,
    )

    # =====================================================
    # STEP 2: TRAIN AND COMPARE CANDIDATE MODELS
    # =====================================================

    logger.info(
        "STEP 2: Training and comparing candidate models."
    )

    runner = HospitalLOSExperimentRunner(
        experiment_name=EXPERIMENT_NAME,
        tracking_uri=TRACKING_URI,
        random_state=RANDOM_STATE,
        log_models=True,
    )

    results = runner.run_experiments(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
    )

    if not results:
        raise RuntimeError(
            "No candidate model experiments were completed."
        )

    # =====================================================
    # STEP 3: DISPLAY MODEL COMPARISON
    # =====================================================

    logger.info(
        "STEP 3: Candidate model comparison."
    )

    for rank, candidate in enumerate(results, start=1):

        logger.info(
            "MODEL RANKING | "
            "rank={} | model={} | "
            "MAE={:.4f} | RMSE={:.4f} | R2={:.4f} | "
            "run_id={}",
            rank,
            candidate.model_name,
            candidate.validation_mae,
            candidate.validation_rmse,
            candidate.validation_r2,
            candidate.run_id,
        )

    # =====================================================
    # STEP 4: SELECT BEST CANDIDATE
    # =====================================================

    logger.info(
        "STEP 4: Selecting best candidate using validation MAE."
    )

    best_candidate = runner.get_best_candidate(results)

    logger.success(
        "Best candidate selected | "
        "model={} | validation_MAE={:.4f} | "
        "validation_RMSE={:.4f} | validation_R2={:.4f} | "
        "run_id={}",
        best_candidate.model_name,
        best_candidate.validation_mae,
        best_candidate.validation_rmse,
        best_candidate.validation_r2,
        best_candidate.run_id,
    )

    # =====================================================
    # STEP 5: FINAL HOLDOUT TEST EVALUATION
    # =====================================================

    logger.info(
        "STEP 5: Evaluating selected model on holdout test data."
    )

    evaluator = HospitalLOSEvaluator()

    with mlflow.start_run(
        run_name="hospital_los_final_test_evaluation"
    ) as evaluation_run:

        evaluation_result = evaluator.evaluate_model(
            model_uri=best_candidate.model_uri,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
        )

        test_metrics = evaluation_result["metrics"]

        logger.success(
            "Final test evaluation completed | "
            "MAE={:.4f} | RMSE={:.4f} | R2={:.4f} | "
            "evaluation_run_id={}",
            test_metrics["mean_absolute_error"],
            test_metrics["root_mean_squared_error"],
            test_metrics["r2_score"],
            evaluation_run.info.run_id,
        )

        # =================================================
        # STEP 6: APPLY QUALITY GATE
        # =================================================

        logger.info(
            "STEP 6: Applying quality gate to holdout metrics."
        )

        quality_gate = HospitalLOSQualityGate()

        gate_result = quality_gate.evaluate(
            metrics={
                "mean_absolute_error":
                    test_metrics["mean_absolute_error"],

                "root_mean_squared_error":
                    test_metrics["root_mean_squared_error"],

                "r2_score":
                    test_metrics["r2_score"],
            }
        )

        logger.info(
            "Quality gate completed | status={} | passed={}",
            gate_result.status,
            gate_result.passed,
        )

    # =====================================================
    # STEP 7: BLOCK REJECTED MODELS
    # =====================================================

    if not gate_result.passed:

        logger.error(
            "Model rejected by quality gate | reasons={}",
            gate_result.reasons,
        )

        raise RuntimeError(
            "Hospital LOS model failed final test quality gate. "
            "Registration and promotion are blocked."
        )

    logger.success(
        "Model passed final test quality gate."
    )

    # =====================================================
    # STEP 8: REGISTER APPROVED MODEL
    # =====================================================

    logger.info(
        "STEP 8: Registering approved model."
    )

    registry = HospitalLOSModelRegistry(
        tracking_uri=TRACKING_URI,
        model_name=REGISTERED_MODEL_NAME,
        champion_alias="champion",
    )

    registration = registry.register_model(
        model_uri=best_candidate.model_uri,
        run_id=best_candidate.run_id,
        description=(
            "Hospital LOS regression model. "
            "Selected using validation MAE and "
            "evaluated on an independent holdout test set."
        ),
        tags={
            "algorithm": best_candidate.model_name,
            "selection_metric": "validation_mae",
            "test_quality_gate": "passed",
            "test_mae": str(
                test_metrics["mean_absolute_error"]
            ),
            "test_rmse": str(
                test_metrics["root_mean_squared_error"]
            ),
            "test_r2": str(
                test_metrics["r2_score"]
            ),
        },
    )

    logger.success(
        "Model registration completed | "
        "model_name={} | version={} | status={}",
        registration.registered_model_name,
        registration.model_version,
        registration.status,
    )

    # =====================================================
    # STEP 9: PROMOTE APPROVED MODEL
    # =====================================================

    promotion = None
    promotion_verification = {
        "verified": False,
        "alias": CHAMPION_ALIAS,
        "model_version": str(registration.model_version),
    }

    # Normalize and validate the supplied reference.
    # This checks its format only; it does not authenticate an
    # approval in an external change-management system.
    if approval_reference is not None:
        if not isinstance(approval_reference, str):
            raise TypeError(
                "approval_reference must be a string or None."
            )

        approval_reference = approval_reference.strip()

        if not approval_reference:
            approval_reference = None
        elif len(approval_reference) > 256:
            raise ValueError(
                "approval_reference must not exceed 256 characters."
            )
        elif any(char in approval_reference for char in ("\n", "\r", "\t")):
            raise ValueError(
                "approval_reference must be a single-line reference."
            )
        elif approval_reference.lower() in {
            "none", "null", "na", "n/a", "pending", "tbd", "test"
        }:
            raise ValueError(
                "A placeholder is not a valid release approval reference."
            )

    registry_client = MlflowClient(tracking_uri=TRACKING_URI)
    registered_version = str(registration.model_version)

    if not gate_result.passed:
        raise RuntimeError(
            "Promotion blocked because the final holdout quality gate did not pass."
        )

    if approval_reference is None:
        registry_client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=registered_version,
            key="release_status",
            value="PENDING_APPROVAL",
        )
        registry_client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=registered_version,
            key="test_quality_gate",
            value="PASSED",
        )

        logger.warning(
            "Model registered but not promoted | "
            "model_name={} | version={} | "
            "reason=missing_release_approval_reference",
            REGISTERED_MODEL_NAME,
            registered_version,
        )

    else:
        logger.info(
            "STEP 9: Release reference supplied. "
            "Checking registered version before promotion | "
            "model_name={} | version={}",
            REGISTERED_MODEL_NAME,
            registered_version,
        )

        model_version = registry_client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=registered_version,
        )

        if str(model_version.status).upper() != "READY":
            raise RuntimeError(
                "Promotion blocked: registered model version is not READY "
                f"(current status={model_version.status})."
            )

        # Persist release evidence on the registered model version.
        # The reference is recorded, not independently verified here.
        version_tags = {
            "release_approval_reference": approval_reference,
            "release_approval_reference_status": "SUPPLIED_NOT_EXTERNALLY_VERIFIED",
            "test_quality_gate": "PASSED",
            "release_status": "PROMOTION_REQUESTED",
            "promotion_requested_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for tag_key, tag_value in version_tags.items():
            registry_client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=registered_version,
                key=tag_key,
                value=tag_value,
            )

        promotion = registry.promote_model(
            version=registration.model_version,
            quality_gate_passed=gate_result.passed,
            approval_reference=approval_reference,
            approval_note=(
                "Passed independent holdout test evaluation and configured "
                "quality gate. Release reference was supplied to the "
                "promotion workflow."
            ),
        )

        # Confirm that MLflow's champion alias points to this exact version.
        champion_version = registry_client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
        )

        if str(champion_version.version) != registered_version:
            raise RuntimeError(
                "Promotion verification failed: champion alias does not "
                f"point to expected version {registered_version}; "
                f"actual version is {champion_version.version}."
            )

        registry_client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=registered_version,
            key="release_status",
            value="PROMOTED",
        )
        registry_client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=registered_version,
            key="promotion_verified_at_utc",
            value=datetime.now(timezone.utc).isoformat(),
        )

        promotion_verification = {
            "verified": True,
            "alias": CHAMPION_ALIAS,
            "model_version": registered_version,
            "approval_reference": approval_reference,
            "approval_reference_status": "SUPPLIED_NOT_EXTERNALLY_VERIFIED",
        }

        logger.success(
            "Champion alias promotion verified | "
            "model_name={} | version={} | alias={} | approval_reference={}",
            REGISTERED_MODEL_NAME,
            registered_version,
            CHAMPION_ALIAS,
            approval_reference,
        )

    # =====================================================
    # STEP 10: RETURN WORKFLOW RESULTS
    # =====================================================

    workflow_result = {
        "status": (
            "PROMOTED"
            if promotion is not None
            else "REGISTERED_PENDING_APPROVAL"
        ),

        "candidate_models": results,

        "best_candidate": best_candidate,

        "evaluation": evaluation_result,

        "quality_gate": gate_result,

        "registration": registration,

        "promotion": promotion,
    }

    logger.success(
        "Hospital LOS ML workflow completed | status={}",
        workflow_result["status"],
    )

    return workflow_result


# ---------------------------------------------------------
# Application entry point
# ---------------------------------------------------------

def main():

    try:
        result = train_hospital_los_model(
            # Supply a real release approval reference
            # when promotion is authorized.
            approval_reference=None,
        )

        logger.success(
            "Training pipeline finished | status={}",
            result["status"],
        )

    except Exception:
        logger.exception(
            "Hospital LOS training pipeline failed."
        )
        raise


if __name__ == "__main__":
    main()