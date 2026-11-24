
import time
from dataclasses import dataclass, field
from typing import Optional

import mlflow

from app.config.settings import settings
logger = settings.get_logger()



@dataclass
class QualityGateResult:
    """Represents the model quality-gate decision."""

    passed: bool
    status: str
    reasons: list[str] = field(default_factory=list)
    evaluated_metrics: dict[str, float] = field(
        default_factory=dict
    )
    execution_time_seconds: float = 0.0


class HospitalLOSQualityGate:
    """
    Quality gate for Hospital Length of Stay regression.

    Responsibilities:
    - Validate regression evaluation metrics.
    - Enforce business-defined performance thresholds.
    - Compare candidate against champion when supplied.
    - Record decisions in MLflow.
    - Log execution details using Loguru.
    """

    def __init__(
        self,
        max_mae_days: Optional[float] = None,
        max_rmse_days: Optional[float] = None,
        min_r2: float = 0.0,
        max_mae_degradation: float = 0.0,
    ):

        self.max_mae_days = (
            max_mae_days
            if max_mae_days is not None
            else settings.min_acceptable_mae_days
        )

        self.max_rmse_days = max_rmse_days
        self.min_r2 = min_r2
        self.max_mae_degradation = max_mae_degradation

        logger.info(
            "Hospital LOS Quality Gate initialized | "
            "max_mae_days={} | max_rmse_days={} | min_r2={}",
            self.max_mae_days,
            self.max_rmse_days,
            self.min_r2,
        )

    @staticmethod
    def _validate_metric(
        metrics: dict,
        metric_name: str,
        failures: list[str],
    ) -> Optional[float]:

        value = metrics.get(metric_name)

        if value is None:
            failures.append(
                f"Required metric missing: {metric_name}"
            )
            return None

        try:
            numeric_value = float(value)

            if not (
                numeric_value == numeric_value
                and abs(numeric_value) != float("inf")
            ):
                raise ValueError("Metric must be finite")

            return numeric_value

        except (TypeError, ValueError):
            failures.append(
                f"Invalid metric value: {metric_name}"
            )
            return None

    def evaluate(
        self,
        metrics: dict[str, float],
        champion_metrics: Optional[dict[str, float]] = None,
        log_to_mlflow: bool = True,
    ) -> QualityGateResult:

        start_time = time.perf_counter()

        logger.info(
            "Starting Hospital LOS model quality validation"
        )

        failures = []
        evaluated_metrics = {}

        try:

            if not metrics:
                raise ValueError(
                    "Evaluation metrics cannot be empty"
                )

            # 1. Validate MAE
            mae = self._validate_metric(
                metrics,
                "mean_absolute_error",
                failures,
            )

            # 2. Validate RMSE
            rmse = self._validate_metric(
                metrics,
                "root_mean_squared_error",
                failures,
            )

            # 3. Validate R2
            r2 = self._validate_metric(
                metrics,
                "r2_score",
                failures,
            )

            evaluated_metrics = {
                key: value
                for key, value in {
                    "mean_absolute_error": mae,
                    "root_mean_squared_error": rmse,
                    "r2_score": r2,
                }.items()
                if value is not None
            }

            # 4. MAE quality validation
            if mae is not None and mae > self.max_mae_days:

                failures.append(
                    f"MAE {mae:.4f} days exceeds "
                    f"maximum allowed {self.max_mae_days:.4f} days"
                )

            # 5. RMSE quality validation
            if (
                self.max_rmse_days is not None
                and rmse is not None
                and rmse > self.max_rmse_days
            ):

                failures.append(
                    f"RMSE {rmse:.4f} days exceeds "
                    f"maximum allowed {self.max_rmse_days:.4f} days"
                )

            # 6. R2 quality validation
            if r2 is not None and r2 < self.min_r2:

                failures.append(
                    f"R2 {r2:.4f} is below "
                    f"minimum required {self.min_r2:.4f}"
                )

            # 7. Compare against existing champion
            if champion_metrics is not None:

                champion_mae = self._validate_metric(
                    champion_metrics,
                    "mean_absolute_error",
                    failures,
                )

                if mae is not None and champion_mae is not None:

                    allowed_mae = (
                        champion_mae
                        * (1 + self.max_mae_degradation)
                    )

                    if mae > allowed_mae:

                        failures.append(
                            f"Candidate MAE {mae:.4f} days "
                            f"is worse than allowed champion "
                            f"MAE threshold {allowed_mae:.4f} days"
                        )

                    logger.info(
                        "Champion comparison | "
                        "candidate_mae={} | champion_mae={} | "
                        "allowed_mae={}",
                        mae,
                        champion_mae,
                        allowed_mae,
                    )

            # 8. Generate decision
            passed = len(failures) == 0

            status = (
                "APPROVED"
                if passed
                else "REJECTED"
            )

            execution_time = (
                time.perf_counter() - start_time
            )

            result = QualityGateResult(
                passed=passed,
                status=status,
                reasons=failures,
                evaluated_metrics=evaluated_metrics,
                execution_time_seconds=round(
                    execution_time,
                    4,
                ),
            )

            # 9. Log quality decision to MLflow
            if log_to_mlflow:

                active_run = mlflow.active_run()

                if active_run is None:
                    raise RuntimeError(
                        "An active MLflow run is required "
                        "to log quality-gate results"
                    )

                mlflow.log_metrics({
                    f"quality_gate_{key}": value
                    for key, value in evaluated_metrics.items()
                })

                mlflow.log_metric(
                    "quality_gate_execution_seconds",
                    execution_time,
                )

                mlflow.set_tags({
                    "quality_gate_status": status,
                    "quality_gate_passed": str(passed).lower(),
                    "quality_gate_failure_count": str(
                        len(failures)
                    ),
                })

            # 10. Structured logging
            if passed:

                logger.success(
                    "Hospital LOS Quality Gate PASSED | "
                    "MAE={} | RMSE={} | R2={}",
                    mae,
                    rmse,
                    r2,
                )

            else:

                logger.warning(
                    "Hospital LOS Quality Gate REJECTED | "
                    "failures={}",
                    failures,
                )

            return result

        except Exception:

            logger.exception(
                "Hospital LOS Quality Gate execution failed"
            )

            raise


# Convenience function for training pipeline
def evaluate_quality_gate(
    metrics: dict[str, float],
    champion_metrics: Optional[dict[str, float]] = None,
    log_to_mlflow: bool = True,
) -> QualityGateResult:

    quality_gate = HospitalLOSQualityGate()

    return quality_gate.evaluate(
        metrics=metrics,
        champion_metrics=champion_metrics,
        log_to_mlflow=log_to_mlflow,
    )
