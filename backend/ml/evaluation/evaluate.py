
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config.settings import settings
logger = settings.get_logger()

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from app.config.settings import settings
artifact_dir = settings.artifact_dir
evaluation_dir = artifact_dir / "evaluation"


class HospitalLOSEvaluator:
    """
    Evaluates Hospital Length of Stay regression models
    using MLflow 3.x and Loguru.

    Responsibilities:
    - Load logged MLflow model
    - Generate predictions
    - Calculate regression metrics
    - Execute MLflow model evaluation
    - Log metrics and artifacts
    - Generate evaluation reports
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        #artifact_dir: str = "artifacts/evaluation",
        artifact_dir: str = evaluation_dir
    ):
        self.tracking_uri = (
            tracking_uri or settings.mlflow_tracking_uri
        )

        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        mlflow.set_tracking_uri(self.tracking_uri)

        logger.info(
            "Hospital LOS evaluator initialized | "
            "tracking_uri={}",
            self.tracking_uri,
        )

    def calculate_metrics(
        self,
        actual: pd.Series,
        predicted: Any,
    ) -> dict[str, float]:

        predicted = pd.Series(predicted)

        metrics = {
            "mean_absolute_error": float(
                mean_absolute_error(actual, predicted)
            ),
            "mean_squared_error": float(
                mean_squared_error(actual, predicted)
            ),
            "root_mean_squared_error": float(
                mean_squared_error(actual, predicted) ** 0.5
            ),
            "r2_score": float(
                r2_score(actual, predicted)
            ),
        }

        return metrics

    def generate_residual_plot(
        self,
        actual: pd.Series,
        predicted: Any,
        output_path: Path,
    ) -> None:

        residuals = actual.to_numpy() - predicted

        plt.figure(figsize=(9, 5))

        plt.scatter(
            predicted,
            residuals,
            alpha=0.6,
        )

        plt.axhline(
            y=0,
            linestyle="--",
        )

        plt.xlabel("Predicted LOS (Days)")
        plt.ylabel("Residuals (Days)")
        plt.title("Hospital LOS Residual Analysis")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(
            "Residual plot generated | path={}",
            output_path,
        )

    def generate_actual_vs_predicted_plot(
        self,
        actual: pd.Series,
        predicted: Any,
        output_path: Path,
    ) -> None:

        plt.figure(figsize=(7, 7))

        plt.scatter(
            actual,
            predicted,
            alpha=0.6,
        )

        min_value = min(
            actual.min(),
            min(predicted),
        )

        max_value = max(
            actual.max(),
            max(predicted),
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--",
        )

        plt.xlabel("Actual LOS (Days)")
        plt.ylabel("Predicted LOS (Days)")
        plt.title("Actual vs Predicted Hospital LOS")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(
            "Actual vs predicted plot generated | path={}",
            output_path,
        )

    def evaluate_model(
        self,
        model_uri: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        run_id: str | None = None,
    ) -> dict:

        start_time = time.perf_counter()

        logger.info(
            "Starting Hospital LOS model evaluation | "
            "model_uri={} | test_rows={}",
            model_uri,
            len(X_test),
        )

        try:

            # 1. Validate evaluation data
            if X_test.empty:
                raise ValueError(
                    "Evaluation feature dataset is empty"
                )

            if len(X_test) != len(y_test):
                raise ValueError(
                    "X_test and y_test row counts do not match"
                )

            if y_test.isna().any():
                raise ValueError(
                    "Target contains missing values"
                )

            # 2. Load MLflow model
            model = mlflow.pyfunc.load_model(model_uri)

            logger.info(
                "MLflow model loaded successfully | model_uri={}",
                model_uri,
            )

            # 3. Generate predictions
            predictions = model.predict(X_test)

            # 4. Calculate regression metrics
            metrics = self.calculate_metrics(
                y_test,
                predictions,
            )

            logger.info(
                "Regression evaluation metrics | "
                "MAE={} | RMSE={} | R2={}",
                metrics["mean_absolute_error"],
                metrics["root_mean_squared_error"],
                metrics["r2_score"],
            )

            # 5. Prepare evaluation dataset
            evaluation_df = X_test.copy()

            evaluation_df["length_of_stay_days"] = y_test.to_numpy()

            # 6. Execute MLflow model evaluation
            mlflow_evaluation = mlflow.models.evaluate(
                model=model_uri,
                data=evaluation_df,
                targets="length_of_stay_days",
                model_type="regressor",
                evaluators=["default"],
                evaluator_config={
                    "default": {
                        "log_model_explainability": False
                    }
                }
            )

            logger.info(
                "MLflow model evaluation completed"
            )

            # 7. Generate evaluation artifacts
            residual_path = (
                self.artifact_dir / "los_residual_plot.png"
            )

            actual_predicted_path = (
                self.artifact_dir
                / "los_actual_vs_predicted.png"
            )

            self.generate_residual_plot(
                y_test,
                predictions,
                residual_path,
            )

            self.generate_actual_vs_predicted_plot(
                y_test,
                predictions,
                actual_predicted_path,
            )

            # 8. Log evaluation metrics and artifacts
            metrics_to_log = {
                f"evaluation_{key}": value
                for key, value in metrics.items()
            }

            metrics_to_log[
                "evaluation_execution_seconds"
            ] = time.perf_counter() - start_time

            mlflow.log_metrics(metrics_to_log)

            mlflow.log_artifact(
                str(residual_path)
            )

            mlflow.log_artifact(
                str(actual_predicted_path)
            )

            # 9. Log evaluation metadata
            mlflow.set_tags({
                "evaluation_status": "completed",
                "evaluation_type": "regression",
                "target_variable": "los_days",
                "evaluation_framework": "mlflow",
            })

            execution_time = (
                time.perf_counter() - start_time
            )

            logger.success(
                "Hospital LOS evaluation completed | "
                "MAE={} days | RMSE={} days | "
                "R2={} | execution_time={} seconds",
                round(metrics["mean_absolute_error"], 4),
                round(metrics["root_mean_squared_error"], 4),
                round(metrics["r2_score"], 4),
                round(execution_time, 4),
            )

            return {
                "status": "success",
                "model_uri": model_uri,
                "metrics": metrics,
                "mlflow_metrics": dict(
                    mlflow_evaluation.metrics
                ),
                "execution_time_seconds": round(
                    execution_time,
                    4,
                ),
                "artifacts": [
                    str(residual_path),
                    str(actual_predicted_path),
                ],
            }

        except Exception:

            logger.exception(
                "Hospital LOS model evaluation failed | "
                "model_uri={}",
                model_uri,
            )

            raise