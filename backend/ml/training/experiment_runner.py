
"""
Hospital Length of Stay (LOS) - MLflow Experiment Runner

Responsibilities:
    1. Train multiple regression candidates.
    2. Evaluate candidates on validation data.
    3. Track parameters, metrics, and tags in MLflow.
    4. Log trained models with signatures and input examples.
    5. Return ranked experiment results.

Important:
    - Validation data is used for model selection.
    - The holdout test set must not be used here.
    - Synthetic dataset results must not be represented as
      production performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from app.config.settings import settings
logger = settings.get_logger()

from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from ml.preprocessing.preprocessing import (
    build_preprocessor,
)

# Linear Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    Lars,
    LassoLars
)

# Tree Models
from sklearn.tree import DecisionTreeRegressor

# Ensemble Models
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)

# Neighbors
from sklearn.neighbors import KNeighborsRegressor

# External Libraries
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor



@dataclass
class ExperimentResult:
    """Metadata and evaluation results for one model candidate."""

    model_name: str
    run_id: str
    model_uri: str

    validation_mae: float
    validation_rmse: float
    validation_r2: float

    execution_time_seconds: float
    parameters: dict[str, Any]

    status: str = "COMPLETED"


class HospitalLOSExperimentRunner:
    """
    Executes and tracks candidate regression experiments.

    Expected input:
        X_train: Training features.
        y_train: Training target (LOS days).
        X_val: Validation features.
        y_val: Validation target (LOS days).

    Returns:
        List of ExperimentResult objects sorted by validation MAE.
    """

    def __init__(
        self,
        experiment_name: str = "hospital-los-regression",
        tracking_uri: str | None = None,
        random_state: int = 42,
        log_models: bool = True,
    ) -> None:

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.random_state = random_state
        self.log_models = log_models

        self._initialize_mlflow()

        logger.info(
            "Hospital LOS Experiment Runner initialized | "
            "experiment_name={}",
            self.experiment_name,
        )

    # ---------------------------------------------------------
    # MLflow initialization
    # ---------------------------------------------------------

    def _initialize_mlflow(self) -> None:
        """Configure MLflow tracking and experiment."""

        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)

            logger.info(
                "MLflow configured | tracking_uri={} | experiment={}",
                mlflow.get_tracking_uri(),
                self.experiment_name,
            )

        except Exception:
            logger.exception(
                "Failed to initialize MLflow | experiment={}",
                self.experiment_name,
            )
            raise

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Validate training and validation datasets."""

        datasets = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }

        for name, dataset in datasets.items():
            if dataset is None or len(dataset) == 0:
                raise ValueError(
                    f"{name} must not be empty."
                )

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")

        if not isinstance(X_val, pd.DataFrame):
            raise TypeError("X_val must be a pandas DataFrame.")

        if not isinstance(y_train, (pd.Series, pd.DataFrame)):
            raise TypeError("y_train must be a pandas Series or DataFrame.")

        if not isinstance(y_val, (pd.Series, pd.DataFrame)):
            raise TypeError("y_val must be a pandas Series or DataFrame.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train have different row counts."
            )

        if len(X_val) != len(y_val):
            raise ValueError(
                "X_val and y_val have different row counts."
            )

        if list(X_train.columns) != list(X_val.columns):
            raise ValueError(
                "Training and validation feature columns must match "
                "in both name and order."
            )

        y_train_values = np.asarray(y_train).reshape(-1)
        y_val_values = np.asarray(y_val).reshape(-1)

        if not np.isfinite(y_train_values).all():
            raise ValueError("y_train contains non-finite values.")

        if not np.isfinite(y_val_values).all():
            raise ValueError("y_val contains non-finite values.")

        if (y_train_values < 0).any() or (y_val_values < 0).any():
            raise ValueError(
                "LOS target values cannot be negative."
            )

        logger.info(
            "Input validation successful | "
            "train_rows={} | validation_rows={} | features={}",
            len(X_train),
            len(X_val),
            len(X_train.columns),
        )

    # ---------------------------------------------------------
    # Candidate models
    # ---------------------------------------------------------

    
    def _build_candidates(self) -> dict:
        """
        Build all regression candidates using a consistent structure.

        Returns:
            {
                "model_name": (
                    estimator,
                    parameters
                )
            }

        Every candidate follows the same tuple structure.
        """

        candidates = {

            # -------------------------------------------------
            # Linear Models
            # -------------------------------------------------

            "linear_regression": LinearRegression(),

            "BayesianRidge": BayesianRidge(),

            "Lasso": Lasso(
                random_state=self.random_state,
                max_iter=10000,
            ),

            "ElasticNet": ElasticNet(
                random_state=self.random_state,
                max_iter=20000,
            ),

            "Huber": HuberRegressor(),

            "Lars": Lars(),

            "LassoLars": LassoLars(),

            # -------------------------------------------------
            # Tree Models
            # -------------------------------------------------

            "DecisionTree": DecisionTreeRegressor(
                random_state=self.random_state,
            ),

            # -------------------------------------------------
            # Neighbor Models
            # -------------------------------------------------

            "KNN": KNeighborsRegressor(
                n_neighbors=5,
            ),

            # -------------------------------------------------
            # Ensemble Models
            # -------------------------------------------------

            "AdaBoost": AdaBoostRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=3,
                    random_state=self.random_state,
                ),
                n_estimators=100,
                learning_rate=0.05,
                loss="linear",
                random_state=self.random_state,
            ),

            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            ),

            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=500,
                random_state=self.random_state,
                n_jobs=-1,
            ),

            "GradientBoosting": GradientBoostingRegressor(
                random_state=self.random_state,
                n_estimators=200,
                learning_rate=0.05,
            ),

            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=20,
                l2_regularization=1.0,
                random_state=self.random_state,
            ),

            "Bagging": BaggingRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            ),

            # -------------------------------------------------
            # External Gradient Boosting Models
            # -------------------------------------------------

            "XGBoost": XGBRegressor(
                objective="reg:squarederror",
                random_state=self.random_state,
                n_estimators=500,
                n_jobs=-1,
                verbosity=0,
            ),

            "LightGBM": LGBMRegressor(
                random_state=self.random_state,
                n_estimators=500,
                n_jobs=-1,
                verbose=-1,
            ),

            "CatBoost": CatBoostRegressor(
                verbose=False,
                random_state=self.random_state,
                iterations=500,
                thread_count=-1,
                allow_writing_files=False
            ),
        }

        # Normalize all candidate definitions.
        normalized_candidates = {}

        for model_name, estimator in candidates.items():

            if not hasattr(estimator, "fit"):
                raise TypeError(
                    f"Invalid estimator for model: {model_name}"
                )

            # Capture model configuration for MLflow.
            raw_parameters = estimator.get_params(deep=False)

            # Convert nested estimator objects to strings
            # so MLflow parameter logging remains consistent.
            parameters = {
                key: (
                    value
                    if isinstance(value, (str, int, float, bool))
                    else str(value)
                )
                for key, value in raw_parameters.items()
            }

            normalized_candidates[model_name] = (
                estimator,
                parameters,
            )

            logger.info(
                "Candidate registered | model={} | estimator={} | "
                "parameter_count={}",
                model_name,
                type(estimator).__name__,
                len(parameters),
            )

        logger.success(
            "Candidate model configuration completed | total_models={}",
            len(normalized_candidates),
        )

        return normalized_candidates


    # ---------------------------------------------------------
    # Pipeline construction
    # ---------------------------------------------------------

    @staticmethod
    def _build_pipeline(
        estimator: RegressorMixin,
        X_train: pd.DataFrame,
    ) -> Pipeline:
        """
        Build a complete preprocessing + regression pipeline.

        The preprocessor must be fitted only on training data.
        """

        preprocessor = build_preprocessor(X_train)

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

    # ---------------------------------------------------------
    # Evaluation metrics
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_metrics(
        y_true: pd.Series,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """Calculate validation regression metrics."""

        actual = np.asarray(y_true).reshape(-1)
        predicted = np.asarray(predictions).reshape(-1)

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted arrays have different lengths."
            )

        if not np.isfinite(predicted).all():
            raise ValueError(
                "Model generated non-finite predictions."
            )

        return {
            "validation_mae": float(
                mean_absolute_error(actual, predicted)
            ),
            "validation_mse": float(
                mean_squared_error(actual, predicted)
            ),
            
            "validation_rmse": float(
                np.sqrt(mean_squared_error(actual, predicted))
            ),
            "validation_r2": float(
                r2_score(actual, predicted)
            ),
            "validation_mape": float(
                mean_absolute_percentage_error(actual, predicted)
            ),

            
        }

    # ---------------------------------------------------------
    # Single candidate execution
    # ---------------------------------------------------------

    def _run_candidate(
        self,
        model_name: str,
        estimator: RegressorMixin,
        parameters: dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> ExperimentResult:

        start_time = time.perf_counter()

        logger.info(
            "Starting model experiment | model={}",
            model_name,
        )

        try:
            with mlflow.start_run(
                run_name=f"hospital_los_{model_name}"
            ) as run:

                run_id = run.info.run_id

                mlflow.set_tags(
                    {
                        "project": "hospital-los",
                        "problem_type": "regression",
                        "target": "los_days",
                        "model_name": model_name,
                        "dataset": "hospital_los",
                        "evaluation_stage": "validation",
                    }
                )

                mlflow.log_params(parameters)

                pipeline = self._build_pipeline(
                    estimator=estimator,
                    X_train=X_train,
                )

                logger.info(
                    "Training candidate model | model={} | run_id={}",
                    model_name,
                    run_id,
                )

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_val)

                metrics = self._calculate_metrics(
                    y_true=y_val,
                    predictions=predictions,
                )

                execution_time = (
                    time.perf_counter() - start_time
                )

                metrics["training_execution_time_seconds"] = float(
                    execution_time
                )

                mlflow.log_metrics(metrics)

                # Model signature and example are derived from
                # the actual feature schema.
                input_example = X_train.head(5).copy()

                signature = mlflow.models.infer_signature(
                    model_input=input_example,
                    model_output=pipeline.predict(input_example),
                )

                model_info = None

                if self.log_models:
                    model_info = mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        name="hospital_los_model",
                        serialization_format="cloudpickle",
                        signature=signature,
                        input_example=input_example,
                        pip_requirements=[
                            "mlflow",
                            "scikit-learn",
                            "pandas",
                            "numpy",
                        ],
                        skops_trusted_types=[
                            "numpy.dtype",
                            "sklearn.tree._tree.Tree"
                        ],

                    )

                model_uri = (
                    model_info.model_uri
                    if model_info is not None
                    else ""
                )

                mlflow.set_tag(
                    "experiment_status",
                    "completed",
                )

                result = ExperimentResult(
                    model_name=model_name,
                    run_id=run_id,
                    model_uri=model_uri,
                    validation_mae=metrics["validation_mae"],
                    validation_rmse=metrics["validation_rmse"],
                    validation_r2=metrics["validation_r2"],
                    execution_time_seconds=execution_time,
                    parameters=parameters,
                )

                logger.success(
                    "Experiment completed | model={} | "
                    "MAE={:.4f} | RMSE={:.4f} | R2={:.4f} | "
                    "execution_time={:.2f}s | run_id={}",
                    model_name,
                    result.validation_mae,
                    result.validation_rmse,
                    result.validation_r2,
                    execution_time,
                    run_id,
                )

                return result

        except Exception:
            logger.exception(
                "Model experiment failed | model={}",
                model_name,
            )
            raise

    # ---------------------------------------------------------
    # Execute all candidates
    # ---------------------------------------------------------

    
    def run_experiments(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> list[ExperimentResult]:

        """
        Execute all candidate models and rank by validation MAE.
        """

        self._validate_inputs(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        candidates = self._build_candidates()

        results: list[ExperimentResult] = []

        logger.info(
            "Starting LOS model comparison | candidates={}",
            list(candidates.keys()),
        )

        for model_name, candidate in candidates.items():

            # Validate candidate structure before unpacking.
            if (
                not isinstance(candidate, tuple)
                or len(candidate) != 2
            ):
                raise TypeError(
                    f"Invalid candidate structure: {model_name}. "
                    "Expected (estimator, parameters)."
                )

            estimator, parameters = candidate

            logger.info(
                "Executing candidate | model={} | estimator={}",
                model_name,
                type(estimator).__name__,
            )

            result = self._run_candidate(
                model_name=model_name,
                estimator=estimator,
                parameters=parameters,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            results.append(result)

        if not results:
            raise RuntimeError(
                "No candidate models completed successfully."
            )

        # Sort candidates using validation MAE.
        results.sort(
            key=lambda result: result.validation_mae
        )

        logger.success(
            "All model experiments completed | total_models={}",
            len(results),
        )

        # Display model comparison.
        for rank, result in enumerate(results, start=1):

            logger.info(
                "MODEL RANKING | rank={} | model={} | "
                "MAE={:.4f} | RMSE={:.4f} | R2={:.4f} | "
                "run_id={}",
                rank,
                result.model_name,
                result.validation_mae,
                result.validation_rmse,
                result.validation_r2,
                result.run_id,
            )

        return results


    # ---------------------------------------------------------
    # Best candidate
    # ---------------------------------------------------------

    @staticmethod
    def get_best_candidate(
        results: list[ExperimentResult],
    ) -> ExperimentResult:
        """Return the candidate with the lowest validation MAE."""

        if not results:
            raise ValueError(
                "No experiment results available."
            )

        best = min(
            results,
            key=lambda result: result.validation_mae,
        )

        logger.success(
            "Best candidate selected | model={} | "
            "validation_mae={:.4f} | run_id={}",
            best.model_name,
            best.validation_mae,
            best.run_id,
        )

        return best