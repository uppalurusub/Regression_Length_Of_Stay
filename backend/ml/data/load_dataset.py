
"""
Hospital LOS - Dataset Loading Module

Responsibilities:
    1. Load the hospital LOS dataset.
    2. Validate the dataset schema.
    3. Check missing values, duplicates and target validity.
    4. Separate features and target.
    5. Split data into training, validation and test sets.
    6. Optionally log dataset metadata to MLflow.

Target:
    los_days

Important:
    This module does not perform model-specific preprocessing.
    Imputation, encoding and scaling belong in the ML pipeline.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from app.config.settings import settings
logger = settings.get_logger()

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

DEFAULT_TARGET_COLUMN = "length_of_stay_days"

DEFAULT_REQUIRED_FEATURES = [
    "age",
    "gender",
    "marital_status",
    "primary_diagnosis",
    "comorbidity_count",
    "blood_pressure",
    "heart_rate",
    "oxygen_saturation",
    "temperature",
    "hemoglobin",
    "glucose",
    "creatinine",
    "admission_type",
    "insurance_type",
    "previous_admissions",
    "department"

]


# ---------------------------------------------------------
# Dataset result
# ---------------------------------------------------------

@dataclass
class DatasetSplit:
    """Container for dataset splits and metadata."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    dataset_path: str
    dataset_hash: str

    total_rows: int
    total_features: int

    train_rows: int
    validation_rows: int
    test_rows: int

    target_column: str


# ---------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------

class HospitalLOSDatasetLoader:
    """
    Load, validate and split hospital LOS datasets.

    Example:
        loader = HospitalLOSDatasetLoader(
            dataset_path="data/hospital_los.csv"
        )

        dataset = loader.load_and_split()

        X_train = dataset.X_train
        y_train = dataset.y_train
    """

    def __init__(
        self,
        dataset_path: str | Path,
        target_column: str = DEFAULT_TARGET_COLUMN,
        required_features: Optional[list[str]] = None,
        test_size: float = 0.20,
        validation_size: float = 0.20,
        random_state: int = 42,
        drop_duplicates: bool = True,
        log_dataset_to_mlflow: bool = False,
    ) -> None:

        self.dataset_path = Path(dataset_path)
        self.target_column = target_column

        self.required_features = (
            required_features
            if required_features is not None
            else DEFAULT_REQUIRED_FEATURES.copy()
        )

        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.drop_duplicates = drop_duplicates
        self.log_dataset_to_mlflow = log_dataset_to_mlflow

        self._validate_configuration()

        logger.info(
            "Hospital LOS Dataset Loader initialized | "
            "dataset_path={} | target={} | random_state={}",
            self.dataset_path,
            self.target_column,
            self.random_state,
        )

    # -----------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------

    def _validate_configuration(self) -> None:
        """Validate dataset loader configuration."""

        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size must be between 0 and 1."
            )

        if not 0 < self.validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if self.test_size + self.validation_size >= 1:
            raise ValueError(
                "test_size + validation_size must be less than 1."
            )

        if not self.target_column:
            raise ValueError("Target column cannot be empty.")

        if len(set(self.required_features)) != len(
            self.required_features
        ):
            raise ValueError(
                "required_features contains duplicate column names."
            )

        if self.target_column in self.required_features:
            raise ValueError(
                "Target column must not be included in required_features."
            )

    # -----------------------------------------------------
    # File validation
    # -----------------------------------------------------

    def _validate_file(self) -> None:
        """Check whether the dataset file exists."""

        if not self.dataset_path.exists():
            logger.error(
                "Dataset file not found | path={}",
                self.dataset_path,
            )

            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}"
            )

        if not self.dataset_path.is_file():
            raise ValueError(
                f"Dataset path is not a file: {self.dataset_path}"
            )

        if self.dataset_path.suffix.lower() != ".csv":
            raise ValueError(
                "Only CSV datasets are supported by this loader."
            )

        logger.info(
            "Dataset file validated | path={}",
            self.dataset_path,
        )

    # -----------------------------------------------------
    # Dataset checksum
    # -----------------------------------------------------

    def _calculate_dataset_hash(self) -> str:
        """Generate SHA-256 checksum for dataset reproducibility."""

        sha256 = hashlib.sha256()

        with self.dataset_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    def load_dataset(self) -> pd.DataFrame:
        """Load the CSV dataset into a Pandas DataFrame."""

        start_time = time.perf_counter()

        try:
            self._validate_file()

            logger.info(
                "Loading hospital LOS dataset | path={}",
                self.dataset_path,
            )

            dataframe = pd.read_csv(
                self.dataset_path,
                low_memory=False,
            )

            if dataframe.empty:
                raise ValueError(
                    "Dataset contains no records."
                )

            if dataframe.columns.duplicated().any():
                duplicated_columns = dataframe.columns[
                    dataframe.columns.duplicated()
                ].tolist()

                raise ValueError(
                    f"Duplicate column names: {duplicated_columns}"
                )

            dataframe.columns = dataframe.columns.str.strip()

            if dataframe.columns.duplicated().any():
                raise ValueError(
                    "Column names became duplicated after whitespace cleanup."
                )

            if dataframe.columns.isna().any():
                raise ValueError(
                    "Dataset contains missing column names."
                )

            if any(not column for column in dataframe.columns):
                raise ValueError(
                    "Dataset contains empty column names."
                )

            execution_time = time.perf_counter() - start_time

            logger.success(
                "Dataset loaded successfully | "
                "rows={} | columns={} | execution_time={:.3f}s",
                dataframe.shape[0],
                dataframe.shape[1],
                execution_time,
            )

            return dataframe

        except Exception:
            logger.exception(
                "Dataset loading failed | path={}",
                self.dataset_path,
            )
            raise

    # -----------------------------------------------------
    # Schema validation
    # -----------------------------------------------------

    def validate_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate required feature and target columns."""

        expected_columns = set(
            self.required_features + [self.target_column]
        )

        actual_columns = set(dataframe.columns)

        missing_columns = expected_columns - actual_columns

        if missing_columns:
            raise ValueError(
                "Dataset schema validation failed. "
                f"Missing columns: {sorted(missing_columns)}"
            )

        logger.info(
            "Dataset schema validated | required_features={} | target={}",
            len(self.required_features),
            self.target_column,
        )

    # -----------------------------------------------------
    # Data quality validation
    # -----------------------------------------------------

    def validate_data_quality(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate missing values, target values and duplicates.

        Feature missing values are retained for later pipeline
        imputation. Target missing values are not permitted.
        """

        try:
            logger.info("Starting dataset quality validation.")

            dataframe = dataframe.copy()

            # Duplicate records
            duplicate_count = int(
                dataframe.duplicated().sum()
            )

            if duplicate_count > 0:
                logger.warning(
                    "Duplicate records detected | count={}",
                    duplicate_count,
                )

                if self.drop_duplicates:
                    dataframe = dataframe.drop_duplicates()

                    logger.info(
                        "Duplicate records removed | remaining_rows={}",
                        len(dataframe),
                    )

            if dataframe.empty:
                raise ValueError(
                    "Dataset is empty after duplicate handling."
                )

            # Target validation
            target = pd.to_numeric(
                dataframe[self.target_column],
                errors="coerce",
            )

            invalid_target_count = int(target.isna().sum())

            if invalid_target_count > 0:
                raise ValueError(
                    f"Target column contains {invalid_target_count} "
                    "missing or non-numeric values."
                )

            target_values = target.to_numpy(dtype=float)

            if not np.isfinite(target_values).all():
                raise ValueError(
                    "Target column contains non-finite values."
                )

            if (target_values < 0).any():
                raise ValueError(
                    "LOS target cannot contain negative values."
                )

            dataframe[self.target_column] = target

            # Feature missing values
            feature_missing = (
                dataframe[self.required_features]
                .isnull()
                .sum()
            )

            missing_summary = feature_missing[
                feature_missing > 0
            ].to_dict()

            if missing_summary:
                logger.warning(
                    "Missing feature values detected | details={}",
                    missing_summary,
                )

                logger.info(
                    "Missing feature values will be handled "
                    "by the preprocessing pipeline."
                )

            # Infinite numeric feature values
            numeric_features = dataframe[
                self.required_features
            ].select_dtypes(include=[np.number])

            if not numeric_features.empty:
                infinite_count = int(
                    np.isinf(
                        numeric_features.to_numpy(dtype=float)
                    ).sum()
                )

                if infinite_count:
                    raise ValueError(
                        f"Numeric features contain {infinite_count} "
                        "infinite values."
                    )

            logger.success(
                "Dataset quality validation completed | "
                "rows={} | duplicate_rows={} | missing_feature_columns={}",
                len(dataframe),
                duplicate_count,
                len(missing_summary),
            )

            return dataframe.reset_index(drop=True)

        except Exception:
            logger.exception(
                "Dataset quality validation failed."
            )
            raise

    # -----------------------------------------------------
    # Feature and target separation
    # -----------------------------------------------------

    def split_features_target(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate model features from the LOS target."""

        try:
            X = dataframe[
                self.required_features
            ].copy()

            y = dataframe[
                self.target_column
            ].copy()

            y.name = self.target_column

            logger.info(
                "Features and target separated | "
                "features={} | records={}",
                X.shape[1],
                X.shape[0],
            )

            return X, y

        except Exception:
            logger.exception(
                "Feature and target separation failed."
            )
            raise

    # -----------------------------------------------------
    # Train / validation / test split
    # -----------------------------------------------------

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Split data into training, validation and test sets.

        Example defaults:
            Training:   64%
            Validation: 16%
            Test:       20%
        """

        try:
            logger.info(
                "Splitting dataset | train={} | validation={} | test={}",
                1 - self.test_size - self.validation_size,
                self.validation_size,
                self.test_size,
            )

            X_train_val, X_test, y_train_val, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            )

            # Convert validation size to a fraction of the
            # remaining training/validation partition.
            relative_validation_size = (
                self.validation_size
                / (1 - self.test_size)
            )

            X_train, X_val, y_train, y_val = (
                train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=relative_validation_size,
                    random_state=self.random_state,
                )
            )

            logger.success(
                "Dataset splitting completed | "
                "train={} | validation={} | test={}",
                len(X_train),
                len(X_val),
                len(X_test),
            )

            return (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            )

        except Exception:
            logger.exception(
                "Dataset splitting failed."
            )
            raise

    # -----------------------------------------------------
    # Optional MLflow dataset logging
    # -----------------------------------------------------

    def log_dataset_metadata(
        self,
        dataframe: pd.DataFrame,
        dataset_hash: str,
    ) -> None:
        """
        Log dataset metadata to the active MLflow run.

        Dataset values and patient-level records are not
        logged as MLflow parameters or tags.
        """

        if not self.log_dataset_to_mlflow:
            return

        if mlflow.active_run() is None:
            raise RuntimeError(
                "Dataset metadata logging requires an active MLflow run."
            )

        mlflow.set_tags(
            {
                "dataset_name": self.dataset_path.name,
                "dataset_sha256": dataset_hash,
                "dataset_rows": str(len(dataframe)),
                "dataset_columns": str(len(dataframe.columns)),
                "target_column": self.target_column,
            }
        )

        logger.info(
            "Dataset metadata logged to MLflow | "
            "dataset_hash={}",
            dataset_hash,
        )

    # -----------------------------------------------------
    # Complete loading workflow
    # -----------------------------------------------------

    def load_and_split(self) -> DatasetSplit:
        """Execute the complete dataset loading workflow."""

        start_time = time.perf_counter()

        try:
            dataframe = self.load_dataset()

            self.validate_schema(dataframe)

            dataframe = self.validate_data_quality(
                dataframe
            )

            dataset_hash = self._calculate_dataset_hash()

            self.log_dataset_metadata(
                dataframe=dataframe,
                dataset_hash=dataset_hash,
            )

            X, y = self.split_features_target(
                dataframe
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = self.split_dataset(X, y)

            result = DatasetSplit(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                dataset_path=str(self.dataset_path),
                dataset_hash=dataset_hash,
                total_rows=len(dataframe),
                total_features=X.shape[1],
                train_rows=len(X_train),
                validation_rows=len(X_val),
                test_rows=len(X_test),
                target_column=self.target_column,
            )

            execution_time = time.perf_counter() - start_time

            logger.success(
                "Hospital LOS dataset workflow completed | "
                "total_rows={} | features={} | "
                "train={} | validation={} | test={} | "
                "execution_time={:.3f}s",
                result.total_rows,
                result.total_features,
                result.train_rows,
                result.validation_rows,
                result.test_rows,
                execution_time,
            )

            return result

        except Exception:
            logger.exception(
                "Hospital LOS dataset workflow failed."
            )
            raise


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_hospital_los_dataset(
    #dataset_path: str = "data/hospital_los.csv",
    dataset_path: str = "data/hospital_length_of_stay.csv",
    random_state: int = 42,
) -> DatasetSplit:
    """Convenience function for the training pipeline."""

    loader = HospitalLOSDatasetLoader(
        dataset_path=dataset_path,
        random_state=random_state,
    )

    return loader.load_and_split()
