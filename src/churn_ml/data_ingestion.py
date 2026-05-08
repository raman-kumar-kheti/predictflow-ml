"""Dataset ingestion, schema validation, and cleaning for customer churn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from churn_ml.config import logger, settings

REQUIRED_COLUMNS: tuple[str, ...] = (
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for col in REQUIRED_COLUMNS if col not in ("customerID", "Churn")
)
NUMERIC_COLUMNS: tuple[str, ...] = ("tenure", "MonthlyCharges", "TotalCharges")
BINARY_INT_COLUMNS: tuple[str, ...] = ("SeniorCitizen",)
CATEGORICAL_COLUMNS: tuple[str, ...] = tuple(
    col for col in FEATURE_COLUMNS if col not in NUMERIC_COLUMNS and col not in BINARY_INT_COLUMNS
)


class DataValidationError(ValueError):
    """Raised when input dataset fails schema or quality checks."""


@dataclass
class ValidationReport:
    """Summary of dataset validation checks."""

    row_count: int
    column_count: int
    missing_values: dict[str, int]
    duplicate_rows: int
    duplicate_customer_ids: int
    churn_distribution: dict[str, int]
    passed: bool


def load_raw_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV dataset."""
    csv_path = Path(path) if path is not None else settings.RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded raw dataset from %s with shape %s", csv_path, df.shape)
    return df


def validate_dataset(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Validate schema, target values, and basic numeric constraints."""
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    expected: Iterable[str] = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing_cols = [col for col in expected if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    if require_target:
        valid_targets = {"Yes", "No"}
        observed_targets = set(df["Churn"].dropna().astype(str).unique())
        if not observed_targets.issubset(valid_targets):
            raise DataValidationError(
                f"Invalid Churn target values: {observed_targets - valid_targets}"
            )
        if df["Churn"].isna().any():
            raise DataValidationError("Target column 'Churn' contains missing values.")

    if (pd.to_numeric(df["tenure"], errors="coerce").dropna() < 0).any():
        raise DataValidationError("Column 'tenure' contains negative values.")

    if (pd.to_numeric(df["MonthlyCharges"], errors="coerce").dropna() < 0).any():
        raise DataValidationError("Column 'MonthlyCharges' contains negative values.")

    missing_values = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else 0
    churn_dist = (
        {str(k): int(v) for k, v in df["Churn"].value_counts().items()} if require_target else {}
    )

    report = ValidationReport(
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        duplicate_customer_ids=duplicate_ids,
        churn_distribution=churn_dist,
        passed=True,
    )
    logger.info(
        "Dataset validation passed: rows=%d cols=%d duplicates=%d",
        report.row_count,
        report.column_count,
        report.duplicate_rows,
    )
    return report


def clean_dataset(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Clean raw churn data: coerce types, handle blanks, and encode target."""
    cleaned = df.copy()

    # Strip whitespace in string columns and convert empty strings to NaN.
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned.loc[cleaned[col].isin(["", "nan", "None", "NA"]), col] = np.nan

    # Coerce TotalCharges (known to contain blank strings for tenure == 0) to float.
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")
    cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # For customers with missing TotalCharges, impute with tenure * MonthlyCharges (0 when tenure == 0).
    missing_tc = cleaned["TotalCharges"].isna()
    if missing_tc.any():
        inferred = (cleaned.loc[missing_tc, "tenure"].fillna(0) * cleaned.loc[missing_tc, "MonthlyCharges"].fillna(0))
        cleaned.loc[missing_tc, "TotalCharges"] = inferred

    if is_training and "customerID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["customerID"]).reset_index(drop=True)

    if is_training and "Churn" in cleaned.columns:
        cleaned["Churn"] = cleaned["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    logger.info("Cleaned dataset shape=%s (is_training=%s)", cleaned.shape, is_training)
    return cleaned
