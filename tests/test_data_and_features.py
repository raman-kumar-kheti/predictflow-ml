"""Tests for dataset ingestion, validation, cleaning, and feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_ml.data_ingestion import (
    FEATURE_COLUMNS,
    DataValidationError,
    clean_dataset,
    load_raw_dataset,
    validate_dataset,
)
from churn_ml.features import ChurnFeatureEngineer, build_preprocessor


def test_raw_dataset_loads_and_validates() -> None:
    df = load_raw_dataset()
    report = validate_dataset(df, require_target=True)
    assert report.passed is True
    assert report.row_count == 7043
    assert report.column_count == 21


def test_validate_dataset_rejects_missing_columns(sample_raw_df: pd.DataFrame) -> None:
    bad_df = sample_raw_df.drop(columns=["Contract"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(bad_df, require_target=True)


def test_validate_dataset_rejects_invalid_target(sample_raw_df: pd.DataFrame) -> None:
    bad_df = sample_raw_df.copy()
    bad_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="Invalid Churn target values"):
        validate_dataset(bad_df, require_target=True)


def test_clean_dataset_handles_blank_total_charges(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_dataset(sample_raw_df, is_training=True)
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert cleaned.loc[1, "TotalCharges"] == pytest.approx(48 * 45.0)


def test_feature_engineering_and_preprocessor(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_dataset(sample_raw_df, is_training=True)
    X = cleaned[list(FEATURE_COLUMNS)]
    engineered = ChurnFeatureEngineer().fit_transform(X)
    assert {"avg_monthly_spend", "charges_ratio", "num_services", "tenure_group"}.issubset(
        engineered.columns
    )
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(FEATURE_COLUMNS)
