"""Feature engineering and Scikit-learn preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ml.data_ingestion import BINARY_INT_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

SERVICE_COLUMNS: tuple[str, ...] = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

ENGINEERED_NUMERIC_COLUMNS: tuple[str, ...] = (
    "avg_monthly_spend",
    "charges_ratio",
    "num_services",
    "is_long_term_contract",
    "has_auto_payment",
)

ENGINEERED_CATEGORICAL_COLUMNS: tuple[str, ...] = ("tenure_group",)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-specific customer churn features without data leakage."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ChurnFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        tenure_safe = df["tenure"].clip(lower=1)
        df["avg_monthly_spend"] = df["TotalCharges"] / tenure_safe
        df["charges_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 1e-6)

        df["num_services"] = sum(
            (df[col].astype(str) == "Yes").astype(int) for col in SERVICE_COLUMNS if col in df.columns
        )
        df["is_long_term_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
        df["has_auto_payment"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)

        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72, np.inf],
            labels=["0-12m", "13-24m", "25-48m", "49-72m", "72m+"],
        ).astype(str)

        return df


def build_preprocessor() -> Pipeline:
    """Create the end-to-end feature engineering + ColumnTransformer pipeline."""
    all_numeric = list(NUMERIC_COLUMNS) + list(BINARY_INT_COLUMNS) + list(ENGINEERED_NUMERIC_COLUMNS)
    all_categorical = list(CATEGORICAL_COLUMNS) + list(ENGINEERED_CATEGORICAL_COLUMNS)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, all_numeric),
            ("cat", categorical_pipeline, all_categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("feature_engineer", ChurnFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
