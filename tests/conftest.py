"""Shared pytest fixtures for customer churn pipeline tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_customer_record() -> dict:
    return {
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 268.5,
    }


@pytest.fixture
def sample_raw_df(sample_customer_record: dict) -> pd.DataFrame:
    r1 = {**sample_customer_record, "customerID": "C1", "Churn": "Yes"}
    r2 = {
        **sample_customer_record,
        "customerID": "C2",
        "tenure": 48,
        "Contract": "Two year",
        "InternetService": "DSL",
        "MonthlyCharges": 45.0,
        "TotalCharges": " ",
        "Churn": "No",
    }
    return pd.DataFrame([r1, r2])
