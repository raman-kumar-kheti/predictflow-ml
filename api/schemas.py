"""Pydantic request and response schemas for the churn prediction API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input schema for a single customer."""

    customerID: str = Field(default="CUST-0001", description="Unique customer identifier")
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float | str = Field(default=0.0)


class PredictionResponse(BaseModel):
    """Prediction response for a single customer."""

    customer_id: str
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    confidence_score: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with summary counts."""

    total_customers: int
    predicted_churn_count: int
    predicted_churn_rate: float
    model_name: str
    model_version: str
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class DriftCheckResponse(BaseModel):
    """Data drift monitoring response."""

    reference_rows: int
    current_rows: int
    psi_threshold: float
    ks_alpha: float
    overall_drift_detected: bool
    drifted_features: list[str]
    features: list[dict[str, Any]]
