"""FastAPI prediction service for customer churn."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftCheckResponse,
    HealthResponse,
    PredictionResponse,
)
from churn_ml.config import logger, settings
from churn_ml.drift import detect_data_drift
from churn_ml.predict import ChurnPredictor

predictor = ChurnPredictor()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load trained model on startup if available."""
    try:
        predictor.load()
    except FileNotFoundError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.MODEL_VERSION,
    description="Production customer churn prediction API with MLflow versioning and drift monitoring.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


def _ensure_model_loaded() -> None:
    if not predictor.is_loaded:
        try:
            predictor.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return service health and loaded model status."""
    meta = predictor.metadata
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_name=str(meta.get("best_model", "unloaded")),
        model_version=str(meta.get("model_version", settings.MODEL_VERSION)),
    )


@app.get("/model/info", tags=["Model"])
def model_info() -> dict[str, Any]:
    """Return active model version, MLflow run ID, metrics, and comparison results."""
    _ensure_model_loaded()
    return predictor.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    """Predict churn probability and confidence for a single customer."""
    _ensure_model_loaded()
    result = predictor.predict_records([customer.model_dump()])[0]
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict churn for a batch of customers."""
    _ensure_model_loaded()
    records = [c.model_dump() for c in payload.customers]
    results = predictor.predict_records(records)
    churn_count = sum(1 for r in results if r["churn_prediction"] == "Yes")
    total = len(results)
    return BatchPredictionResponse(
        total_customers=total,
        predicted_churn_count=churn_count,
        predicted_churn_rate=round(churn_count / total, 4) if total else 0.0,
        model_name=results[0]["model_name"],
        model_version=results[0]["model_version"],
        predictions=[PredictionResponse(**r) for r in results],
    )


@app.post("/drift/check", response_model=DriftCheckResponse, tags=["Monitoring"])
def check_drift(payload: BatchPredictionRequest) -> DriftCheckResponse:
    """Check incoming batch features for distribution drift against training data."""
    df = pd.DataFrame([c.model_dump() for c in payload.customers])
    try:
        report = detect_data_drift(df)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return DriftCheckResponse(**report)
