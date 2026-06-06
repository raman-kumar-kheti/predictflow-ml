"""Tests for training artifacts, prediction, and data drift monitoring."""

from __future__ import annotations

import json

import pandas as pd

from churn_ml.config import settings
from churn_ml.data_ingestion import load_raw_dataset
from churn_ml.drift import detect_data_drift
from churn_ml.predict import ChurnPredictor


def test_trained_artifacts_exist_and_predict(sample_customer_record: dict) -> None:
    model_path = settings.ARTIFACTS_DIR / "best_model.joblib"
    meta_path = settings.ARTIFACTS_DIR / "model_metadata.json"
    assert model_path.exists(), "Run `python -m churn_ml.train` before running tests."
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["best_metrics"]["roc_auc"] > 0.75

    predictor = ChurnPredictor().load()
    preds = predictor.predict_records([sample_customer_record])
    assert len(preds) == 1
    assert preds[0]["churn_prediction"] in {"Yes", "No"}
    assert 0.0 <= preds[0]["churn_probability"] <= 1.0
    assert 0.5 <= preds[0]["confidence_score"] <= 1.0


def test_drift_detection_flags_shifted_distribution() -> None:
    raw = load_raw_dataset().head(200).copy()
    no_drift = detect_data_drift(raw)
    assert "features" in no_drift

    shifted = raw.copy()
    shifted["MonthlyCharges"] = 9999.0
    shifted["tenure"] = 0
    drift_report = detect_data_drift(shifted)
    assert drift_report["overall_drift_detected"] is True
    assert "MonthlyCharges" in drift_report["drifted_features"]
