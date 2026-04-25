"""Streamlit dashboard for model comparison, feature importance, and drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from churn_ml.config import settings
from churn_ml.drift import detect_data_drift
from churn_ml.predict import ChurnPredictor

st.set_page_config(page_title="Churn MLOps Dashboard", layout="wide")
st.title("Customer Churn Prediction Platform — MLOps Dashboard")

metadata_path = settings.ARTIFACTS_DIR / "model_metadata.json"
if not metadata_path.exists():
    st.warning("No trained model metadata found. Run `python -m churn_ml.train` first.")
    st.stop()

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", metadata["best_model"])
col2.metric("Model Version", metadata["model_version"])
col3.metric("Best ROC-AUC", f"{metadata['best_metrics']['roc_auc']:.4f}")
col4.metric("Best F1-Score", f"{metadata['best_metrics']['f1_score']:.4f}")

st.subheader("Model Performance Comparison")
rows = []
for item in metadata.get("comparison", []):
    m = item["metrics"]
    rows.append(
        {
            "Model": item["model_name"],
            "Imbalance Strategy": item["imbalance_strategy"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

roc_img = settings.REPORTS_DIR / "roc_comparison.png"
fi_img = settings.REPORTS_DIR / f"feature_importance_{metadata['best_model']}.png"
cm_img = settings.REPORTS_DIR / f"confusion_matrix_{metadata['best_model']}.png"

c1, c2 = st.columns(2)
if roc_img.exists():
    c1.image(str(roc_img), caption="ROC-AUC Model Comparison")
if cm_img.exists():
    c2.image(str(cm_img), caption=f"Confusion Matrix ({metadata['best_model']})")
if fi_img.exists():
    st.image(str(fi_img), caption=f"Top Feature Importances ({metadata['best_model']})")

st.subheader("Interactive Prediction & Drift Check")
uploaded = st.file_uploader("Upload a customer CSV for batch prediction and drift analysis", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    predictor = ChurnPredictor().load()
    preds = predictor.predict_records(df.to_dict(orient="records"))
    st.dataframe(pd.DataFrame(preds).head(25), use_container_width=True)

    drift_report = detect_data_drift(df)
    st.write("Overall Drift Detected:", drift_report["overall_drift_detected"])
    st.dataframe(pd.DataFrame(drift_report["features"]), use_container_width=True)
