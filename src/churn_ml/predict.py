"""Model loading and single/batch prediction with confidence scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn_ml.config import logger, settings
from churn_ml.data_ingestion import FEATURE_COLUMNS, clean_dataset


def _risk_level(churn_probability: float) -> str:
    if churn_probability >= 0.70:
        return "HIGH"
    if churn_probability >= 0.40:
        return "MEDIUM"
    return "LOW"


class ChurnPredictor:
    """Wrapper around the selected Scikit-learn pipeline and its metadata."""

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or (settings.ARTIFACTS_DIR / "best_model.joblib")
        self.metadata_path = metadata_path or (settings.ARTIFACTS_DIR / "model_metadata.json")
        self._pipeline: Any = None
        self._metadata: dict[str, Any] = {}

    def load(self) -> "ChurnPredictor":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. Run training first."
            )
        self._pipeline = joblib.load(self.model_path)
        if self.metadata_path.exists():
            self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded model %s (version=%s)",
            self._metadata.get("best_model", "unknown"),
            self._metadata.get("model_version", settings.MODEL_VERSION),
        )
        return self

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict churn label, probability, confidence score, and risk tier."""
        if not self.is_loaded:
            self.load()

        df = pd.DataFrame(records)
        customer_ids = (
            df["customerID"].tolist() if "customerID" in df.columns else [f"row-{i}" for i in range(len(df))]
        )
        cleaned = clean_dataset(df, is_training=False)
        X = cleaned[list(FEATURE_COLUMNS)].copy()

        probas = self._pipeline.predict_proba(X)[:, 1]
        preds = (probas >= 0.5).astype(int)

        results: list[dict[str, Any]] = []
        for cid, pred, prob in zip(customer_ids, preds, probas):
            p = float(prob)
            confidence = max(p, 1.0 - p)
            results.append(
                {
                    "customer_id": str(cid),
                    "churn_prediction": "Yes" if int(pred) == 1 else "No",
                    "churn_probability": round(p, 4),
                    "confidence_score": round(confidence, 4),
                    "risk_tier": _risk_level(p),
                    "model_name": self._metadata.get("best_model", "unknown"),
                    "model_version": self._metadata.get("model_version", settings.MODEL_VERSION),
                }
            )
        return results
