"""Data drift monitoring using Population Stability Index (PSI) and KS test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churn_ml.config import logger, settings
from churn_ml.data_ingestion import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean_dataset


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index for a numeric feature."""
    exp_vals = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    act_vals = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if len(exp_vals) == 0 or len(act_vals) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(exp_vals, quantiles))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_vals, bins=edges)
    act_counts, _ = np.histogram(act_vals, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / len(exp_vals), eps, None)
    act_pct = np.clip(act_counts / len(act_vals), eps, None)
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)


def calculate_categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Compute Population Stability Index for a categorical feature."""
    exp_dist = expected.astype(str).value_counts(normalize=True)
    act_dist = actual.astype(str).value_counts(normalize=True)
    categories = set(exp_dist.index) | set(act_dist.index)
    eps = 1e-6
    psi = 0.0
    for cat in categories:
        e = max(float(exp_dist.get(cat, 0.0)), eps)
        a = max(float(act_dist.get(cat, 0.0)), eps)
        psi += (a - e) * np.log(a / e)
    return round(float(psi), 4)


def detect_data_drift(
    current_df: pd.DataFrame,
    reference_path: Path | None = None,
    psi_threshold: float | None = None,
    ks_alpha: float | None = None,
) -> dict[str, Any]:
    """Compare incoming data against the training reference distribution."""
    ref_path = reference_path or (settings.ARTIFACTS_DIR / "reference_data.parquet")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {ref_path}")

    psi_limit = settings.DRIFT_PSI_THRESHOLD if psi_threshold is None else psi_threshold
    alpha = settings.DRIFT_KS_ALPHA if ks_alpha is None else ks_alpha

    ref_df = pd.read_parquet(ref_path)
    cur_clean = clean_dataset(current_df, is_training=False)

    feature_reports: list[dict[str, Any]] = []
    drifted_features: list[str] = []

    for col in NUMERIC_COLUMNS:
        if col not in cur_clean.columns or col not in ref_df.columns:
            continue
        psi = calculate_psi(ref_df[col], cur_clean[col])
        ks_stat, p_val = ks_2samp(
            pd.to_numeric(ref_df[col], errors="coerce").dropna(),
            pd.to_numeric(cur_clean[col], errors="coerce").dropna(),
        )
        drifted = bool(psi >= psi_limit or p_val < alpha)
        if drifted:
            drifted_features.append(col)
        feature_reports.append(
            {
                "feature": col,
                "type": "numeric",
                "psi": psi,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_val), 4),
                "drift_detected": drifted,
            }
        )

    for col in CATEGORICAL_COLUMNS:
        if col not in cur_clean.columns or col not in ref_df.columns:
            continue
        psi = calculate_categorical_psi(ref_df[col], cur_clean[col])
        drifted = bool(psi >= psi_limit)
        if drifted:
            drifted_features.append(col)
        feature_reports.append(
            {
                "feature": col,
                "type": "categorical",
                "psi": psi,
                "ks_statistic": None,
                "ks_p_value": None,
                "drift_detected": drifted,
            }
        )

    report = {
        "reference_rows": int(len(ref_df)),
        "current_rows": int(len(cur_clean)),
        "psi_threshold": psi_limit,
        "ks_alpha": alpha,
        "overall_drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "features": feature_reports,
    }
    logger.info("Drift check complete: drifted_features=%s", drifted_features)
    return report
