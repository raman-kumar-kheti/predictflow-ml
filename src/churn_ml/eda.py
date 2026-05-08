"""Automated Exploratory Data Analysis (EDA) for customer churn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from churn_ml.config import logger, settings
from churn_ml.data_ingestion import NUMERIC_COLUMNS, clean_dataset, load_raw_dataset, validate_dataset


def run_eda(
    df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate EDA statistics and visualizations for the churn dataset."""
    out_dir = output_dir or (settings.REPORTS_DIR / "eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        raw_df = load_raw_dataset()
        validate_dataset(raw_df, require_target=True)
        df = clean_dataset(raw_df, is_training=True)

    sns.set_theme(style="whitegrid")

    # 1. Target distribution plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Churn"].value_counts().sort_index()
    ax.bar(["Retained (0)", "Churned (1)"], [counts.get(0, 0), counts.get(1, 0)], color=["#2a9d8f", "#e76f51"])
    ax.set_title("Customer Churn Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "churn_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Numeric feature distributions by Churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLUMNS):
        sns.kdeplot(data=df, x=col, hue="Churn", fill=True, common_norm=False, ax=ax)
        ax.set_title(f"{col} by Churn")
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_distributions.png", dpi=150)
    plt.close(fig)

    # 3. Correlation heatmap of numeric features + Churn
    corr_cols = list(NUMERIC_COLUMNS) + ["SeniorCitizen", "Churn"]
    corr_matrix = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Churn rate by Contract and InternetService
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ("Contract", "InternetService")):
        rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        rates.plot(kind="bar", ax=ax, color="#264653")
        ax.set_title(f"Churn Rate by {col}")
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "categorical_churn_rates.png", dpi=150)
    plt.close(fig)

    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "churn_rate": round(float(df["Churn"].mean()), 4),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "numeric_summary": df[list(NUMERIC_COLUMNS)].describe().round(3).to_dict(),
        "churn_by_contract": {
            str(k): round(float(v), 4) for k, v in df.groupby("Contract")["Churn"].mean().items()
        },
        "artifacts": [
            str(out_dir / "churn_distribution.png"),
            str(out_dir / "numeric_distributions.png"),
            str(out_dir / "correlation_heatmap.png"),
            str(out_dir / "categorical_churn_rates.png"),
        ],
    }

    summary_path = out_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved EDA report and 4 plots to %s", out_dir)
    return summary


if __name__ == "__main__":
    run_eda()
