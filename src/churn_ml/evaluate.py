"""Model evaluation, confusion matrix, ROC curve, and feature importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def save_confusion_matrix_plot(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> list[list[int]]:
    """Save a confusion matrix heatmap PNG and return matrix values."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return cm.tolist()


def save_roc_comparison_plot(
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """Plot ROC curves for all evaluated models on one figure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, auc) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model ROC-AUC Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def extract_feature_importance(
    pipeline: Any,
    top_n: int = 15,
) -> list[dict[str, float | str]]:
    """Extract top feature importances or absolute coefficients from a fitted pipeline."""
    column_transformer = (
        pipeline.named_steps["column_transformer"]
        if "column_transformer" in pipeline.named_steps
        else pipeline.named_steps["preprocessor"].named_steps["column_transformer"]
    )
    classifier = pipeline.named_steps["classifier"]
    feature_names = list(column_transformer.get_feature_names_out())

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        importances = np.abs(np.asarray(classifier.coef_[0], dtype=float))
    else:
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": str(name), "importance": round(float(val), 6)} for name, val in pairs]


def save_feature_importance_plot(
    importances: list[dict[str, float | str]],
    model_name: str,
    output_path: Path,
) -> None:
    """Save horizontal bar chart of top feature importances."""
    if not importances:
        return
    df_imp = pd.DataFrame(importances).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_imp["feature"], df_imp["importance"], color="#2a9d8f")
    ax.set_title(f"Top Feature Importances — {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_roc_curve_data(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute FPR, TPR, and ROC-AUC for plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = float(roc_auc_score(y_true, y_proba))
    return fpr, tpr, auc
