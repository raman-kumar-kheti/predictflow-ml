"""Training, class-imbalance handling, MLflow tracking, and best-model selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_ml.config import logger, set_seeds, settings
from churn_ml.data_ingestion import FEATURE_COLUMNS, clean_dataset, load_raw_dataset, validate_dataset
from churn_ml.evaluate import (
    build_roc_curve_data,
    compute_classification_metrics,
    extract_feature_importance,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_roc_comparison_plot,
)
from churn_ml.features import build_preprocessor


def get_candidate_models(random_state: int) -> dict[str, dict[str, Any]]:
    """Return candidate classifiers and imbalance strategies."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=random_state,
            ),
            "use_smote": False,
            "imbalance_strategy": "class_weight=balanced",
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            "use_smote": False,
            "imbalance_strategy": "class_weight=balanced",
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state,
            ),
            "use_smote": True,
            "imbalance_strategy": "SMOTE",
        },
    }


def _build_model_pipeline(
    estimator: Any,
    use_smote: bool,
    random_state: int,
) -> Pipeline | ImbPipeline:
    preprocessor = build_preprocessor()
    steps = list(preprocessor.steps)
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("classifier", estimator))
        return ImbPipeline(steps=steps)
    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def train_and_evaluate(
    data_path: Path | str | None = None,
    artifacts_dir: Path | None = None,
    reports_dir: Path | None = None,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """Run reproducible training across all models, log to MLflow, and save the best model."""
    seed = set_seeds(settings.RANDOM_STATE)
    art_dir = artifacts_dir or settings.ARTIFACTS_DIR
    rep_dir = reports_dir or settings.REPORTS_DIR
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(data_path)
    validate_dataset(raw_df, require_target=True)
    df = clean_dataset(raw_df, is_training=True)

    X = df[list(FEATURE_COLUMNS)].copy()
    y = df["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        stratify=y,
        random_state=seed,
    )

    # Save reference dataset for drift monitoring.
    ref_path = art_dir / "reference_data.parquet"
    X_train.to_parquet(ref_path, index=False)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    exp_name = experiment_name or settings.EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)

    candidates = get_candidate_models(seed)
    comparison_results: list[dict[str, Any]] = []
    roc_curves: dict[str, tuple[Any, Any, float]] = {}

    best_model_name = ""
    best_pipeline: Any = None
    best_metrics: dict[str, float] = {}
    best_run_id = ""
    best_score = -1.0

    for model_name, cfg in candidates.items():
        with mlflow.start_run(run_name=model_name) as run:
            pipeline = _build_model_pipeline(cfg["estimator"], cfg["use_smote"], seed)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = compute_classification_metrics(y_test, y_pred, y_proba)

            cm_path = rep_dir / f"confusion_matrix_{model_name}.png"
            cm_values = save_confusion_matrix_plot(y_test, y_pred, model_name, cm_path)

            importances = extract_feature_importance(pipeline, top_n=15)
            fi_path = rep_dir / f"feature_importance_{model_name}.png"
            save_feature_importance_plot(importances, model_name, fi_path)

            roc_curves[model_name] = build_roc_curve_data(y_test, y_proba)

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "imbalance_strategy": cfg["imbalance_strategy"],
                    "test_size": settings.TEST_SIZE,
                    "random_state": seed,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            if fi_path.exists():
                mlflow.log_artifact(str(fi_path), artifact_path="plots")
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            entry = {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "imbalance_strategy": cfg["imbalance_strategy"],
                "metrics": metrics,
                "confusion_matrix": cm_values,
                "top_features": importances,
            }
            comparison_results.append(entry)
            logger.info("Trained %s | metrics=%s", model_name, metrics)

            # Primary selection metric: ROC-AUC, tie-breaker: F1-score
            composite = metrics["roc_auc"] + 0.001 * metrics["f1_score"]
            if composite > best_score:
                best_score = composite
                best_model_name = model_name
                best_pipeline = pipeline
                best_metrics = metrics
                best_run_id = run.info.run_id

    roc_path = rep_dir / "roc_comparison.png"
    save_roc_comparison_plot(roc_curves, roc_path)

    # Register best model in MLflow Model Registry (local file store supports registry).
    model_version = settings.MODEL_VERSION
    try:
        model_uri = f"runs:/{best_run_id}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=settings.REGISTERED_MODEL_NAME)
        model_version = f"{settings.MODEL_VERSION} (mlflow-v{registered.version})"
    except Exception as exc:  # pragma: no cover - fallback if registry backend unavailable
        logger.warning("MLflow model registration skipped: %s", exc)

    # Persist best model locally for low-latency FastAPI serving.
    best_model_path = art_dir / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)

    metadata = {
        "project": settings.PROJECT_NAME,
        "best_model": best_model_name,
        "model_version": model_version,
        "mlflow_run_id": best_run_id,
        "mlflow_experiment": exp_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_metrics": best_metrics,
        "comparison": comparison_results,
        "feature_columns": list(FEATURE_COLUMNS),
        "reference_data_path": str(ref_path),
    }

    (art_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (rep_dir / "model_comparison.json").write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")
    logger.info("Best model selected: %s (ROC-AUC=%.4f, F1=%.4f)", best_model_name, best_metrics["roc_auc"], best_metrics["f1_score"])
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
