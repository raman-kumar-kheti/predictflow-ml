"""Configuration, paths, reproducibility, and structured logging.

Adapted from Made With ML's configuration and logging patterns for a
CPU-friendly tabular customer churn MLOps pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "Customer Churn Prediction Platform"
    MODEL_VERSION: str = "1.0.0"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    EXPERIMENT_NAME: str = "customer-churn-classification"
    REGISTERED_MODEL_NAME: str = "customer-churn-best-model"
    LOG_LEVEL: str = "INFO"
    DRIFT_PSI_THRESHOLD: float = 0.2
    DRIFT_KS_ALPHA: float = 0.05

    ROOT_DIR: Path = Path(__file__).resolve().parents[2]
    RAW_DATA_PATH: Path = ROOT_DIR / "data" / "raw" / "telco_customer_churn.csv"
    ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
    REPORTS_DIR: Path = ROOT_DIR / "reports"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    MLFLOW_TRACKING_URI: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.MLFLOW_TRACKING_URI:
            mlflow_db = (self.ROOT_DIR / "mlflow.db").resolve()
            self.MLFLOW_TRACKING_URI = f"sqlite:///{mlflow_db}"
        for directory in (self.ARTIFACTS_DIR, self.REPORTS_DIR, self.LOGS_DIR):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            payload["data"] = record.extra_data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str = "churn_ml") -> logging.Logger:
    """Return a configured logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    file_handler = logging.FileHandler(settings.LOGS_DIR / "pipeline.jsonl", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def set_seeds(seed: int | None = None) -> int:
    """Set Python and NumPy random seeds for reproducibility."""
    resolved = settings.RANDOM_STATE if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(resolved)
    random.seed(resolved)
    np.random.seed(resolved)
    return resolved


logger = get_logger()
