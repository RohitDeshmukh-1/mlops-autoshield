"""Phase 1 & 2 — Model training, evaluation, and MLflow tracking.

Trains XGBoost (default), Random Forest, or Logistic Regression,
logs everything to MLflow, and pushes to the model registry.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

from src.config import MODEL_CFG, MLFLOW_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ── Model factory ──────────────────────────────────────────────────
def build_model(model_type: str | None = None, **override_params) -> Any:
    """Instantiate an sklearn-compatible classifier."""
    model_type = model_type or MODEL_CFG["type"]
    params = {**MODEL_CFG["params"].get(model_type, {}), **override_params}

    if model_type == "xgboost":
        return XGBClassifier(**params, use_label_encoder=False)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "logistic_regression":
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ── Evaluation ─────────────────────────────────────────────────────
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics."""
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    latency_ms = (time.perf_counter() - start) * 1000 / max(len(X_test), 1)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "avg_latency_ms": float(latency_ms),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        metrics["avg_precision"] = float(average_precision_score(y_test, y_proba))

    logger.info("Evaluation → %s", json.dumps(metrics, indent=2))
    return metrics


# ── Training pipeline ──────────────────────────────────────────────
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str | None = None,
    track_mlflow: bool = True,
    **extra_params,
) -> tuple[Any, Dict[str, float]]:
    """Train → evaluate → optionally log to MLflow.

    Returns (model, metrics_dict).
    """
    model = build_model(model_type, **extra_params)
    model_type = model_type or MODEL_CFG["type"]

    logger.info("Training %s model …", model_type)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    if track_mlflow:
        _log_to_mlflow(model, model_type, metrics)

    return model, metrics


def _log_to_mlflow(model: Any, model_type: str, metrics: Dict[str, float]):
    """Log params, metrics, and model artefact to MLflow."""
    tracking_uri = (PROJECT_ROOT / MLFLOW_CFG["tracking_uri"]).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_CFG["experiment_name"])

    with mlflow.start_run(run_name=f"{model_type}_run") as run:
        # Log params
        params = MODEL_CFG["params"].get(model_type, {})
        mlflow.log_params({k: str(v) for k, v in params.items()})

        # Log metrics
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("MLflow run logged → %s (run_id=%s)", model_type, run.info.run_id)
        return run.info.run_id


# ── Persistence ────────────────────────────────────────────────────
def save_model(model: Any, name: str = "model") -> Path:
    """Save model to disk with joblib."""
    save_dir = PROJECT_ROOT / MODEL_CFG["save_path"]
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str = "model") -> Any:
    """Load model from disk."""
    path = PROJECT_ROOT / MODEL_CFG["save_path"] / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}")
    return joblib.load(path)
