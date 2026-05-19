"""Phase 4 — Prometheus metrics collector.

Exposes model-serving metrics (latency, predictions, accuracy, drift scores)
via the prometheus_client library.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
)

# ── Custom registry (avoids conflicts in tests) ───────────────────
REGISTRY = CollectorRegistry()

# ── Counters ──────────────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "model_prediction_total",
    "Total number of predictions served",
    ["model_version"],
    registry=REGISTRY,
)

FRAUD_DETECTED_COUNT = Counter(
    "model_fraud_detected_total",
    "Total fraud predictions (class=1)",
    registry=REGISTRY,
)

# ── Histograms ────────────────────────────────────────────────────
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

# ── Gauges ────────────────────────────────────────────────────────
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Current model accuracy on evaluation set",
    registry=REGISTRY,
)

MODEL_F1 = Gauge(
    "model_f1_score",
    "Current model F1 score",
    registry=REGISTRY,
)

DRIFT_PSI_MAX = Gauge(
    "drift_psi_max",
    "Maximum PSI across all features",
    registry=REGISTRY,
)

DRIFT_FEATURES_COUNT = Gauge(
    "drift_features_drifted",
    "Number of features currently drifted",
    registry=REGISTRY,
)

GATE_DECISION = Gauge(
    "gate_decision",
    "Last gate decision (1=APPROVE, 0=BLOCK)",
    registry=REGISTRY,
)


# ── Helpers ───────────────────────────────────────────────────────
def record_prediction(latency: float, is_fraud: bool, model_version: str = "v1"):
    """Record a single prediction event."""
    PREDICTION_COUNT.labels(model_version=model_version).inc()
    PREDICTION_LATENCY.observe(latency)
    if is_fraud:
        FRAUD_DETECTED_COUNT.inc()


def update_model_metrics(accuracy: float, f1: float):
    """Update model quality gauges."""
    MODEL_ACCURACY.set(accuracy)
    MODEL_F1.set(f1)


def update_drift_metrics(max_psi: float, drifted_count: int):
    """Update drift gauges from latest drift report."""
    DRIFT_PSI_MAX.set(max_psi)
    DRIFT_FEATURES_COUNT.set(drifted_count)


def update_gate_decision(approved: bool):
    """Update gate decision gauge."""
    GATE_DECISION.set(1 if approved else 0)


def get_metrics_text() -> str:
    """Return Prometheus exposition format text."""
    return generate_latest(REGISTRY).decode("utf-8")


def track_latency(func: Callable) -> Callable:
    """Decorator to track function execution latency."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)
        return result
    return wrapper
