"""Shared fixtures for AutoShield test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create a small synthetic fraud dataset for testing."""
    rng = np.random.RandomState(42)
    n = 500
    data = {f"V{i}": rng.randn(n) for i in range(1, 29)}
    data["Time"] = np.sort(rng.uniform(0, 172_792, n))
    data["Amount"] = np.abs(rng.lognormal(3, 2, n))
    fraud_mask = rng.rand(n) < 0.05
    data["Class"] = fraud_mask.astype(int)
    # Make fraud rows slightly distinct
    for col in ["V1", "V2", "V3"]:
        data[col] = np.where(fraud_mask, data[col] + 2.0, data[col])
    return pd.DataFrame(data)


@pytest.fixture
def sample_metrics() -> dict:
    """Good model metrics that should pass the gate."""
    return {
        "accuracy": 0.95,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "avg_latency_ms": 5.0,
        "roc_auc": 0.92,
        "avg_precision": 0.85,
    }


@pytest.fixture
def bad_metrics() -> dict:
    """Poor model metrics that should be blocked by the gate."""
    return {
        "accuracy": 0.60,
        "precision": 0.30,
        "recall": 0.20,
        "f1": 0.24,
        "avg_latency_ms": 200.0,
    }


@pytest.fixture
def reference_data() -> np.ndarray:
    """Reference feature data (no drift)."""
    rng = np.random.RandomState(42)
    return rng.randn(500, 10)


@pytest.fixture
def current_data_no_drift(reference_data) -> np.ndarray:
    """Current data sampled from same distribution (no drift)."""
    rng = np.random.RandomState(99)
    return rng.randn(500, 10)


@pytest.fixture
def current_data_drifted(reference_data) -> np.ndarray:
    """Current data with significant drift injected."""
    rng = np.random.RandomState(99)
    data = rng.randn(500, 10)
    # Inject large drift in first 5 features
    data[:, :5] += 5.0  # shift mean by 5 std devs
    return data
