"""Tests for FastAPI endpoints."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


class TestMetricsEndpoint:
    def test_metrics_returns_text(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200


class TestGateEndpoint:
    def test_gate_approve(self, client):
        r = client.post("/gate/evaluate", json={
            "accuracy": 0.95, "precision": 0.80,
            "recall": 0.75, "f1": 0.77, "avg_latency_ms": 5.0,
        })
        assert r.status_code == 200
        assert r.json()["decision"] == "APPROVE"

    def test_gate_block(self, client):
        r = client.post("/gate/evaluate", json={
            "accuracy": 0.50, "precision": 0.20,
            "recall": 0.15, "f1": 0.17, "avg_latency_ms": 500.0,
        })
        assert r.status_code == 200
        assert r.json()["decision"] == "BLOCK"


class TestDriftEndpoint:
    def test_drift_check(self, client):
        rng = np.random.RandomState(42)
        r = client.post("/drift/check", json={
            "reference": rng.randn(50, 3).tolist(),
            "current": (rng.randn(50, 3) + 5.0).tolist(),
        })
        assert r.status_code == 200
        assert r.json()["report"]["drift_detected"] is True


class TestOrchestrateEndpoint:
    def test_orchestrate(self, client):
        r = client.post("/orchestrate", json={
            "accuracy": 0.95, "precision": 0.80,
            "recall": 0.75, "f1": 0.77, "avg_latency_ms": 5.0,
        })
        assert r.status_code == 200
        assert "verdict" in r.json()
