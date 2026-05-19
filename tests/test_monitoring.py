"""Tests for Phase 4 — Monitoring (Prometheus metrics, drift reporting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.metrics import (
    record_prediction,
    update_model_metrics,
    update_drift_metrics,
    update_gate_decision,
    get_metrics_text,
    REGISTRY,
)


class TestPrometheusMetrics:
    """Tests for Prometheus metric collection."""

    def test_record_prediction(self):
        """Recording predictions should update counters."""
        record_prediction(latency=0.01, is_fraud=False, model_version="v1")
        record_prediction(latency=0.02, is_fraud=True, model_version="v1")
        metrics_text = get_metrics_text()
        assert "model_prediction_total" in metrics_text
        assert "model_fraud_detected_total" in metrics_text
        assert "model_prediction_latency_seconds" in metrics_text

    def test_update_model_metrics(self):
        """Model quality gauges should be settable."""
        update_model_metrics(accuracy=0.95, f1=0.88)
        metrics_text = get_metrics_text()
        assert "model_accuracy" in metrics_text
        assert "model_f1_score" in metrics_text

    def test_update_drift_metrics(self):
        """Drift gauges should reflect current state."""
        update_drift_metrics(max_psi=0.35, drifted_count=3)
        metrics_text = get_metrics_text()
        assert "drift_psi_max" in metrics_text
        assert "drift_features_drifted" in metrics_text

    def test_update_gate_decision(self):
        """Gate gauge should be 1 for approve, 0 for block."""
        update_gate_decision(True)
        metrics_text = get_metrics_text()
        assert "gate_decision" in metrics_text

    def test_metrics_text_format(self):
        """Metrics should be in Prometheus exposition format."""
        text = get_metrics_text()
        assert isinstance(text, str)
        # Should contain HELP and TYPE lines
        assert "# HELP" in text or "# TYPE" in text


class TestDriftReporter:
    """Tests for Evidently drift report generation."""

    def test_fallback_report(self, tmp_path):
        """Fallback JSON report should be generated when Evidently is missing."""
        from src.monitoring.drift_reporter import _fallback_report

        rng = np.random.RandomState(42)
        ref_df = pd.DataFrame({"V1": rng.randn(100), "V2": rng.randn(100)})
        cur_df = pd.DataFrame({"V1": rng.randn(100) + 3, "V2": rng.randn(100)})

        path = _fallback_report(ref_df, cur_df, tmp_path, "test_001")
        assert path.exists()
        assert path.suffix == ".json"

        import json
        with open(path) as f:
            report = json.load(f)

        assert "features" in report
        assert len(report["features"]) == 2
        assert "psi" in report["features"][0]
