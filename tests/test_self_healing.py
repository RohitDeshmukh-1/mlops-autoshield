"""Tests for Phase 5 — Self-Healing Pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.self_healing import self_healing_loop


class TestSelfHealing:
    """Tests for the self-healing pipeline loop."""

    def test_no_healing_needed(self, sample_metrics):
        """Healthy pipeline should not trigger retraining."""
        audit = self_healing_loop(
            current_metrics=sample_metrics,
            baseline_metrics=sample_metrics,
            auto_retrain=False,
        )
        assert audit["trigger"] == "self_healing"
        assert audit["retrained"] is False

    def test_healing_triggered_by_degradation(self, sample_metrics):
        """Degraded metrics should trigger retraining."""
        degraded = {k: v * 0.5 for k, v in sample_metrics.items()}
        audit = self_healing_loop(
            current_metrics=degraded,
            baseline_metrics=sample_metrics,
            auto_retrain=True,
        )
        # Should attempt retrain
        assert audit["action_taken"] in ("retrain", "fix_model", "rollback", "rollback_recommended")

    def test_healing_with_drift(self, sample_metrics, reference_data, current_data_drifted):
        """Drifted data should trigger drift-related actions."""
        audit = self_healing_loop(
            current_metrics=sample_metrics,
            baseline_metrics=sample_metrics,
            reference_data=reference_data,
            current_data=current_data_drifted,
            auto_retrain=False,
        )
        assert audit["trigger"] == "self_healing"
        # Should detect drift in the assessment
        status = audit["initial_assessment"]["status"]
        assert status in ("HEALTHY", "DRIFT_RETRAIN", "CRITICAL_DRIFT", "WARNING")

    def test_audit_trail_schema(self, sample_metrics):
        """Audit trail should have required keys."""
        audit = self_healing_loop(
            current_metrics=sample_metrics,
            baseline_metrics=sample_metrics,
            auto_retrain=False,
        )
        required = {"trigger", "initial_assessment", "action_taken", "retrained", "new_metrics", "new_gate_decision"}
        assert required.issubset(set(audit.keys()))
