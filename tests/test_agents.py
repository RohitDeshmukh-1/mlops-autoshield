"""Tests for Phase 3 — GenAI Agents (gate, drift, observability, orchestrator)."""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.gate_agent import gate_decision_rules, gate_decision_llm
from src.agents.drift_agent import compute_psi, compute_ks_test, detect_drift, drift_analysis_rules
from src.agents.observability_agent import check_performance_degradation, observability_summary_rules
from src.agents.orchestrator import run_orchestrator, _synthesise_verdict


class TestGateAgent:
    """Tests for CI/CD gate agent."""

    def test_approve_good_metrics(self, sample_metrics):
        """Good metrics should be approved."""
        result = gate_decision_rules(sample_metrics)
        assert result["decision"] == "APPROVE"
        assert len(result["failed_metrics"]) == 0
        assert result["rationale"] != ""

    def test_block_bad_metrics(self, bad_metrics):
        """Bad metrics should be blocked."""
        result = gate_decision_rules(bad_metrics)
        assert result["decision"] == "BLOCK"
        assert len(result["failed_metrics"]) > 0
        assert len(result["recommendations"]) > 0

    def test_block_low_accuracy(self, sample_metrics):
        """Model with low accuracy should be blocked."""
        metrics = {**sample_metrics, "accuracy": 0.50}
        result = gate_decision_rules(metrics)
        assert result["decision"] == "BLOCK"
        failed_names = [f["metric"] for f in result["failed_metrics"]]
        assert "accuracy" in failed_names

    def test_block_high_latency(self, sample_metrics):
        """Model with high latency should be blocked."""
        metrics = {**sample_metrics, "avg_latency_ms": 500.0}
        result = gate_decision_rules(metrics)
        assert result["decision"] == "BLOCK"
        failed_names = [f["metric"] for f in result["failed_metrics"]]
        assert "avg_latency_ms" in failed_names

    def test_gate_llm_fallback(self, sample_metrics):
        """LLM gate should fall back to rules when no API key."""
        result = gate_decision_llm(sample_metrics)
        assert result["decision"] in ("APPROVE", "BLOCK")

    def test_gate_result_schema(self, sample_metrics):
        """Gate result should have required keys."""
        result = gate_decision_rules(sample_metrics)
        required_keys = {"decision", "failed_metrics", "rationale", "recommendations"}
        assert required_keys.issubset(set(result.keys()))


class TestDriftAgent:
    """Tests for drift detection."""

    def test_compute_psi_no_drift(self):
        """PSI should be low for same distribution."""
        rng = np.random.RandomState(42)
        ref = rng.randn(1000)
        cur = rng.randn(1000)
        psi = compute_psi(ref, cur)
        assert psi < 0.2  # below threshold

    def test_compute_psi_with_drift(self):
        """PSI should be high for shifted distribution."""
        rng = np.random.RandomState(42)
        ref = rng.randn(1000)
        cur = rng.randn(1000) + 5.0  # large shift
        psi = compute_psi(ref, cur)
        assert psi > 0.2  # above threshold

    def test_ks_test_no_drift(self):
        """KS test should not reject H0 for same distribution."""
        rng = np.random.RandomState(42)
        ref = rng.randn(1000)
        cur = rng.randn(1000)
        result = compute_ks_test(ref, cur)
        assert result["p_value"] > 0.05  # do not reject

    def test_ks_test_with_drift(self):
        """KS test should reject H0 for different distributions."""
        rng = np.random.RandomState(42)
        ref = rng.randn(1000)
        cur = rng.randn(1000) + 3.0
        result = compute_ks_test(ref, cur)
        assert result["p_value"] < 0.05  # reject H0

    def test_detect_drift_no_drift(self, reference_data, current_data_no_drift):
        """No drift should be detected for same distribution."""
        report = detect_drift(reference_data, current_data_no_drift)
        assert report["total_features"] == 10
        # Most features should not drift (allow some noise)
        assert report["drifted_count"] <= 3

    def test_detect_drift_with_drift(self, reference_data, current_data_drifted):
        """Drift should be detected for shifted data."""
        report = detect_drift(reference_data, current_data_drifted)
        assert report["drift_detected"] is True
        assert report["drifted_count"] >= 3  # at least the shifted features

    def test_drift_report_schema(self, reference_data, current_data_no_drift):
        """Drift report should have required keys."""
        report = detect_drift(reference_data, current_data_no_drift)
        required_keys = {"total_features", "drifted_count", "drifted_features", "drift_detected", "details"}
        assert required_keys.issubset(set(report.keys()))

    def test_drift_analysis_rules_no_drift(self, reference_data, current_data_no_drift):
        """Rules should return LOW severity for no drift."""
        report = detect_drift(reference_data, current_data_no_drift)
        # Force no drift for this test
        report["drifted_count"] = 0
        analysis = drift_analysis_rules(report)
        assert analysis["severity"] == "LOW"
        assert analysis["action"] == "monitor"

    def test_drift_analysis_rules_critical(self):
        """Rules should return CRITICAL for majority drift."""
        report = {"drifted_count": 8, "total_features": 10, "drifted_features": []}
        analysis = drift_analysis_rules(report)
        assert analysis["severity"] == "CRITICAL"
        assert analysis["action"] == "rollback"


class TestObservabilityAgent:
    """Tests for performance monitoring."""

    def test_no_degradation(self, sample_metrics):
        """No alerts when metrics are stable."""
        result = check_performance_degradation(sample_metrics, sample_metrics)
        assert result["alert_triggered"] is False
        assert result["alert_count"] == 0

    def test_accuracy_degradation(self, sample_metrics):
        """Alert should fire when accuracy drops significantly."""
        degraded = {**sample_metrics, "accuracy": 0.70}
        result = check_performance_degradation(sample_metrics, degraded)
        assert result["alert_triggered"] is True
        alert_metrics = [a["metric"] for a in result["alerts"]]
        assert "accuracy" in alert_metrics

    def test_latency_increase(self, sample_metrics):
        """Alert should fire when latency increases significantly."""
        degraded = {**sample_metrics, "avg_latency_ms": 50.0}
        result = check_performance_degradation(sample_metrics, degraded)
        assert result["alert_triggered"] is True

    def test_summary_ok(self, sample_metrics):
        """Summary should report OK when no alerts."""
        alert_report = check_performance_degradation(sample_metrics, sample_metrics)
        summary = observability_summary_rules(alert_report)
        assert summary["severity"] == "OK"

    def test_summary_critical(self, sample_metrics):
        """Summary should report CRITICAL when many metrics degrade."""
        degraded = {k: v * 0.5 for k, v in sample_metrics.items() if isinstance(v, float)}
        alert_report = check_performance_degradation(sample_metrics, degraded)
        summary = observability_summary_rules(alert_report)
        assert summary["severity"] in ("WARNING", "CRITICAL")


class TestOrchestrator:
    """Tests for the orchestrator agent."""

    def test_healthy_pipeline(self, sample_metrics, reference_data, current_data_no_drift):
        """Healthy pipeline should return deploy action."""
        report = run_orchestrator(
            metrics=sample_metrics,
            baseline_metrics=sample_metrics,
            reference_data=reference_data,
            current_data=current_data_no_drift,
        )
        assert report["verdict"]["status"] in ("HEALTHY", "WARNING")
        assert report["verdict"]["gate_decision"] == "APPROVE"

    def test_blocked_model(self, bad_metrics):
        """Bad model should be blocked."""
        report = run_orchestrator(
            metrics=bad_metrics,
            baseline_metrics=bad_metrics,
        )
        assert report["verdict"]["status"] == "BLOCKED"
        assert report["verdict"]["action"] == "fix_model"

    def test_synthesise_approve(self, sample_metrics):
        """Synthesise should return HEALTHY when all pass."""
        gate = {"decision": "APPROVE", "failed_metrics": [], "rationale": "", "recommendations": []}
        verdict = _synthesise_verdict(gate, None, None)
        assert verdict["status"] == "HEALTHY"
        assert verdict["action"] == "deploy"

    def test_synthesise_block(self, bad_metrics):
        """Synthesise should return BLOCKED when gate blocks."""
        gate = {"decision": "BLOCK", "failed_metrics": [{}], "rationale": "", "recommendations": []}
        verdict = _synthesise_verdict(gate, None, None)
        assert verdict["status"] == "BLOCKED"
        assert verdict["action"] == "fix_model"

    def test_orchestrator_with_drift(self, sample_metrics, reference_data, current_data_drifted):
        """Drifted data should trigger drift-related verdict."""
        report = run_orchestrator(
            metrics=sample_metrics,
            baseline_metrics=sample_metrics,
            reference_data=reference_data,
            current_data=current_data_drifted,
        )
        # Should detect drift
        assert report["drift"]["report"]["drift_detected"] is True
