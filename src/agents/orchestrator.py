"""Phase 3 — Orchestrator Agent.

Coordinates the three sub-agents (gate, drift, observability) and
produces a unified pipeline verdict with self-healing recommendations.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np

from src.agents.gate_agent import gate_decision_llm
from src.agents.drift_agent import detect_drift, drift_analysis_llm
from src.agents.observability_agent import (
    check_performance_degradation,
    observability_summary_llm,
)
from src.config import AGENT_CFG, GROQ_API_KEY

logger = logging.getLogger(__name__)


def run_orchestrator(
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float] | None = None,
    reference_data: np.ndarray | None = None,
    current_data: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> Dict[str, Any]:
    """Execute the full agent pipeline:

    1. Gate Agent   → approve/block the model
    2. Drift Agent  → detect data drift (if data provided)
    3. Obs Agent    → check for performance degradation
    4. Synthesise   → produce a final verdict with action plan

    Returns a unified report dictionary.
    """
    logger.info("═══ Orchestrator Agent starting ═══")

    # 1. Gate decision
    logger.info("▸ Running Gate Agent …")
    gate_result = gate_decision_llm(metrics)

    # 2. Drift analysis (optional — needs reference + current data)
    drift_report = None
    drift_analysis = None
    if reference_data is not None and current_data is not None:
        logger.info("▸ Running Drift Agent …")
        drift_report = detect_drift(reference_data, current_data, feature_names)
        drift_analysis = drift_analysis_llm(drift_report)

    # 3. Observability check (needs baseline)
    obs_report = None
    obs_summary = None
    if baseline_metrics is not None:
        logger.info("▸ Running Observability Agent …")
        obs_report = check_performance_degradation(baseline_metrics, metrics)
        obs_summary = observability_summary_llm(obs_report, drift_report)

    # 4. Synthesise final verdict
    verdict = _synthesise_verdict(gate_result, drift_analysis, obs_summary)

    report = {
        "verdict": verdict,
        "gate": gate_result,
        "drift": {"report": drift_report, "analysis": drift_analysis},
        "observability": {"report": obs_report, "summary": obs_summary},
    }

    logger.info("═══ Orchestrator verdict: %s — action: %s ═══",
                verdict["status"], verdict["action"])
    return report


def _synthesise_verdict(
    gate: Dict[str, Any],
    drift: Dict[str, Any] | None,
    obs: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Combine sub-agent results into a single verdict."""
    # Determine overall status
    if gate["decision"] == "BLOCK":
        status = "BLOCKED"
        action = "fix_model"
    elif drift and drift.get("action") == "rollback":
        status = "CRITICAL_DRIFT"
        action = "rollback"
    elif drift and drift.get("action") == "retrain":
        status = "DRIFT_RETRAIN"
        action = "retrain"
    elif obs and obs.get("action") == "retrain":
        status = "DEGRADED"
        action = "retrain"
    elif obs and obs.get("action") == "investigate":
        status = "WARNING"
        action = "investigate"
    else:
        status = "HEALTHY"
        action = "deploy"

    return {
        "status": status,
        "action": action,
        "gate_decision": gate["decision"],
        "drift_severity": drift.get("severity", "N/A") if drift else "N/A",
        "obs_severity": obs.get("severity", "OK") if obs else "OK",
    }
