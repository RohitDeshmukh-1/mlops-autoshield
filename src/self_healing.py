"""Phase 5 — Self-healing pipeline.

Orchestrates the full self-healing loop:
  Drift detected → LLM decision → auto-retrain → gate re-eval → promote
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import numpy as np

from src.agents.orchestrator import run_orchestrator
from src.data.loader import load_raw_data, preprocess, split_and_scale
from src.models.trainer import train_model, save_model, evaluate_model
from src.monitoring.metrics import (
    update_model_metrics,
    update_drift_metrics,
    update_gate_decision,
)

logger = logging.getLogger(__name__)


def self_healing_loop(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    reference_data: np.ndarray | None = None,
    current_data: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    auto_retrain: bool = True,
) -> Dict[str, Any]:
    """Execute the full self-healing pipeline.

    1. Run orchestrator to assess pipeline health
    2. If action is 'retrain', trigger automatic retraining
    3. Re-evaluate the new model via gate agent
    4. Return full audit trail
    """
    logger.info("═══ Self-Healing Loop triggered ═══")

    # Step 1: Orchestrator assessment
    orch_report = run_orchestrator(
        metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        reference_data=reference_data,
        current_data=current_data,
        feature_names=feature_names,
    )

    action = orch_report["verdict"]["action"]
    audit = {
        "trigger": "self_healing",
        "initial_assessment": orch_report["verdict"],
        "action_taken": action,
        "retrained": False,
        "new_metrics": None,
        "new_gate_decision": None,
    }

    # Step 2: Auto retrain if needed
    if action in ("retrain", "fix_model") and auto_retrain:
        logger.info("▸ Triggering automatic retraining …")
        try:
            retrain_result = _retrain_pipeline()
            audit["retrained"] = True
            audit["new_metrics"] = retrain_result["metrics"]
            audit["new_gate_decision"] = retrain_result["gate_decision"]

            # Update Prometheus gauges
            m = retrain_result["metrics"]
            update_model_metrics(m.get("accuracy", 0), m.get("f1", 0))
            update_gate_decision(retrain_result["gate_decision"]["decision"] == "APPROVE")

        except Exception as exc:
            logger.error("Retraining failed: %s", exc)
            audit["retrain_error"] = str(exc)

    elif action == "rollback":
        logger.warning("▸ Rollback recommended — manual intervention required")
        audit["action_taken"] = "rollback_recommended"

    elif action == "deploy":
        logger.info("▸ Pipeline healthy — no action needed")

    logger.info("═══ Self-Healing Loop complete ═══")
    logger.info("Audit trail: %s", json.dumps(audit, indent=2, default=str))
    return audit


def _retrain_pipeline() -> Dict[str, Any]:
    """Re-run the full training pipeline and evaluate."""
    from src.agents.gate_agent import gate_decision_llm

    # Load and process data
    df = load_raw_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, scaler, feat_names = split_and_scale(df)

    # Train new model
    model, metrics = train_model(
        X_train, y_train, X_test, y_test,
        track_mlflow=True,
    )

    # Save
    save_model(model, "model")

    # Gate evaluation on new model
    gate_decision = gate_decision_llm(metrics)

    return {
        "metrics": metrics,
        "gate_decision": gate_decision,
        "feature_names": feat_names,
    }
