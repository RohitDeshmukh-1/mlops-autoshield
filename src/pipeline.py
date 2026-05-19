"""Main pipeline runner — executes all 5 phases end-to-end."""
from __future__ import annotations

import json
import logging
import sys

from src.data.loader import load_raw_data, preprocess, split_and_scale, save_processed_data
from src.models.trainer import train_model, save_model
from src.agents.orchestrator import run_orchestrator
from src.self_healing import self_healing_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autoshield.pipeline")


def run_pipeline():
    """Execute the full MLOps AutoShield pipeline."""
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║    MLOps AutoShield — Full Pipeline      ║")
    logger.info("╚══════════════════════════════════════════╝")

    # ── Phase 1: Data Layer ──────────────────────────────────────
    logger.info("\n📦 Phase 1 — Data Layer & Baseline Model")
    df = load_raw_data()
    df = preprocess(df)
    save_processed_data(df, "processed")

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    # ── Phase 2: Training + MLflow ───────────────────────────────
    logger.info("\n🧪 Phase 2 — Experiment Tracking & Model Training")
    model, metrics = train_model(X_train, y_train, X_test, y_test, track_mlflow=True)
    save_model(model, "model")

    # Save scaler alongside model
    import joblib
    from src.config import PROJECT_ROOT, MODEL_CFG
    scaler_path = PROJECT_ROOT / MODEL_CFG["save_path"] / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    logger.info("Baseline metrics: %s", json.dumps(metrics, indent=2))

    # ── Phase 3: GenAI Agents ────────────────────────────────────
    logger.info("\n🤖 Phase 3 — GenAI CI/CD Agents")
    orch_report = run_orchestrator(
        metrics=metrics,
        baseline_metrics=metrics,
        reference_data=X_train[:500],
        current_data=X_test[:500],
        feature_names=feature_names,
    )
    logger.info("Orchestrator verdict: %s", json.dumps(orch_report["verdict"], indent=2))

    # ── Phase 4: Monitoring ──────────────────────────────────────
    logger.info("\n📊 Phase 4 — Monitoring & Observability")
    from src.monitoring.metrics import update_model_metrics, update_gate_decision, get_metrics_text
    update_model_metrics(metrics["accuracy"], metrics["f1"])
    update_gate_decision(orch_report["gate"]["decision"] == "APPROVE")
    logger.info("Prometheus metrics registered ✓")

    # ── Phase 5: Self-Healing Demo ───────────────────────────────
    logger.info("\n🔄 Phase 5 — Self-Healing Loop (simulated)")
    # Simulate degraded metrics to trigger self-healing
    degraded_metrics = {k: v * 0.85 for k, v in metrics.items() if isinstance(v, float)}
    healing_audit = self_healing_loop(
        current_metrics=degraded_metrics,
        baseline_metrics=metrics,
        reference_data=X_train[:500],
        current_data=X_test[:500],
        feature_names=feature_names,
        auto_retrain=True,
    )

    logger.info("\n✅ Pipeline complete!")
    logger.info("Final audit: %s", json.dumps(healing_audit, indent=2, default=str))
    return {
        "metrics": metrics,
        "orchestrator": orch_report,
        "healing_audit": healing_audit,
    }


if __name__ == "__main__":
    run_pipeline()
