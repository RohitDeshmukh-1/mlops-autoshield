"""Phase 4 — FastAPI model-serving API.

Endpoints:
  POST /predict        → real-time inference
  GET  /health         → liveness check
  GET  /metrics        → Prometheus exposition format
  POST /drift/check    → run drift detection
  POST /gate/evaluate  → run gate agent on metrics
  POST /orchestrate    → run full agent pipeline
"""
from __future__ import annotations

import time
import logging
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from src.monitoring.metrics import (
    record_prediction,
    get_metrics_text,
    update_model_metrics,
    update_drift_metrics,
    update_gate_decision,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoShield API",
    description="Self-healing MLOps model serving with GenAI agents",
    version="1.0.0",
)

# ── Global state ──────────────────────────────────────────────────
_model = None
_scaler = None
_feature_names: list[str] = []
_baseline_metrics: Dict[str, float] = {}


# ── Request / Response schemas ────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[List[float]]  # batch of feature vectors


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    latency_ms: float


class MetricsInput(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_latency_ms: float = 0.0
    roc_auc: float | None = None


class DriftCheckRequest(BaseModel):
    reference: List[List[float]]
    current: List[List[float]]
    feature_names: List[str] | None = None


# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
async def load_model_on_startup():
    """Load model and scaler at startup."""
    global _model, _scaler, _feature_names, _baseline_metrics
    try:
        from src.models.trainer import load_model
        _model = load_model("model")
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("No model found at startup — train one first")

    try:
        import joblib
        from src.config import PROJECT_ROOT, MODEL_CFG
        scaler_path = PROJECT_ROOT / MODEL_CFG["save_path"] / "scaler.joblib"
        if scaler_path.exists():
            _scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded")
    except Exception:
        logger.warning("No scaler found")


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(req.features)
    if _scaler is not None:
        X = _scaler.transform(X)

    start = time.perf_counter()
    preds = _model.predict(X).tolist()
    probas = _model.predict_proba(X)[:, 1].tolist() if hasattr(_model, "predict_proba") else [0.0] * len(preds)
    latency = time.perf_counter() - start

    # Record metrics
    for p in preds:
        record_prediction(latency / max(len(preds), 1), bool(p))

    return PredictResponse(
        predictions=preds,
        probabilities=[round(p, 4) for p in probas],
        latency_ms=round(latency * 1000, 2),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return get_metrics_text()


@app.post("/gate/evaluate")
async def gate_evaluate(req: MetricsInput):
    from src.agents.gate_agent import gate_decision_llm
    metrics = req.model_dump(exclude_none=True)
    result = gate_decision_llm(metrics)
    update_gate_decision(result["decision"] == "APPROVE")
    return result


@app.post("/drift/check")
async def drift_check(req: DriftCheckRequest):
    from src.agents.drift_agent import detect_drift, drift_analysis_llm
    ref = np.array(req.reference)
    cur = np.array(req.current)
    report = detect_drift(ref, cur, req.feature_names)
    analysis = drift_analysis_llm(report)

    if report["drifted_count"] > 0:
        max_psi = max(d["psi"] for d in report["details"])
        update_drift_metrics(max_psi, report["drifted_count"])

    return {"report": report, "analysis": analysis}


@app.post("/orchestrate")
async def orchestrate(req: MetricsInput):
    from src.agents.orchestrator import run_orchestrator
    metrics = req.model_dump(exclude_none=True)
    result = run_orchestrator(
        metrics=metrics,
        baseline_metrics=_baseline_metrics or metrics,
    )
    return result
