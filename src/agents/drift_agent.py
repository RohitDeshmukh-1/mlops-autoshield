"""Phase 3 — Drift Analysis Agent.

Detects data drift using PSI and KS-test, with LLM-powered interpretation.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from src.config import AGENT_CFG, DRIFT_CFG, GROQ_API_KEY

logger = logging.getLogger(__name__)


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-4
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_hist / max(len(reference), 1) + eps
    cur_pct = cur_hist / max(len(current), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test."""
    stat, p_value = stats.ks_2samp(reference, current)
    return {"statistic": float(stat), "p_value": float(p_value)}


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Run PSI + KS-test per feature and return drift report."""
    psi_threshold = DRIFT_CFG.get("psi_threshold", 0.2)
    ks_alpha = DRIFT_CFG.get("ks_alpha", 0.05)

    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
        current = current.reshape(-1, 1)

    n_features = reference.shape[1]
    feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

    results = []
    drifted_features = []

    for i in range(n_features):
        ref_col = reference[:, i]
        cur_col = current[:, i]

        psi = compute_psi(ref_col, cur_col)
        ks = compute_ks_test(ref_col, cur_col)

        is_drifted = psi > psi_threshold or ks["p_value"] < ks_alpha
        result = {
            "feature": feature_names[i],
            "psi": round(psi, 4),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
            "drifted": is_drifted,
        }
        results.append(result)
        if is_drifted:
            drifted_features.append(feature_names[i])

    report = {
        "total_features": n_features,
        "drifted_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
        "details": results,
    }

    logger.info("Drift report: %d/%d features drifted", len(drifted_features), n_features)
    return report


def drift_analysis_llm(drift_report: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to interpret drift report and suggest actions."""
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            return _call_groq_drift(drift_report)
        except Exception as exc:
            logger.warning("LLM drift analysis failed (%s), using rules", exc)

    return drift_analysis_rules(drift_report)


def _call_groq_drift(drift_report: Dict[str, Any]) -> Dict[str, Any]:
    """Call Groq for drift interpretation."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    system = (
        "You are a data drift analyst. Given a drift report with PSI and KS-test "
        "results per feature, assess the severity (LOW/MEDIUM/HIGH/CRITICAL), "
        "explain root causes, and recommend actions. Return JSON: "
        '{"severity": "...", "summary": "...", "action": "retrain|monitor|rollback", '
        '"explanation": "..."}'
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=AGENT_CFG["llm"]["model"],
        temperature=AGENT_CFG["llm"]["temperature"],
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(drift_report, indent=2)),
    ])

    text = response.content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


def drift_analysis_rules(drift_report: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based drift interpretation fallback."""
    ratio = drift_report["drifted_count"] / max(drift_report["total_features"], 1)

    if ratio == 0:
        return {"severity": "LOW", "summary": "No drift detected.", "action": "monitor", "explanation": "All features stable."}
    elif ratio < 0.2:
        return {"severity": "MEDIUM", "summary": f"{drift_report['drifted_count']} features drifted.", "action": "monitor", "explanation": "Minor drift, continue monitoring."}
    elif ratio < 0.5:
        return {"severity": "HIGH", "summary": f"{drift_report['drifted_count']} features drifted significantly.", "action": "retrain", "explanation": "Significant drift detected; retraining recommended."}
    else:
        return {"severity": "CRITICAL", "summary": f"Majority of features drifted ({drift_report['drifted_count']}/{drift_report['total_features']}).", "action": "rollback", "explanation": "Critical drift — rollback or immediate retrain."}
