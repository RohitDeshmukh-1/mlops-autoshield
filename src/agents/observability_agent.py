"""Phase 3 — Observability Agent.

Monitors model performance metrics and generates alerts with LLM summaries.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from src.config import AGENT_CFG, GROQ_API_KEY

logger = logging.getLogger(__name__)


def check_performance_degradation(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    threshold: float | None = None,
) -> Dict[str, Any]:
    """Compare current vs baseline metrics and flag degradation."""
    threshold = threshold or AGENT_CFG["observability_agent"]["alert_threshold"]
    alerts = []

    higher_is_better = {"accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"}
    lower_is_better = {"avg_latency_ms"}

    for metric, baseline_val in baseline_metrics.items():
        current_val = current_metrics.get(metric)
        if current_val is None or baseline_val == 0:
            continue

        if metric in higher_is_better:
            degradation = (baseline_val - current_val) / baseline_val
            if degradation > threshold:
                alerts.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "degradation_pct": round(degradation * 100, 2),
                    "direction": "decreased",
                })
        elif metric in lower_is_better:
            increase = (current_val - baseline_val) / max(baseline_val, 1e-6)
            if increase > threshold:
                alerts.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "increase_pct": round(increase * 100, 2),
                    "direction": "increased",
                })

    return {
        "alert_triggered": len(alerts) > 0,
        "alert_count": len(alerts),
        "alerts": alerts,
        "threshold_pct": threshold * 100,
    }


def observability_summary_llm(
    alert_report: Dict[str, Any],
    drift_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Use LLM to produce a human-friendly observability summary."""
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            return _call_groq_obs(alert_report, drift_report)
        except Exception as exc:
            logger.warning("LLM observability call failed (%s), using rules", exc)

    return observability_summary_rules(alert_report)


def _call_groq_obs(
    alert_report: Dict[str, Any],
    drift_report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Call Groq for observability summary."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    system = (
        "You are an ML observability agent. Given alert and drift reports, produce "
        "a concise incident summary with severity, root cause hypothesis, and "
        'recommended action. Return JSON: {"severity": "...", "summary": "...", '
        '"root_cause": "...", "action": "..."}'
    )

    context = {"alerts": alert_report}
    if drift_report:
        context["drift"] = drift_report

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=AGENT_CFG["llm"]["model"],
        temperature=AGENT_CFG["llm"]["temperature"],
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(context, indent=2)),
    ])

    text = response.content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)


def observability_summary_rules(alert_report: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based observability summary fallback."""
    if not alert_report["alert_triggered"]:
        return {
            "severity": "OK",
            "summary": "All metrics within acceptable range.",
            "root_cause": "N/A",
            "action": "continue_monitoring",
        }

    count = alert_report["alert_count"]
    severity = "WARNING" if count <= 2 else "CRITICAL"
    degraded = [a["metric"] for a in alert_report["alerts"]]

    return {
        "severity": severity,
        "summary": f"{count} metric(s) degraded: {', '.join(degraded)}.",
        "root_cause": "Possible data drift or model staleness.",
        "action": "retrain" if severity == "CRITICAL" else "investigate",
    }
