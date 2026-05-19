"""Phase 3 — GenAI CI/CD Gate Agent.

Uses Llama 3.3 70B (via Groq) to evaluate model metrics and make
approve / block decisions for production deployments.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from src.config import AGENT_CFG, GROQ_API_KEY

logger = logging.getLogger(__name__)

GATE_SYSTEM_PROMPT = """You are AutoShield Gate Agent — an expert ML deployment reviewer.

Your job is to evaluate a candidate model's test metrics and decide whether it
should be promoted to production.

**Decision criteria (thresholds):**
- Accuracy  >= {min_accuracy}
- Precision >= {min_precision}
- Recall    >= {min_recall}
- F1 Score  >= {min_f1}
- Avg inference latency <= {max_latency_ms} ms

**Rules:**
1. If ALL thresholds are met -> output APPROVE with a short justification.
2. If ANY threshold is violated -> output BLOCK, list every failed metric,
   and suggest concrete remediation steps.
3. Always return valid JSON with this schema:
   {{"decision": "APPROVE" | "BLOCK", "failed_metrics": [...], "rationale": "...", "recommendations": [...]}}
"""

GATE_USER_TEMPLATE = """Evaluate the following model metrics for production readiness:

```json
{metrics_json}
```

Return your decision as JSON.
"""


def _build_gate_prompt(metrics: Dict[str, float]) -> tuple[str, str]:
    """Build system + user prompts for the gate agent."""
    gate_cfg = AGENT_CFG["gate_agent"]
    system = GATE_SYSTEM_PROMPT.format(**gate_cfg)
    user = GATE_USER_TEMPLATE.format(metrics_json=json.dumps(metrics, indent=2))
    return system, user


def gate_decision_llm(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Call the LLM to get an approve/block decision.

    Falls back to rule-based logic if Groq is unavailable.
    """
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            return _call_groq(metrics)
        except Exception as exc:
            logger.warning("LLM gate call failed (%s), falling back to rules", exc)

    return gate_decision_rules(metrics)


def _call_groq(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Call Groq API with Llama 3.3 70B."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=AGENT_CFG["llm"]["model"],
        temperature=AGENT_CFG["llm"]["temperature"],
        max_tokens=AGENT_CFG["llm"]["max_tokens"],
    )

    system, user = _build_gate_prompt(metrics)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    text = response.content.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    result = json.loads(text)
    logger.info("LLM gate decision: %s", result["decision"])
    return result


def gate_decision_rules(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Deterministic rule-based fallback for the gate agent."""
    gate_cfg = AGENT_CFG["gate_agent"]
    failed = []

    checks = {
        "accuracy": ("min_accuracy", ">="),
        "precision": ("min_precision", ">="),
        "recall": ("min_recall", ">="),
        "f1": ("min_f1", ">="),
        "avg_latency_ms": ("max_latency_ms", "<="),
    }

    for metric_key, (cfg_key, op) in checks.items():
        val = metrics.get(metric_key)
        threshold = gate_cfg[cfg_key]
        if val is None:
            continue
        if op == ">=" and val < threshold:
            failed.append({
                "metric": metric_key, "value": val,
                "threshold": threshold, "required": f">= {threshold}",
            })
        elif op == "<=" and val > threshold:
            failed.append({
                "metric": metric_key, "value": val,
                "threshold": threshold, "required": f"<= {threshold}",
            })

    if not failed:
        return {
            "decision": "APPROVE",
            "failed_metrics": [],
            "rationale": "All metrics meet production thresholds.",
            "recommendations": [],
        }

    recommendations = []
    for f in failed:
        if f["metric"] == "recall":
            recommendations.append("Increase recall by adjusting class weights or oversampling.")
        elif f["metric"] == "precision":
            recommendations.append("Improve precision by tuning decision threshold.")
        elif f["metric"] == "avg_latency_ms":
            recommendations.append("Reduce latency by simplifying model or feature selection.")
        else:
            recommendations.append(
                f"Improve {f['metric']} (current: {f['value']:.4f}, required: {f['required']})."
            )

    return {
        "decision": "BLOCK",
        "failed_metrics": failed,
        "rationale": f"{len(failed)} metric(s) below production threshold.",
        "recommendations": recommendations,
    }
