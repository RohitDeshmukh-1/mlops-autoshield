"""Phase 4 — Evidently AI drift reports.

Generates HTML drift reports and data quality summaries using Evidently.
Falls back to a custom report if Evidently is unavailable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import DRIFT_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate an Evidently data drift HTML report.

    Falls back to a JSON summary if Evidently is not installed.
    """
    output_dir = Path(output_dir or PROJECT_ROOT / DRIFT_CFG["report_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        html_path = output_dir / f"drift_report_{timestamp}.html"
        report.save_html(str(html_path))
        logger.info("Evidently drift report saved → %s", html_path)
        return html_path

    except ImportError:
        logger.warning("Evidently not installed, generating JSON fallback report")
        return _fallback_report(reference_df, current_df, output_dir, timestamp)


def _fallback_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
) -> Path:
    """Simple JSON drift summary when Evidently is unavailable."""
    from src.agents.drift_agent import compute_psi, compute_ks_test

    report = {"generated_at": timestamp, "features": []}

    numeric_cols = reference_df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        psi = compute_psi(ref, cur)
        ks = compute_ks_test(ref, cur)
        report["features"].append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_statistic": round(ks["statistic"], 4),
            "ks_p_value": round(ks["p_value"], 4),
        })

    json_path = output_dir / f"drift_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Fallback drift report saved → %s", json_path)
    return json_path
