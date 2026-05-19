"""Shared configuration loader for AutoShield pipeline."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Project root — two levels up from src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    """Load the YAML config from *path* (defaults to configs/config.yaml)."""
    path = path or CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


_cfg = load_config()

# ── Convenience accessors ──────────────────────────────────────────
DATA_CFG = _cfg["data"]
FEATURE_CFG = _cfg["features"]
MODEL_CFG = _cfg["model"]
MLFLOW_CFG = _cfg["mlflow"]
AGENT_CFG = _cfg["agents"]
MONITOR_CFG = _cfg["monitoring"]
API_CFG = _cfg["api"]
DRIFT_CFG = _cfg["drift"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
