"""Tests for configuration loading."""
from __future__ import annotations

import pytest
from src.config import load_config, DATA_CFG, MODEL_CFG, AGENT_CFG


class TestConfig:
    """Tests for config module."""

    def test_config_loads(self):
        """Config should load without errors."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "model" in cfg
        assert "agents" in cfg

    def test_data_config(self):
        """Data config should have required keys."""
        assert "raw_path" in DATA_CFG
        assert "processed_path" in DATA_CFG
        assert "target_column" in DATA_CFG
        assert "test_size" in DATA_CFG

    def test_model_config(self):
        """Model config should have type and params."""
        assert "type" in MODEL_CFG
        assert "params" in MODEL_CFG
        assert MODEL_CFG["type"] in ("xgboost", "random_forest", "logistic_regression")

    def test_agent_config(self):
        """Agent config should have LLM and gate settings."""
        assert "llm" in AGENT_CFG
        assert "gate_agent" in AGENT_CFG
        assert "drift_agent" in AGENT_CFG
        assert AGENT_CFG["gate_agent"]["min_accuracy"] > 0
