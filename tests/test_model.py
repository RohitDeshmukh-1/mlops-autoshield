"""Tests for Phase 2 — Model Training & Evaluation."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import preprocess, split_and_scale
from src.models.trainer import build_model, evaluate_model, train_model, save_model, load_model


class TestModelFactory:
    """Tests for model instantiation."""

    def test_build_xgboost(self):
        from xgboost import XGBClassifier
        model = build_model("xgboost")
        assert isinstance(model, XGBClassifier)

    def test_build_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        model = build_model("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_build_logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        model = build_model("logistic_regression")
        assert isinstance(model, LogisticRegression)

    def test_build_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_model("neural_network")


class TestModelTraining:
    """Tests for training and evaluation."""

    def test_train_xgboost(self, synthetic_df):
        """XGBoost should train and produce valid metrics."""
        df = preprocess(synthetic_df)
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)

        model, metrics = train_model(
            X_train, y_train, X_test, y_test,
            model_type="xgboost",
            track_mlflow=False,
        )

        assert model is not None
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "avg_latency_ms" in metrics

        # Metrics should be valid numbers
        for k, v in metrics.items():
            assert isinstance(v, float)
            assert 0 <= v or k == "avg_latency_ms"  # latency can be any positive number

    def test_train_random_forest(self, synthetic_df):
        """Random Forest should train successfully."""
        df = preprocess(synthetic_df)
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)

        model, metrics = train_model(
            X_train, y_train, X_test, y_test,
            model_type="random_forest",
            track_mlflow=False,
        )

        assert model is not None
        assert metrics["accuracy"] > 0.5  # better than random

    def test_evaluate_model(self, synthetic_df):
        """Evaluation should produce all expected metric keys."""
        df = preprocess(synthetic_df)
        X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)

        model = build_model("xgboost")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        expected_keys = {"accuracy", "precision", "recall", "f1", "avg_latency_ms", "roc_auc", "avg_precision"}
        assert expected_keys.issubset(set(metrics.keys()))


class TestModelPersistence:
    """Tests for model save/load."""

    def test_save_and_load(self, synthetic_df, tmp_path):
        """Model should be saveable and loadable."""
        import src.config as cfg
        original = cfg.MODEL_CFG["save_path"]
        cfg.MODEL_CFG["save_path"] = str(tmp_path)

        try:
            df = preprocess(synthetic_df)
            X_train, X_test, y_train, y_test, _, _ = split_and_scale(df)
            model, _ = train_model(X_train, y_train, X_test, y_test, track_mlflow=False)

            save_model(model, "test_model")
            loaded = load_model("test_model")

            # Loaded model should produce same predictions
            preds_original = model.predict(X_test[:5])
            preds_loaded = loaded.predict(X_test[:5])
            np.testing.assert_array_equal(preds_original, preds_loaded)
        finally:
            cfg.MODEL_CFG["save_path"] = original

    def test_load_missing_model(self, tmp_path):
        """Loading a non-existent model should raise FileNotFoundError."""
        import src.config as cfg
        original = cfg.MODEL_CFG["save_path"]
        cfg.MODEL_CFG["save_path"] = str(tmp_path)
        try:
            with pytest.raises(FileNotFoundError):
                load_model("nonexistent_model")
        finally:
            cfg.MODEL_CFG["save_path"] = original
