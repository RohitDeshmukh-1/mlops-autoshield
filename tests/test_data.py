"""Tests for Phase 1 — Data Layer (data loading, preprocessing, splitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    load_raw_data,
    preprocess,
    split_and_scale,
    _generate_synthetic_dataset,
    get_scaler,
)


class TestDataLoading:
    """Tests for data loading and synthetic generation."""

    def test_synthetic_generation(self, tmp_path):
        """Synthetic dataset should have correct schema and fraud rate."""
        path = tmp_path / "test.csv"
        df = _generate_synthetic_dataset(path, n_samples=1000)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert "Class" in df.columns
        assert "Amount" in df.columns
        assert "Time" in df.columns

        # Should have V1..V28
        v_cols = [c for c in df.columns if c.startswith("V")]
        assert len(v_cols) == 28

        # Fraud rate should be roughly 2%
        fraud_rate = df["Class"].mean()
        assert 0.0 < fraud_rate < 0.10  # generous bounds

        # File should be saved
        assert path.exists()

    def test_load_raw_data_generates_if_missing(self, tmp_path):
        """load_raw_data should auto-generate when file is missing."""
        # Override config path temporarily
        import src.config as cfg
        original = cfg.DATA_CFG["raw_path"]
        cfg.DATA_CFG["raw_path"] = str(tmp_path / "raw")
        try:
            df = load_raw_data(tmp_path / "raw" / "creditcard.csv")
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        finally:
            cfg.DATA_CFG["raw_path"] = original


class TestPreprocessing:
    """Tests for feature engineering pipeline."""

    def test_preprocess_adds_features(self, synthetic_df):
        """Preprocessing should add Amount_log and Hour columns."""
        df = preprocess(synthetic_df)
        assert "Amount_log" in df.columns
        assert "Hour" in df.columns
        assert df["Hour"].min() >= 0
        assert df["Hour"].max() <= 23

    def test_preprocess_no_nulls(self, synthetic_df):
        """Output should have no null values."""
        # Inject some nulls
        synthetic_df.loc[0, "V1"] = np.nan
        synthetic_df.loc[5, "Amount"] = np.nan
        df = preprocess(synthetic_df)
        assert not df.isnull().any().any()

    def test_preprocess_drops_duplicates(self, synthetic_df):
        """Duplicate rows should be removed."""
        # Add duplicate rows
        df_with_dups = pd.concat([synthetic_df, synthetic_df.iloc[:10]], ignore_index=True)
        df = preprocess(df_with_dups)
        assert len(df) <= len(synthetic_df)

    def test_preprocess_preserves_target(self, synthetic_df):
        """Target column should be preserved after preprocessing."""
        df = preprocess(synthetic_df)
        assert "Class" in df.columns
        assert set(df["Class"].unique()).issubset({0, 1})


class TestScaling:
    """Tests for scalers and train/test splitting."""

    def test_get_scaler_standard(self):
        from sklearn.preprocessing import StandardScaler
        scaler = get_scaler("standard")
        assert isinstance(scaler, StandardScaler)

    def test_get_scaler_minmax(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = get_scaler("minmax")
        assert isinstance(scaler, MinMaxScaler)

    def test_get_scaler_robust(self):
        from sklearn.preprocessing import RobustScaler
        scaler = get_scaler("robust")
        assert isinstance(scaler, RobustScaler)

    def test_split_and_scale(self, synthetic_df):
        """Split should produce correct shapes and scaled values."""
        df = preprocess(synthetic_df)
        X_train, X_test, y_train, y_test, scaler, feat_names = split_and_scale(df)

        # Shapes
        assert X_train.shape[0] + X_test.shape[0] == len(df)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

        # Feature names
        assert isinstance(feat_names, list)
        assert len(feat_names) == X_train.shape[1]

        # Scaled features should be roughly centered
        assert abs(X_train.mean()) < 1.0  # after StandardScaler, close to 0

    def test_split_stratification(self, synthetic_df):
        """Train and test sets should have similar fraud rates."""
        df = preprocess(synthetic_df)
        _, _, y_train, y_test, _, _ = split_and_scale(df)

        train_rate = y_train.mean()
        test_rate = y_test.mean()

        # Should be within 2 percentage points
        assert abs(train_rate - test_rate) < 0.02
