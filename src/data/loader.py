"""Phase 1 — Data ingestion & preprocessing.

Loads the UCI Credit Card Fraud dataset, applies feature engineering,
and produces train/test splits versioned via DVC.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.config import DATA_CFG, FEATURE_CFG, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ── Data loading ────────────────────────────────────────────────────
def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset.

    If no local file exists, generate a synthetic credit-card-fraud-like
    dataset for demo purposes.
    """
    raw_dir = PROJECT_ROOT / DATA_CFG["raw_path"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = path or raw_dir / "creditcard.csv"

    if Path(csv_path).exists():
        logger.info("Loading raw data from %s", csv_path)
        return pd.read_csv(csv_path)

    logger.info("Raw file not found — generating synthetic dataset")
    return _generate_synthetic_dataset(csv_path)


def _generate_synthetic_dataset(save_path: Path, n_samples: int = 10_000) -> pd.DataFrame:
    """Create a synthetic fraud dataset mimicking the UCI schema."""
    rng = np.random.RandomState(DATA_CFG.get("random_state", 42))

    # 28 PCA-like features + Time + Amount
    feature_cols = [f"V{i}" for i in range(1, 29)]
    data = {col: rng.randn(n_samples) for col in feature_cols}
    data["Time"] = np.sort(rng.uniform(0, 172_792, n_samples))
    data["Amount"] = np.abs(rng.lognormal(3, 2, n_samples))

    # ~2% fraud (imbalanced)
    fraud_mask = rng.rand(n_samples) < 0.02
    data["Class"] = fraud_mask.astype(int)

    # Make fraud rows slightly distinct (shift means)
    for col in feature_cols[:5]:
        data[col] = np.where(fraud_mask, data[col] + rng.uniform(1, 3), data[col])

    df = pd.DataFrame(data)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info("Synthetic dataset saved → %s (%d rows, %d fraud)", save_path, len(df), fraud_mask.sum())
    return df


# ── Preprocessing / Feature pipeline ───────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw data."""
    df = df.copy()

    # Drop duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < n_before:
        logger.info("Dropped %d duplicate rows", n_before - len(df))

    # Handle missing values (fill numeric with median)
    if df.isnull().any().any():
        for col in df.select_dtypes(include="number").columns:
            df[col].fillna(df[col].median(), inplace=True)
        logger.info("Imputed missing values with median")

    # Log-transform Amount (heavy right skew)
    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])

    # Time features
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] / 3600).astype(int) % 24

    return df


def get_scaler(name: str = "standard"):
    """Return a scaler instance by name."""
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    return scalers.get(name, StandardScaler)()


def split_and_scale(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, list]:
    """Split into train/test and scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    target_col = target_col or DATA_CFG["target_column"]
    test_size = test_size or DATA_CFG["test_size"]
    random_state = DATA_CFG.get("random_state", 42)

    # Select feature columns
    feature_cols = [c for c in FEATURE_CFG["numeric_features"] if c in df.columns]
    # Add engineered features if present
    for extra in ["Amount_log", "Hour"]:
        if extra in df.columns and extra not in feature_cols:
            feature_cols.append(extra)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = get_scaler(FEATURE_CFG.get("scaler", "standard"))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        "Split complete — train=%d, test=%d, features=%d, fraud_rate=%.2f%%",
        len(X_train), len(X_test), X_train.shape[1],
        100 * y_train.mean(),
    )
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_processed_data(
    df: pd.DataFrame, name: str = "processed"
) -> Path:
    """Save processed dataframe to data/processed/."""
    out_dir = PROJECT_ROOT / DATA_CFG["processed_path"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Processed data saved → %s", out_path)
    return out_path
