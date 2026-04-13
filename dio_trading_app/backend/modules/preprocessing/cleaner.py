"""
Dio Trading App — Preprocessing Module
========================================
Cleans, normalises, and validates raw OHLCV data before
feature engineering and model training.
"""

import numpy as np
import pandas as pd
from loguru import logger


def remove_outliers(df: pd.DataFrame, col: str = "returns", z_thresh: float = 5.0) -> pd.DataFrame:
    """Remove rows where a column's z-score exceeds threshold."""
    z = (df[col] - df[col].mean()) / df[col].std()
    mask = z.abs() <= z_thresh
    removed = (~mask).sum()
    if removed > 0:
        logger.debug(f"Removed {removed} outlier rows in '{col}'")
    return df[mask]


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill small gaps (up to 3 missing bars) in OHLCV data.
    Larger gaps are left as NaN and will be dropped during feature engineering.
    """
    df = df.copy()
    ohlc = ["open", "high", "low", "close"]
    df[ohlc] = df[ohlc].ffill(limit=3)
    # Volume proxy: fill gaps with rolling median
    df["volume_proxy"] = df["volume_proxy"].fillna(
        df["volume_proxy"].rolling(10, min_periods=1).median()
    )
    return df


def normalise_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise volume_proxy using a rolling 50-bar z-score.
    This makes volume comparable across different market periods.
    """
    roll_mean = df["volume_proxy"].rolling(50, min_periods=10).mean()
    roll_std = df["volume_proxy"].rolling(50, min_periods=10).std()
    df["volume_proxy_norm"] = (df["volume_proxy"] - roll_mean) / roll_std.replace(0, np.nan)
    return df


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Assert OHLC data integrity and drop bad rows."""
    valid = (
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"]) &
        (df["high"] >= df["low"]) &
        (df["close"] > 0)
    )
    bad = (~valid).sum()
    if bad > 0:
        logger.warning(f"Dropping {bad} rows with invalid OHLC relationships")
    return df[valid]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    logger.info(f"Preprocessing {len(df)} bars...")
    df = validate_ohlc(df)
    df = fill_gaps(df)
    df = remove_outliers(df, col="returns")
    df = normalise_volume(df)
    logger.success(f"Preprocessing complete: {len(df)} bars remaining")
    return df
