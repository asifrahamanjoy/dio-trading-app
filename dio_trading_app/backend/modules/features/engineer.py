"""
Dio Trading App — Feature Engineering Module
==============================================
Constructs higher-level features for the ML model by combining
signals from technical, liquidity, and event modules.

These engineered features are specifically designed to capture
multi-condition setups that the strategy engine looks for.
"""

import numpy as np
import pandas as pd
from loguru import logger


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode candlestick pattern context as numeric features."""
    # Candle body relative to ATR
    df["body_atr_ratio"] = df["body"] / df["atr"].replace(0, np.nan)
    df["wick_ratio"] = (df["upper_wick"] + df["lower_wick"]) / df["body"].replace(0, np.nan)

    # Doji: body very small relative to range
    df["doji"] = df["body"] < df["hl_range"] * 0.1

    # Strong candle: body > 60% of range
    df["strong_bull_candle"] = (df["close"] > df["open"]) & (df["body"] > df["hl_range"] * 0.6)
    df["strong_bear_candle"] = (df["close"] < df["open"]) & (df["body"] > df["hl_range"] * 0.6)

    return df


def add_multi_bar_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling context features (past N bars)."""
    for n in [3, 5, 10]:
        df[f"close_above_n{n}_ago"] = df["close"] > df["close"].shift(n)
        df[f"return_{n}bar"] = df["close"].pct_change(n)
        df[f"vol_avg_{n}bar"] = df["volume_proxy"].rolling(n).mean()
        df[f"range_avg_{n}bar"] = df["hl_range"].rolling(n).mean()

    # Consecutive bullish/bearish bars
    bull_streak = (df["close"] > df["open"]).astype(int)
    df["bull_streak_3"] = bull_streak.rolling(3).sum() == 3
    df["bear_streak_3"] = bull_streak.rolling(3).sum() == 0

    # RSI momentum (rate of change of RSI)
    if "rsi_14" in df.columns:
        df["rsi_roc_3"] = df["rsi_14"] - df["rsi_14"].shift(3)
        df["rsi_roc_5"] = df["rsi_14"] - df["rsi_14"].shift(5)

    # ATR expansion/contraction
    if "atr" in df.columns:
        atr_sma = df["atr"].rolling(20).mean()
        df["atr_ratio"] = df["atr"] / atr_sma.replace(0, np.nan)

    # Price distance from key MAs (normalized by ATR)
    for ma in ["ema_9", "ema_21", "ema_50"]:
        if ma in df.columns and "atr" in df.columns:
            df[f"dist_{ma}_atr"] = (df["close"] - df[ma]) / df["atr"].replace(0, np.nan)

    return df


def add_confluence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute a fast confluence score for BUY and SELL setups.
    This speeds up the signal engine by avoiding repeated condition checks.
    """
    # BUY confluence
    buy_score = (
        df.get("ema_bull_align", pd.Series(False, index=df.index)).astype(int) * 3 +
        (df.get("rsi_14", 50).between(35, 65)).astype(int) * 2 +
        (df.get("macd_line", 0) > df.get("macd_signal", 0)).astype(int) * 2 +
        df.get("above_sma200", pd.Series(False, index=df.index)).astype(int) * 2 +
        df.get("uptrend_structure", pd.Series(False, index=df.index)).astype(int) * 1 +
        (df.get("vol_condition", "") == "price_up_vol_up").astype(int) * 2 +
        df.get("bullish_sweep", pd.Series(False, index=df.index)).astype(int) * 2 +
        (df.get("liquidity_zone", "") == "high").astype(int) * 1 +
        (~df.get("event_window", pd.Series(False, index=df.index))).astype(int) * 1
    )

    # SELL confluence
    sell_score = (
        df.get("ema_bear_align", pd.Series(False, index=df.index)).astype(int) * 3 +
        (df.get("rsi_14", 50).between(35, 65)).astype(int) * 2 +
        (df.get("macd_line", 0) < df.get("macd_signal", 0)).astype(int) * 2 +
        (~df.get("above_sma200", pd.Series(True, index=df.index))).astype(int) * 2 +
        df.get("downtrend_structure", pd.Series(False, index=df.index)).astype(int) * 1 +
        (df.get("vol_condition", "") == "price_down_vol_up").astype(int) * 2 +
        df.get("bearish_sweep", pd.Series(False, index=df.index)).astype(int) * 2 +
        (df.get("liquidity_zone", "") == "high").astype(int) * 1 +
        (~df.get("event_window", pd.Series(False, index=df.index))).astype(int) * 1
    )

    df["buy_confluence"] = buy_score
    df["sell_confluence"] = sell_score

    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify the current market regime.
    Regime affects signal filtering — ranging markets suppress trend signals.
    """
    # ADX-like trend strength using EMA spread
    if "ema_21" in df.columns and "ema_50" in df.columns:
        ema_spread = (df["ema_21"] - df["ema_50"]).abs() / df["ema_50"]
        df["trend_strength"] = ema_spread * 100  # as % of price

        df["market_regime"] = np.select(
            [
                ema_spread > 0.003,    # strong trend
                ema_spread > 0.001,    # mild trend
            ],
            ["trending", "mild_trend"],
            default="ranging"
        )
    else:
        df["trend_strength"] = 0.0
        df["market_regime"] = "unknown"

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps."""
    logger.info("Engineering features...")
    df = add_candlestick_patterns(df)
    df = add_multi_bar_context(df)
    df = add_confluence_score(df)
    df = add_market_regime(df)
    logger.success(f"Feature engineering complete. Columns: {len(df.columns)}")
    return df
