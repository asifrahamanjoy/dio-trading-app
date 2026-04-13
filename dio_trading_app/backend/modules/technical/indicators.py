"""
Dio Trading App — Technical Analysis Module
=============================================
Computes all technical indicators used in signal generation.
Pure functions: input DataFrame → output DataFrame with added columns.

Indicators implemented:
  RSI, MACD, EMA (9/21/50), SMA (200), ATR, Bollinger Bands,
  Momentum, Volatility, Support/Resistance levels,
  Trend detection, Breakout & Reversal detection,
  Volume condition classification.
"""

import numpy as np
import pandas as pd
from loguru import logger

from backend.core.config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    EMA_SHORT, EMA_MED, EMA_LONG, SMA_200,
    ATR_PERIOD, BB_PERIOD, BB_STD, MOMENTUM_PERIOD
)


# ─── RSI ─────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = df[f"rsi_{period}"] > RSI_OVERBOUGHT
    df["rsi_oversold"] = df[f"rsi_{period}"] < RSI_OVERSOLD
    df["rsi_divergence_bull"] = (
        (df["close"] < df["close"].shift(1)) &
        (df[f"rsi_{period}"] > df[f"rsi_{period}"].shift(1))
    )
    df["rsi_divergence_bear"] = (
        (df["close"] > df["close"].shift(1)) &
        (df[f"rsi_{period}"] < df[f"rsi_{period}"].shift(1))
    )
    return df


# ─── MACD ─────────────────────────────────────────────────────────────────────

def add_macd(df: pd.DataFrame, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIGNAL) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=sig, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    df["macd_bullish_cross"] = (
        (df["macd_line"] > df["macd_signal"]) &
        (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    )
    df["macd_bearish_cross"] = (
        (df["macd_line"] < df["macd_signal"]) &
        (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
    )
    return df


# ─── Moving Averages ──────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df[f"ema_{EMA_SHORT}"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df[f"ema_{EMA_MED}"] = df["close"].ewm(span=EMA_MED, adjust=False).mean()
    df[f"ema_{EMA_LONG}"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()
    df[f"sma_{SMA_200}"] = df["close"].rolling(SMA_200).mean()

    # Price position relative to MAs
    df["above_ema9"] = df["close"] > df[f"ema_{EMA_SHORT}"]
    df["above_ema21"] = df["close"] > df[f"ema_{EMA_MED}"]
    df["above_ema50"] = df["close"] > df[f"ema_{EMA_LONG}"]
    df["above_sma200"] = df["close"] > df[f"sma_{SMA_200}"]

    # EMA alignment (bullish = 9>21>50, bearish = 9<21<50)
    df["ema_bull_align"] = (
        (df[f"ema_{EMA_SHORT}"] > df[f"ema_{EMA_MED}"]) &
        (df[f"ema_{EMA_MED}"] > df[f"ema_{EMA_LONG}"])
    )
    df["ema_bear_align"] = (
        (df[f"ema_{EMA_SHORT}"] < df[f"ema_{EMA_MED}"]) &
        (df[f"ema_{EMA_MED}"] < df[f"ema_{EMA_LONG}"])
    )

    # Golden / Death cross (EMA 9/21)
    df["golden_cross"] = (
        (df[f"ema_{EMA_SHORT}"] > df[f"ema_{EMA_MED}"]) &
        (df[f"ema_{EMA_SHORT}"].shift(1) <= df[f"ema_{EMA_MED}"].shift(1))
    )
    df["death_cross"] = (
        (df[f"ema_{EMA_SHORT}"] < df[f"ema_{EMA_MED}"]) &
        (df[f"ema_{EMA_SHORT}"].shift(1) >= df[f"ema_{EMA_MED}"].shift(1))
    )
    return df


# ─── ATR ──────────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=period - 1, min_periods=period).mean()
    df["atr_pct"] = df["atr"] / df["close"]  # ATR as % of price
    return df


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std: float = BB_STD) -> pd.DataFrame:
    sma = df["close"].rolling(period).mean()
    rolling_std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + std * rolling_std
    df["bb_lower"] = sma - std * rolling_std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Squeeze: BB width narrowing (low volatility before breakout)
    bb_width_ma = df["bb_width"].rolling(20).mean()
    df["bb_squeeze"] = df["bb_width"] < bb_width_ma * 0.8

    df["price_at_upper_bb"] = df["close"] >= df["bb_upper"]
    df["price_at_lower_bb"] = df["close"] <= df["bb_lower"]
    return df


# ─── Momentum ─────────────────────────────────────────────────────────────────

def add_momentum(df: pd.DataFrame, period: int = MOMENTUM_PERIOD) -> pd.DataFrame:
    df["momentum"] = df["close"] - df["close"].shift(period)
    df["momentum_pct"] = df["momentum"] / df["close"].shift(period)
    df["roc"] = (df["close"] / df["close"].shift(period) - 1) * 100  # Rate of Change
    return df


# ─── Volatility ───────────────────────────────────────────────────────────────

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["volatility_20"] = df["log_returns"].rolling(window).std() * np.sqrt(252 * 24)  # annualised
    df["volatility_regime"] = pd.cut(
        df["volatility_20"],
        bins=[-np.inf, 0.05, 0.10, 0.15, np.inf],
        labels=["very_low", "low", "medium", "high"]
    )
    return df


# ─── Support & Resistance ─────────────────────────────────────────────────────

def add_support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect rolling pivot-based support and resistance.
    Uses local highs/lows over a lookback window.
    """
    df["pivot_high"] = df["high"].where(
        (df["high"] == df["high"].rolling(lookback, center=True).max()),
        other=np.nan
    )
    df["pivot_low"] = df["low"].where(
        (df["low"] == df["low"].rolling(lookback, center=True).min()),
        other=np.nan
    )

    # Forward-fill the most recent pivot as dynamic S/R
    df["resistance"] = df["pivot_high"].ffill()
    df["support"] = df["pivot_low"].ffill()

    df["near_resistance"] = (df["high"] >= df["resistance"] * 0.998) & (df["high"] <= df["resistance"] * 1.002)
    df["near_support"] = (df["low"] <= df["support"] * 1.002) & (df["low"] >= df["support"] * 0.998)

    return df


# ─── Trend Detection ─────────────────────────────────────────────────────────

def add_trend(df: pd.DataFrame, fast: int = EMA_MED, slow: int = EMA_LONG) -> pd.DataFrame:
    """
    Multi-layer trend detection:
    - Primary trend: price vs SMA200
    - Secondary trend: EMA21 vs EMA50
    - Short-term momentum: RSI + MACD alignment
    """
    if f"ema_{fast}" not in df.columns:
        df = add_moving_averages(df)

    df["trend_primary"] = np.where(
        df["close"] > df.get(f"sma_{SMA_200}", df["close"]), "bullish",
        np.where(df["close"] < df.get(f"sma_{SMA_200}", df["close"]), "bearish", "neutral")
    )

    df["trend_secondary"] = np.where(
        df[f"ema_{fast}"] > df[f"ema_{slow}"], "bullish",
        np.where(df[f"ema_{fast}"] < df[f"ema_{slow}"], "bearish", "neutral")
    )

    # Higher highs / higher lows (simple: compare to 10-bar ago)
    df["hh"] = df["high"] > df["high"].shift(10)
    df["hl"] = df["low"] > df["low"].shift(10)
    df["lh"] = df["high"] < df["high"].shift(10)
    df["ll"] = df["low"] < df["low"].shift(10)

    df["uptrend_structure"] = df["hh"] & df["hl"]
    df["downtrend_structure"] = df["lh"] & df["ll"]

    return df


# ─── Breakout & Reversal ──────────────────────────────────────────────────────

def add_breakout_reversal(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    roll_high = df["high"].rolling(lookback).max().shift(1)
    roll_low = df["low"].rolling(lookback).min().shift(1)

    df["breakout_up"] = df["close"] > roll_high
    df["breakout_down"] = df["close"] < roll_low

    # Candle patterns for reversal
    df["bullish_engulf"] = (
        (df["open"] > df["close"].shift(1)) &   # gap below prev close
        (df["close"] > df["open"].shift(1)) &    # closes above prev open
        (df["close"] > df["open"])               # bullish candle
    )
    df["bearish_engulf"] = (
        (df["open"] < df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1)) &
        (df["close"] < df["open"])
    )

    # Pin bar / hammer (lower wick > 2× body)
    df["hammer"] = (df["lower_wick"] > 2 * df["body"]) & (df["upper_wick"] < df["body"])
    df["shooting_star"] = (df["upper_wick"] > 2 * df["body"]) & (df["lower_wick"] < df["body"])

    return df


# ─── Volume Condition Analysis ────────────────────────────────────────────────

def add_volume_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each bar into one of 4 volume-price conditions.

    DISCLAIMER: Volume here is proxy/tick volume — not true spot forex volume.
    Interpretation is directional only; treat with appropriate skepticism.

    Conditions:
      1. price_up_vol_up    — Price rises, volume rises   → Bullish confirmation
      2. price_down_vol_down— Price falls, volume falls   → Weak sell, possible reversal
      3. price_down_vol_up  — Price falls, volume rises   → Bearish confirmation
      4. price_up_vol_down  — Price rises, volume falls   → Weak buy, possible reversal
    """
    price_up = df["close"] > df["close"].shift(1)
    vol_up = df["volume_proxy"] > df["volume_proxy"].shift(1)

    df["vol_condition"] = "unknown"
    df.loc[price_up & vol_up, "vol_condition"] = "price_up_vol_up"
    df.loc[~price_up & ~vol_up, "vol_condition"] = "price_down_vol_down"
    df.loc[~price_up & vol_up, "vol_condition"] = "price_down_vol_up"
    df.loc[price_up & ~vol_up, "vol_condition"] = "price_up_vol_down"

    # Volume MA for relative comparison
    df["volume_ma_20"] = df["volume_proxy"].rolling(20).mean()
    df["volume_ratio"] = df["volume_proxy"] / df["volume_ma_20"]
    df["high_volume"] = df["volume_ratio"] > 1.5
    df["low_volume"] = df["volume_ratio"] < 0.5

    return df


def compute_volume_condition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame of frequency and next-bar direction
    for each of the 4 volume-price conditions.

    This feeds into the historical win rate calculation for signals.
    """
    df = df.copy()
    df["next_close"] = df["close"].shift(-1)
    df["next_direction"] = (df["next_close"] > df["close"]).map({True: "up", False: "down"})

    stats = []
    for cond in ["price_up_vol_up", "price_down_vol_down", "price_down_vol_up", "price_up_vol_down"]:
        subset = df[df["vol_condition"] == cond]
        n = len(subset)
        if n == 0:
            continue
        pct_of_total = n / len(df) * 100
        next_up = (subset["next_direction"] == "up").sum()
        stats.append({
            "condition": cond,
            "count": n,
            "pct_of_total": round(pct_of_total, 2),
            "next_bar_up_count": int(next_up),
            "next_bar_up_rate": round(next_up / n * 100, 2),
            "next_bar_down_rate": round((n - next_up) / n * 100, 2),
        })
    return pd.DataFrame(stats)


# ─── Master Builder ───────────────────────────────────────────────────────────

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical indicators in correct order."""
    logger.info("Computing all technical indicators...")
    df = add_rsi(df)
    df = add_macd(df)
    df = add_moving_averages(df)
    df = add_atr(df)
    df = add_bollinger_bands(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_support_resistance(df)
    df = add_trend(df)
    df = add_breakout_reversal(df)
    df = add_volume_conditions(df)
    logger.success(f"Technical indicators computed. Total columns: {len(df.columns)}")
    return df
