"""
Dio Trading App — Liquidity Analysis Module
============================================
Classifies each bar by trading session, session overlap,
liquidity conditions, and potential stop-hunt / sweep zones.

Sessions are defined in UTC. EUR/USD liquidity follows these windows:
  - Tokyo:       00:00 – 09:00 UTC  (low EUR/USD activity)
  - London:      07:00 – 16:00 UTC  (highest EUR/USD activity)
  - New York:    12:00 – 21:00 UTC  (high activity, key US data)
  - London/NY:   12:00 – 16:00 UTC  (peak overlap, highest liquidity)

VOLUME DISCLAIMER: All volume/tick figures are proxy values.
"""

import numpy as np
import pandas as pd
from loguru import logger

from backend.core.config import SESSIONS


# ─── Session Classification ───────────────────────────────────────────────────

def _time_in_range(hour_utc: int, start_str: str, end_str: str) -> bool:
    start = int(start_str.split(":")[0])
    end = int(end_str.split(":")[0])
    if start <= end:
        return start <= hour_utc < end
    return hour_utc >= start or hour_utc < end  # overnight wrap


def add_session_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each bar with its active trading session(s)."""
    hour = df.index.hour  # UTC hour

    df["session_tokyo"] = [_time_in_range(h, *SESSIONS["tokyo"].values()) for h in hour]
    df["session_london"] = [_time_in_range(h, *SESSIONS["london"].values()) for h in hour]
    df["session_ny"] = [_time_in_range(h, *SESSIONS["new_york"].values()) for h in hour]
    df["session_overlap"] = df["session_london"] & df["session_ny"]

    # Primary session label (priority: overlap > london > ny > tokyo > off)
    conditions = [
        df["session_overlap"],
        df["session_london"] & ~df["session_overlap"],
        df["session_ny"] & ~df["session_overlap"],
        df["session_tokyo"],
    ]
    choices = ["london_ny_overlap", "london", "new_york", "tokyo"]
    df["session_primary"] = np.select(conditions, choices, default="off_hours")

    return df


# ─── Session Statistics ───────────────────────────────────────────────────────

def compute_session_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, compute average volatility and volume proxy.
    Returns a summary table used in the signal engine to weight
    liquidity risk.
    """
    if "session_primary" not in df.columns:
        df = add_session_labels(df)

    stats = []
    for session in ["london_ny_overlap", "london", "new_york", "tokyo", "off_hours"]:
        sub = df[df["session_primary"] == session]
        if len(sub) == 0:
            continue
        stats.append({
            "session": session,
            "bar_count": len(sub),
            "avg_atr": round(sub["atr"].mean(), 6) if "atr" in sub else None,
            "avg_volume_proxy": round(sub["volume_proxy"].mean(), 2),
            "avg_hl_range": round(sub["hl_range"].mean(), 6),
            "pct_of_data": round(len(sub) / len(df) * 100, 2),
        })
    return pd.DataFrame(stats)


# ─── Liquidity Zone Detection ─────────────────────────────────────────────────

def add_liquidity_zones(df: pd.DataFrame, lookback: int = 48) -> pd.DataFrame:
    """
    Mark high and low liquidity zones based on relative volume proxy
    and session.

    High liquidity zones: volume proxy > 1.5× rolling mean AND in
                          London or NY session.
    Low liquidity zones:  volume proxy < 0.5× rolling mean OR in
                          off-hours.

    These zones affect signal confidence — we prefer signals in
    high-liquidity windows and penalize off-hours signals.
    """
    if "session_primary" not in df.columns:
        df = add_session_labels(df)

    vol_ma = df["volume_proxy"].rolling(lookback).mean()
    vol_ratio = df["volume_proxy"] / vol_ma

    high_liquidity_session = df["session_primary"].isin(
        ["london_ny_overlap", "london", "new_york"]
    )

    df["liquidity_zone"] = "normal"
    df.loc[(vol_ratio > 1.5) & high_liquidity_session, "liquidity_zone"] = "high"
    df.loc[(vol_ratio < 0.5) | (df["session_primary"] == "off_hours"), "liquidity_zone"] = "low"

    df["liquidity_score"] = np.select(
        [df["liquidity_zone"] == "high", df["liquidity_zone"] == "low"],
        [1.0, 0.3],
        default=0.7
    )

    return df


# ─── Stop-Hunt / Sweep Zone Detection ────────────────────────────────────────

def add_stop_hunt_zones(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detect potential stop-hunt / liquidity sweep patterns.

    A stop hunt / sweep is suspected when:
    1. Price briefly spikes beyond a recent swing high/low
    2. ...and then reverses within the same or next bar
    3. Often paired with a long wick

    These zones are important: a signal fired AFTER a confirmed sweep
    often has a higher win rate (smart money reversals).

    DISCLAIMER: This is heuristic detection, not certainty.
    """
    roll_high = df["high"].rolling(lookback).max().shift(1)
    roll_low = df["low"].rolling(lookback).min().shift(1)

    # Bearish sweep: price wick above recent high but closes back below
    df["bearish_sweep"] = (
        (df["high"] > roll_high) &
        (df["close"] < roll_high) &
        (df["upper_wick"] > df["body"] * 1.5)
    )

    # Bullish sweep: price wick below recent low but closes back above
    df["bullish_sweep"] = (
        (df["low"] < roll_low) &
        (df["close"] > roll_low) &
        (df["lower_wick"] > df["body"] * 1.5)
    )

    # Sweep zone label
    df["stop_hunt"] = np.where(
        df["bearish_sweep"], "bearish_sweep",
        np.where(df["bullish_sweep"], "bullish_sweep", "none")
    )

    return df


# ─── Session Open Levels ──────────────────────────────────────────────────────

def add_session_opens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Record the opening price of each major session.
    Session opens are key liquidity levels — price often revisits them.
    """
    df["london_open_price"] = np.where(
        (df.index.hour == 7) & df["session_london"], df["open"], np.nan
    )
    df["ny_open_price"] = np.where(
        (df.index.hour == 12) & df["session_ny"], df["open"], np.nan
    )
    df["london_open_price"] = df["london_open_price"].ffill()
    df["ny_open_price"] = df["ny_open_price"].ffill()

    # Distance from session open (in pips for EUR/USD)
    df["dist_from_london_open"] = (df["close"] - df["london_open_price"]) * 10_000
    df["dist_from_ny_open"] = (df["close"] - df["ny_open_price"]) * 10_000

    return df


# ─── Master Builder ───────────────────────────────────────────────────────────

def compute_all_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing liquidity analysis...")
    df = add_session_labels(df)
    df = add_liquidity_zones(df)
    df = add_stop_hunt_zones(df)
    df = add_session_opens(df)
    logger.success("Liquidity analysis complete.")
    return df
