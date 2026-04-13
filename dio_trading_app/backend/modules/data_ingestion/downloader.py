"""
Dio Trading App — Data Ingestion Module
========================================
Downloads and caches EUR/USD OHLCV data from Yahoo Finance.

VOLUME DISCLAIMER:
  EUR/USD spot forex has NO centralized volume. This module fetches:
  - Tick/proxy volume from yfinance EUR/USD (EURUSD=X) — not true volume
  - CME EUR/USD futures volume (6E=F) as a secondary volume proxy
  Volume data is labeled accordingly throughout the codebase.
"""

import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

from backend.core.config import (
    SYMBOL, FUTURES_SYMBOL, LOOKBACK_YEARS,
    INTERVAL_PRIMARY, INTERVAL_DAILY,
    RAW_DIR, CACHE_DIR, VOLUME_DISCLAIMER
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cache_path(symbol: str, interval: str, suffix: str = "parquet") -> Path:
    key = hashlib.md5(f"{symbol}_{interval}".encode()).hexdigest()[:8]
    return CACHE_DIR / f"{symbol.replace('=', '').replace('/', '')}_{interval}_{key}.{suffix}"


def _is_cache_fresh(path: Path, max_age_hours: int = 4) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_hours * 3600


# ─── Core Downloader ──────────────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str = SYMBOL,
    interval: str = INTERVAL_PRIMARY,
    years: int = LOOKBACK_YEARS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data for a given symbol and interval.
    Uses local Parquet cache to avoid repeated downloads.

    Returns a DataFrame with columns:
        open, high, low, close, volume (proxy), returns, log_returns
    Volume column is tagged with a _proxy suffix and includes disclaimer.
    """
    cache = _cache_path(symbol, interval)

    if not force_refresh and _is_cache_fresh(cache, max_age_hours=4):
        logger.info(f"Loading cached data: {cache}")
        df = pd.read_parquet(cache)
        logger.info(f"Loaded {len(df)} rows from cache ({symbol} {interval})")
        return df

    end_date = datetime.utcnow()
    total_days = years * 365 + 30

    # yfinance caps intraday data depending on interval:
    #   <60m intervals (5m, 15m, 30m): last 60 days only
    #   60m/1h: ~730 days per request
    #   1d+: full range
    short_intraday = interval in ("1m", "2m", "5m", "15m", "30m")
    if short_intraday:
        total_days = min(total_days, 59)  # yfinance hard limit
        max_chunk_days = 59
    elif interval not in ("1d", "1wk", "1mo"):
        max_chunk_days = 700
    else:
        max_chunk_days = total_days

    chunks: list[pd.DataFrame] = []
    chunk_end = end_date
    remaining = total_days

    while remaining > 0:
        chunk_span = min(remaining, max_chunk_days)
        chunk_start = chunk_end - timedelta(days=chunk_span)
        logger.info(
            f"Downloading {symbol} [{interval}] chunk: "
            f"{chunk_start.date()} → {chunk_end.date()}"
        )
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                prepost=False,
            )
            if not raw.empty:
                chunks.append(raw)
        except Exception as e:
            logger.warning(f"Chunk download failed ({chunk_start.date()}-{chunk_end.date()}): {e}")

        chunk_end = chunk_start - timedelta(days=1)
        remaining -= chunk_span

    if not chunks:
        raise ValueError(f"No data returned for {symbol} at interval {interval}. "
                         "yfinance may not support this interval for forex.")

    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    df = _clean_ohlcv(combined, symbol=symbol, interval=interval)
    df.to_parquet(cache)
    logger.success(f"Downloaded and cached {len(df)} rows for {symbol} [{interval}]")
    return df


def fetch_live_quote(symbol: str = SYMBOL) -> dict:
    """
    Fetch a recent display-only quote for the dashboard.

    This is intentionally separate from the analysis pipeline so that
    live price display can be refreshed without altering signal inputs.
    """
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period="1d", interval="1m", auto_adjust=True, prepost=False)

    if raw.empty:
        raise ValueError(f"No live quote returned for {symbol}.")

    raw.columns = [c.lower() for c in raw.columns]
    latest = raw.iloc[-1]
    timestamp = pd.to_datetime(raw.index[-1], utc=True)

    fast_info = {}
    try:
        fast_info = dict(getattr(ticker, "fast_info", {}) or {})
    except Exception:
        fast_info = {}

    last_price = fast_info.get("lastPrice")
    if last_price is None:
        last_price = fast_info.get("regularMarketPrice")
    if last_price is None:
        last_price = latest["close"]

    return {
        "price": float(last_price),
        "open": float(fast_info.get("open", latest.get("open", latest["close"]))),
        "high": float(fast_info.get("dayHigh", latest.get("high", latest["close"]))),
        "low": float(fast_info.get("dayLow", latest.get("low", latest["close"]))),
        "timestamp": timestamp.isoformat(),
        "source_interval": "1m",
        "source_name": "yfinance_fast_info",
        "symbol": symbol,
    }


def _clean_ohlcv(raw: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Standardize column names, remove bad rows, add basic derived columns."""
    df = raw.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "datetime"

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"volume": "volume_proxy"})

    # Forex pairs often have no volume — fill with synthetic tick proxy
    if "volume_proxy" in df.columns and df["volume_proxy"].isna().all():
        logger.warning(
            "Volume data is entirely NaN (typical for spot forex). "
            "Generating synthetic tick proxy from HL range."
        )
        # Use bar range as a volatility-based proxy for activity
        hl = (df["high"] - df["low"]).abs()
        df["volume_proxy"] = (hl / hl.rolling(50, min_periods=1).mean() * 1000).fillna(1000)
    elif "volume_proxy" in df.columns:
        df["volume_proxy"] = df["volume_proxy"].fillna(0)

    # Drop rows where OHLC is zero or NaN
    ohlc_cols = ["open", "high", "low", "close"]
    for col in ohlc_cols:
        df[col] = df[col].replace(0, np.nan)
    df = df.dropna(subset=ohlc_cols)

    # Ensure correct OHLC relationship
    df = df[df["high"] >= df["low"]]
    df = df[df["high"] >= df["open"]]
    df = df[df["high"] >= df["close"]]

    # Add derived columns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"] = df["high"] - df["low"]
    df["body"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Volume metadata
    df.attrs["volume_type"] = "tick_proxy" if "=X" in symbol else "futures_proxy"
    df.attrs["volume_disclaimer"] = VOLUME_DISCLAIMER
    df.attrs["symbol"] = symbol
    df.attrs["interval"] = interval

    df.dropna(subset=["returns"], inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_futures_volume(years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Fetch CME EUR/USD futures (6E=F) daily volume as a proxy for
    institutional activity in the EUR/USD market.
    Futures volume is the closest available proxy to real spot volume.
    """
    cache = _cache_path(FUTURES_SYMBOL, "1d")
    if _is_cache_fresh(cache, max_age_hours=12):
        return pd.read_parquet(cache)

    logger.info(f"Downloading EUR/USD futures volume proxy ({FUTURES_SYMBOL})")
    end = datetime.utcnow()
    start = end - timedelta(days=years * 365 + 30)

    ticker = yf.Ticker(FUTURES_SYMBOL)
    raw = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )

    if raw.empty:
        logger.warning("Futures volume not available. Volume proxy will be tick-only.")
        return pd.DataFrame()

    df = _clean_ohlcv(raw, symbol=FUTURES_SYMBOL, interval="1d")
    df = df[["close", "volume_proxy"]].rename(columns={
        "close": "futures_close",
        "volume_proxy": "futures_volume"
    })
    df.to_parquet(cache)
    logger.success(f"Futures volume proxy: {len(df)} daily rows")
    return df


def load_multi_timeframe(years: int = LOOKBACK_YEARS) -> dict[str, pd.DataFrame]:
    """
    Load data across multiple timeframes for multi-timeframe analysis.
    Returns dict: {"1h": df_1h, "1d": df_daily}

    Note: yfinance supports "1h" for ~730 days max. For full 5y hourly data
    we fetch in chunks and concatenate.
    """
    frames = {}

    # Daily — full 5 years
    frames["1d"] = fetch_ohlcv(SYMBOL, "1d", years=years)

    # Hourly — yfinance caps at ~730 days; fetch in 2 chunks
    hourly_chunks = []
    now = datetime.utcnow()
    for chunk_start_offset in [years * 365, 730]:
        chunk_start = now - timedelta(days=chunk_start_offset)
        chunk_end = now - timedelta(days=max(0, chunk_start_offset - 730))
        try:
            t = yf.Ticker(SYMBOL)
            raw = t.history(
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval="1h",
                auto_adjust=True,
            )
            if not raw.empty:
                hourly_chunks.append(_clean_ohlcv(raw, SYMBOL, "1h"))
        except Exception as e:
            logger.warning(f"Hourly chunk download failed: {e}")

    if hourly_chunks:
        combined = pd.concat(hourly_chunks)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        frames["1h"] = combined
    else:
        # Fall back to whatever we can get
        frames["1h"] = fetch_ohlcv(SYMBOL, "1h", years=2)

    logger.info(f"Multi-TF loaded — daily: {len(frames['1d'])} bars, hourly: {len(frames.get('1h', []))} bars")
    return frames


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a human-readable summary of a loaded OHLCV dataframe."""
    return {
        "rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "columns": list(df.columns),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "volume_type": df.attrs.get("volume_type", "unknown"),
        "volume_disclaimer": df.attrs.get("volume_disclaimer", VOLUME_DISCLAIMER),
        "close_range": {
            "min": round(df["close"].min(), 5),
            "max": round(df["close"].max(), 5),
            "mean": round(df["close"].mean(), 5),
        },
    }
