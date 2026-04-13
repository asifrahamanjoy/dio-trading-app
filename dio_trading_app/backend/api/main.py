"""
Dio Trading App — FastAPI Backend
===================================
REST API providing endpoints for:
  - Market data and analysis
  - Signal retrieval
  - Backtest execution and results
  - Model training / prediction
  - Alert management
  - Dashboard summary

Run with:
    uvicorn backend.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from typing import Optional
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import (
    settings, VOLUME_DISCLAIMER, RISK_REWARD_RATIO, INTERVAL_PRIMARY,
    DEFAULT_MULTI_TF_SCAN, DEFAULT_SIGNAL_MODE, MAX_SCAN_BARS, MULTI_TF_CONFIGS,
    CONFIRMATION_ONLY_TIMEFRAMES, RECOMMENDED_LIVE_SETUP,
    PAIR_LABEL, SUPPORTED_PAIRS, get_pair_label, get_pair_display_name,
    get_symbol_for_pair, get_volume_disclaimer_for_pair, get_recommended_live_setup,
    get_timeframe_config, get_multi_tf_configs, get_direction_bias, normalize_pair,
)
from backend.core.database import init_db, get_db, Signal as DBSignal, BacktestRun
from backend.modules.data_ingestion.downloader import fetch_ohlcv, fetch_live_quote
from backend.modules.preprocessing.cleaner import preprocess
from backend.modules.technical.indicators import compute_all_indicators
from backend.modules.liquidity.analysis import compute_all_liquidity
from backend.modules.events.fundamental import (
    load_event_calendar, flag_event_windows,
    get_upcoming_high_impact_events, fetch_market_news_headlines,
)
from backend.modules.features.engineer import engineer_all_features
from backend.modules.backtesting.engine import run_backtest
from backend.modules.backtesting.optimizer import run_optimization, get_best_config
from backend.modules.prediction.model import (
    build_training_dataset, train_model, load_model, predict_probability
)


def _training_lookback_years(timeframe: str) -> int:
    if timeframe == "1d":
        return 10
    if timeframe == "4h":
        return 8
    if timeframe == "1h":
        return 5
    if timeframe == "15m":
        return 2
    return 5
from backend.modules.signals.engine import (
    scan_for_signals,
    scan_multi_tf_signals,
    timeframe_signal_diagnostics,
)


# ─── App Lifecycle ─────────────────────────────────────────────────────────────

_app_state = {}

ALLOWED_SIGNAL_TIMEFRAMES = {
    "EUR/USD": ["1d", "4h", "1h", "15m"],
    "JPY/USD": ["1d", "4h", "15m"],
    "XAU/USD": ["1d", "4h", "1h", "15m"],
}


def _allowed_timeframes_for_pair(pair: str) -> list[str]:
    return ALLOWED_SIGNAL_TIMEFRAMES.get(pair, [])


def _validate_signal_pair_timeframe(pair: str, timeframe: str) -> str:
    normalized_timeframe = str(timeframe).lower().strip()
    allowed_timeframes = _allowed_timeframes_for_pair(pair)
    if not allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Pair '{pair}' is not allowed for signals.")
    if normalized_timeframe not in allowed_timeframes:
        allowed_text = ", ".join(tf.upper() for tf in allowed_timeframes)
        raise HTTPException(
            status_code=400,
            detail=f"Timeframe '{timeframe}' is not allowed for {pair}. Allowed: {allowed_text}",
        )
    return normalized_timeframe


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising Dio Trading App backend...")
    init_db()
    _app_state["enriched_cache"] = {}
    _app_state["response_cache"] = {}

    _app_state["model_bundles"] = {}
    for pair in SUPPORTED_PAIRS:
        bundle = load_model(pair=pair)
        if bundle:
            _app_state["model_bundles"][pair] = bundle
            logger.success(f"Prediction model loaded for {pair}.")
        else:
            logger.warning(f"No trained model found for {pair}. Run /train first.")

    # Store backtest win rates (will be populated after first backtest)
    _app_state["backtest_win_rates"] = {
        pair: {"BUY": 0.52, "SELL": 0.52}
        for pair in SUPPORTED_PAIRS
    }

    yield

    logger.info("Dio Trading App backend shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dio Trading App API",
    description=(
        "Multi-market analysis, signal generation, and backtesting platform. "
        "All signals are probabilistic. Not financial advice. "
        f"Volume disclaimer: {VOLUME_DISCLAIMER}"
    ),
    version=settings.version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    pair: str = PAIR_LABEL
    timeframe: str = INTERVAL_PRIMARY
    strategy_name: str = "DioMultiCondition_v1"
    rr: float = RISK_REWARD_RATIO
    force_refresh: bool = False
    recent_signals: int = 20


class TrainRequest(BaseModel):
    pair: str = PAIR_LABEL
    timeframe: str = INTERVAL_PRIMARY
    force_refresh: bool = False


class OptimizationRequest(BaseModel):
    pair: str = PAIR_LABEL
    timeframes: list[str] = DEFAULT_MULTI_TF_SCAN
    score_mode: str = "win_rate"


# ─── Shared helper: build enriched DataFrame ──────────────────────────────────

def _get_model_bundle(pair: str) -> dict | None:
    pair = normalize_pair(pair)
    bundles = _app_state.setdefault("model_bundles", {})
    if pair not in bundles:
        bundles[pair] = load_model(pair=pair)
    return bundles.get(pair)


def _get_backtest_win_rates(pair: str) -> dict:
    pair = normalize_pair(pair)
    win_rates = _app_state.setdefault("backtest_win_rates", {})
    return win_rates.get(pair, {"BUY": 0.52, "SELL": 0.52})


def _enrich_df(pair: str, timeframe: str, force_refresh: bool = False) -> pd.DataFrame:
    """Download, preprocess, and add all indicators/features to a DataFrame."""
    df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe, force_refresh=force_refresh)
    df = preprocess(df)
    df = compute_all_indicators(df)
    df = compute_all_liquidity(df)
    events = load_event_calendar()
    df = flag_event_windows(df, events)
    df = engineer_all_features(df)
    return df


def _cache_ttl_for_timeframe(timeframe: str) -> int:
    ttl_map = {
        "15m": 120,
        "1h": 300,
        "1d": 1800,
    }
    return ttl_map.get(timeframe, 300)


def _enrich_df_cached(pair: str, timeframe: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return cached enriched data when it is still fresh enough for the timeframe."""
    if force_refresh:
        return _enrich_df(pair, timeframe, force_refresh=True)

    cache = _app_state.setdefault("enriched_cache", {})
    cache_key = (normalize_pair(pair), timeframe)
    cached = cache.get(cache_key)
    now = time.time()
    ttl = _cache_ttl_for_timeframe(timeframe)

    if cached and (now - cached["created_at"] <= ttl):
        logger.info(f"Using in-memory enriched cache for {pair} {timeframe}")
        return cached["df"]

    df = _enrich_df(pair, timeframe, force_refresh=False)
    cache[cache_key] = {
        "df": df,
        "created_at": now,
    }
    return df


def _get_cached_response(cache_key: tuple, ttl: int):
    cache = _app_state.setdefault("response_cache", {})
    cached = cache.get(cache_key)
    if not cached:
        return None

    if time.time() - cached["created_at"] > ttl:
        cache.pop(cache_key, None)
        return None

    return deepcopy(cached["data"])


def _set_cached_response(cache_key: tuple, data: dict):
    cache = _app_state.setdefault("response_cache", {})
    cache[cache_key] = {
        "created_at": time.time(),
        "data": deepcopy(data),
    }


def _build_timeframe_overview(
    pair: str,
    timeframe: str,
    model_bundle: Optional[dict] = None,
    signal_mode: str = DEFAULT_SIGNAL_MODE,
) -> dict:
    df = _enrich_df_cached(pair, timeframe)
    latest = df.iloc[-1]
    cfg = get_timeframe_config(pair, timeframe)
    diagnostics = timeframe_signal_diagnostics(
        df=df,
        model_bundle=model_bundle,
        pair=pair,
        timeframe=timeframe,
        signal_mode=signal_mode,
    )
    return {
        "timeframe": timeframe,
        "role": "confirmation-only" if timeframe in CONFIRMATION_ONLY_TIMEFRAMES else "entry",
        "current_price": round(float(latest.get("close", 0.0)), 5),
        "trend_primary": str(latest.get("trend_primary", "unknown")),
        "liquidity_zone": str(latest.get("liquidity_zone", "unknown")),
        "volatility_regime": str(latest.get("volatility_regime", "unknown")),
        "event_window": bool(latest.get("event_window", False)),
        "risk_level": cfg.get("risk_level", "UNKNOWN"),
        "backtest_win_rate": cfg.get("backtest_win_rate", 0.0),
        "profit_factor": cfg.get("backtest_profit_factor", 0.0),
        "rr_ratio": cfg.get("rr_ratio", 1.0),
        "last_bar_time": str(df.index[-1]),
        "signal_diagnostics": diagnostics,
    }


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "app": settings.app_name,
        "version": settings.version,
        "status": "running",
        "supported_pairs": SUPPORTED_PAIRS,
        "disclaimer": "Not financial advice. Probabilistic signals only.",
    }


@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "supported_pairs": SUPPORTED_PAIRS,
    }


@app.get("/market/upcoming", tags=["dashboard"])
async def market_upcoming(
    pair: str = PAIR_LABEL,
    hours_ahead: int = 168,
    limit: int = 8,
):
    """Upcoming high-impact events and optional market headlines for dashboard top section."""
    try:
        pair = normalize_pair(pair)
        safe_hours = max(1, min(hours_ahead, 24 * 30))
        safe_limit = max(1, min(limit, 20))

        cache_key = ("market_upcoming", pair, safe_hours, safe_limit)
        cached = _get_cached_response(cache_key, ttl=300)
        if cached is not None:
            return cached

        events = get_upcoming_high_impact_events(
            pair=pair,
            hours_ahead=safe_hours,
            max_items=safe_limit,
        )
        headlines = fetch_market_news_headlines(max_articles=min(safe_limit, 10))

        response = {
            "pair": get_pair_label(pair),
            "hours_ahead": safe_hours,
            "upcoming_events": events,
            "news_headlines": headlines,
            "generated_at": datetime.utcnow().isoformat(),
        }
        _set_cached_response(cache_key, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Market Data ─────────────────────────────────────────────────────────────

@app.get("/data/summary", tags=["data"])
async def data_summary(pair: str = PAIR_LABEL, timeframe: str = INTERVAL_PRIMARY, force_refresh: bool = False):
    """Return summary of available market data."""
    try:
        pair = normalize_pair(pair)
        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe, force_refresh=force_refresh)
        return {
            "pair": get_pair_label(pair),
            "display_name": get_pair_display_name(pair),
            "timeframe": timeframe,
            "total_bars": len(df),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "latest_close": round(float(df["close"].iloc[-1]), 5),
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/ohlcv", tags=["data"])
async def get_ohlcv(pair: str = PAIR_LABEL, timeframe: str = INTERVAL_PRIMARY, limit: int = 200):
    """Return recent OHLCV bars as JSON."""
    try:
        pair = normalize_pair(pair)
        cache_key = ("data_ohlcv", pair, timeframe, limit)
        cached = _get_cached_response(cache_key, ttl=30)
        if cached is not None:
            return cached

        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe)
        df_out = df.tail(limit).copy()
        df_out.index = df_out.index.astype(str)
        response = {
            "pair": get_pair_label(pair),
            "timeframe": timeframe,
            "bars": df_out.reset_index().to_dict(orient="records"),
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
        }
        _set_cached_response(cache_key, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/volume-conditions", tags=["data"])
async def volume_condition_stats(pair: str = PAIR_LABEL, timeframe: str = "1h"):
    """Return statistics for all 4 volume-price market conditions."""
    try:
        from backend.modules.technical.indicators import (
            compute_all_indicators, compute_volume_condition_stats
        )
        pair = normalize_pair(pair)
        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe)
        df = compute_all_indicators(df)
        stats = compute_volume_condition_stats(df)
        return {
            "pair": get_pair_label(pair),
            "timeframe": timeframe,
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
            "conditions": stats.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Analysis ────────────────────────────────────────────────────────────────

@app.get("/analysis/full", tags=["analysis"])
async def full_analysis(pair: str = PAIR_LABEL, timeframe: str = "1h", limit: int = 100):
    """
    Run full technical + liquidity + event analysis on recent bars.
    Returns the enriched DataFrame as JSON.
    """
    try:
        pair = normalize_pair(pair)
        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe)
        df = compute_all_indicators(df)
        df = compute_all_liquidity(df)

        events = load_event_calendar()
        df = flag_event_windows(df, events)

        df_out = df.tail(limit).copy()
        df_out.index = df_out.index.astype(str)

        # Convert non-JSON-serializable types
        for col in df_out.columns:
            if df_out[col].dtype.name == "category":
                df_out[col] = df_out[col].astype(str)

        return {
            "pair": get_pair_label(pair),
            "timeframe": timeframe,
            "bars": len(df_out),
            "data": df_out.reset_index().fillna(0).to_dict(orient="records"),
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
        }
    except Exception as e:
        logger.error(f"Full analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/session-stats", tags=["analysis"])
async def session_statistics(pair: str = PAIR_LABEL, timeframe: str = "1h"):
    """Return per-session liquidity and volatility statistics."""
    try:
        from backend.modules.liquidity.analysis import compute_session_stats
        pair = normalize_pair(pair)
        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe)
        df = compute_all_indicators(df)
        df = compute_all_liquidity(df)
        stats = compute_session_stats(df)
        return stats.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Backtest ────────────────────────────────────────────────────────────────

@app.post("/backtest/run", tags=["backtest"])
async def run_backtest_endpoint(
    req: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Run full backtest on EUR/USD data. May take 30–120 seconds.
    Returns complete performance metrics.
    """
    try:
        pair = normalize_pair(req.pair)
        df = _enrich_df(pair, req.timeframe, force_refresh=req.force_refresh)
        direction_bias = get_direction_bias(pair, req.timeframe)

        result = run_backtest(
            df, strategy_name=req.strategy_name,
            timeframe=req.timeframe, rr=req.rr,
            direction_bias=direction_bias,
            recent_signals=req.recent_signals,
        )

        # If recent-signals scope has no closed trades, fall back to full history.
        # This avoids confusing all-zero metrics in UI when the latest N window is empty.
        fallback_used = False
        if (
            result.analysis_scope == "recent_signals"
            and result.total_signals == 0
            and result.recent_signals_used == 0
        ):
            logger.warning(
                f"No closed trades in recent scope for {pair} {req.timeframe}; "
                "falling back to full-history metrics."
            )
            result = run_backtest(
                df,
                strategy_name=req.strategy_name,
                timeframe=req.timeframe,
                rr=req.rr,
                direction_bias=direction_bias,
                recent_signals=0,
            )
            fallback_used = True

        # Cache win rates for signal engine only for full-history backtests.
        if result.analysis_scope == "full_history":
            _app_state.setdefault("backtest_win_rates", {})[pair] = {
                "BUY": result.buy_win_rate,
                "SELL": result.sell_win_rate,
                "BUY_count": result.buy_signals,
                "SELL_count": result.sell_signals,
            }

        # Exclude trade log from main response for brevity
        result_dict = {k: v for k, v in result.__dict__.items() if k != "trades"}
        result_dict["trade_count"] = len(result.trades)
        result_dict["direction_bias"] = direction_bias
        result_dict["fallback_from_recent_scope"] = fallback_used

        return {
            "status": "complete",
            "pair": get_pair_label(pair),
            "disclaimer": "Backtest results are hypothetical. Past performance does not guarantee future results.",
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
            **result_dict,
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/trades", tags=["backtest"])
async def backtest_trades(req: BacktestRequest = Depends()):
    """Return detailed trade log from the last backtest."""
    try:
        pair = normalize_pair(req.pair)
        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=req.timeframe)
        df = compute_all_indicators(df)
        df = compute_all_liquidity(df)
        events = load_event_calendar()
        df = flag_event_windows(df, events)
        result = run_backtest(df, strategy_name=req.strategy_name, timeframe=req.timeframe)
        return {"trades": result.trades[:200]}  # cap at 200
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Model Training & Prediction ─────────────────────────────────────────────

@app.post("/model/train", tags=["model"])
async def train_model_endpoint(req: TrainRequest):
    """
    Train the ML direction probability model on 5 years of EUR/USD data.
    Training takes 2–5 minutes. Saves model to disk.
    """
    try:
        pair = normalize_pair(req.pair)
        lookback_years = _training_lookback_years(req.timeframe)
        df = fetch_ohlcv(
            symbol=get_symbol_for_pair(pair),
            interval=req.timeframe,
            years=lookback_years,
            force_refresh=req.force_refresh,
        )
        df = compute_all_indicators(df)
        df = compute_all_liquidity(df)
        events = load_event_calendar()
        df = flag_event_windows(df, events)

        X, y = build_training_dataset(df, timeframe=req.timeframe)
        train_result = train_model(X, y, pair=pair, timeframe=req.timeframe, save=True)

        _app_state.setdefault("model_bundles", {})[pair] = train_result["model_bundle"]

        return {
            "status": "trained",
            "pair": get_pair_label(pair),
            "samples": len(X),
            "features": len(train_result["features"]),
            "lookback_years": lookback_years,
            "cv_mean_auc": train_result["cv_mean_auc"],
            "test_auc": train_result["test_auc"],
            "feature_importance_top10": dict(
                list(train_result["feature_importance"].items())[:10]
            ),
            "disclaimer": (
                "AUC > 0.5 indicates some predictive ability above random. "
                "This does NOT mean the model is profitable. "
                "Use in conjunction with all other signal conditions."
            ),
        }
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/predict", tags=["model"])
async def predict_endpoint(pair: str = PAIR_LABEL, timeframe: str = "1h"):
    """Get probability prediction for the current market bar."""
    pair = normalize_pair(pair)
    bundle = _get_model_bundle(pair)
    if not bundle:
        raise HTTPException(status_code=400, detail="Model not trained. Run POST /model/train first.")

    try:
        df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe)
        df = compute_all_indicators(df)
        df = compute_all_liquidity(df)
        events = load_event_calendar()
        df = flag_event_windows(df, events)

        latest = df.iloc[-1]
        prediction = predict_probability(latest, bundle)
        return {
            "pair": get_pair_label(pair),
            "timeframe": timeframe,
            "timestamp": str(df.index[-1]),
            "current_price": float(latest["close"]),
            "model_timeframe": bundle.get("timeframe", "unknown"),
            **prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Signals ─────────────────────────────────────────────────────────────────

@app.get("/signals/latest", tags=["signals"])
async def get_latest_signals(
    pair: str = PAIR_LABEL,
    timeframe: str = INTERVAL_PRIMARY,
    scan_bars: int = 3,
    signal_mode: str = DEFAULT_SIGNAL_MODE,
):
    """
    Scan recent bars for active trading signals.
    Returns only high-confidence signals that pass all quality gates.
    """
    pair = normalize_pair(pair)
    timeframe = _validate_signal_pair_timeframe(pair, timeframe)
    bundle = _get_model_bundle(pair)
    win_rates = _get_backtest_win_rates(pair)

    try:
        cache_key = ("signals_latest", pair, timeframe, scan_bars, signal_mode)
        cached = _get_cached_response(cache_key, ttl=20)
        if cached is not None:
            return cached

        df = _enrich_df_cached(pair, timeframe)
        diagnostics = timeframe_signal_diagnostics(
            df=df,
            model_bundle=bundle,
            pair=pair,
            timeframe=timeframe,
            signal_mode=signal_mode,
        )

        signals = scan_for_signals(
            df,
            backtest_win_rates=win_rates,
            model_bundle=bundle,
            pair=pair,
            timeframe=timeframe,
            last_n_bars=scan_bars,
            signal_mode=signal_mode,
        )

        response = {
            "pair": get_pair_label(pair),
            "timeframe": timeframe,
            "signal_mode": signal_mode,
            "timestamp": datetime.utcnow().isoformat(),
            "signal_count": len(signals),
            "signals": [s.to_dict() for s in signals],
            "confirmation_only": timeframe in CONFIRMATION_ONLY_TIMEFRAMES,
            "recommended_live_setup": get_recommended_live_setup(pair),
            "signal_diagnostics": diagnostics,
            "no_signal_message": (
                "No signals meet the quality threshold right now. "
                "This is normal — the system prioritises quality over quantity."
            ) if not signals else None,
            "disclaimer": (
                "All signals are probabilistic. Not financial advice. "
                "Always apply your own risk management."
            ),
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
        }
        _set_cached_response(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"Signal scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/multi-tf", tags=["signals"])
async def get_multi_tf_signals(
    pair: str = PAIR_LABEL,
    scan_bars: int = 3,
    signal_mode: str = DEFAULT_SIGNAL_MODE,
):
    """
    Scan all configured timeframes (1D, 1H, 15M) for signals.
    Each signal includes risk classification and risk calculation breakdown.
    Risky timeframes (1H, 15M) are flagged with explicit warnings.
    """
    pair = normalize_pair(pair)
    bundle = _get_model_bundle(pair)
    pair_tf_configs = get_multi_tf_configs(pair)
    allowed_timeframes = _allowed_timeframes_for_pair(pair)
    if not allowed_timeframes:
        raise HTTPException(status_code=400, detail=f"Pair '{pair}' is not allowed for signals.")

    try:
        cache_key = ("signals_multi_tf", pair, scan_bars, signal_mode)
        cached = _get_cached_response(cache_key, ttl=20)
        if cached is not None:
            return cached

        signals = scan_multi_tf_signals(
            enrich_fn=_enrich_df_cached,
            model_bundle=bundle,
            pair=pair,
            timeframes=allowed_timeframes,
            last_n_bars=min(scan_bars, MAX_SCAN_BARS),
            signal_mode=signal_mode,
        )

        # Safety filter in case downstream logic returns unexpected timeframes.
        signals = [s for s in signals if str(s.timeframe).lower() in allowed_timeframes]

        best_timeframe = max(
            allowed_timeframes,
            key=lambda tf: (
                pair_tf_configs.get(tf, {}).get("total_trades", 0) >= 5,
                pair_tf_configs.get(tf, {}).get("data_years", 0.0) >= 1.0,
                pair_tf_configs.get(tf, {}).get("backtest_win_rate", 0.0),
                pair_tf_configs.get(tf, {}).get("backtest_profit_factor", 0.0),
                -pair_tf_configs.get(tf, {}).get("backtest_max_dd_pct", 100.0),
            ),
        )
        best_config = pair_tf_configs.get(best_timeframe, {})
        timeframe_overview = [
            _build_timeframe_overview(
                pair,
                tf,
                model_bundle=bundle,
                signal_mode=signal_mode,
            )
            for tf in allowed_timeframes
        ]

        # Group by risk level for summary
        risk_summary = {}
        for s in signals:
            level = s.risk_level
            if level not in risk_summary:
                risk_summary[level] = {"count": 0, "timeframes": set()}
            risk_summary[level]["count"] += 1
            risk_summary[level]["timeframes"].add(s.timeframe)
        # Convert sets to lists for JSON
        for v in risk_summary.values():
            v["timeframes"] = sorted(v["timeframes"])

        response = {
            "pair": get_pair_label(pair),
            "scan_type": "multi-timeframe",
            "signal_mode": signal_mode,
            "scan_bars": min(scan_bars, MAX_SCAN_BARS),
            "timeframes_scanned": allowed_timeframes,
            "timestamp": datetime.utcnow().isoformat(),
            "signal_count": len(signals),
            "risk_summary": risk_summary,
            "timeframe_overview": timeframe_overview,
            "confirmation_only_timeframes": CONFIRMATION_ONLY_TIMEFRAMES,
            "recommended_live_setup": get_recommended_live_setup(pair),
            "best_timeframe": {
                "timeframe": best_timeframe,
                "win_rate": best_config.get("backtest_win_rate", 0.0),
                "profit_factor": best_config.get("backtest_profit_factor", 0.0),
                "max_drawdown_pct": best_config.get("backtest_max_dd_pct", 0.0),
                "rr_ratio": best_config.get("rr_ratio", 1.0),
                "risk_level": best_config.get("risk_level", "UNKNOWN"),
            },
            "signals": [s.to_dict() for s in signals],
            "no_signal_message": (
                "No signals meet the quality threshold on any timeframe right now. "
                "This is normal — the system prioritises quality over quantity."
            ) if not signals else None,
            "disclaimer": (
                "All signals are probabilistic. Not financial advice. "
                "Signals from HIGH and VERY_HIGH risk timeframes should be treated "
                "as INFORMATIONAL ONLY. Always apply your own risk management."
            ),
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
        }
        _set_cached_response(cache_key, response)
        return response
    except Exception as e:
        logger.error(f"Multi-TF signal scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Dashboard Summary ────────────────────────────────────────────────────────

@app.get("/dashboard/summary", tags=["dashboard"])
async def dashboard_summary(pair: str = PAIR_LABEL):
    """Quick summary for the Streamlit dashboard home page."""
    try:
        pair = normalize_pair(pair)
        cache_key = ("dashboard_summary", pair, INTERVAL_PRIMARY)
        cached = _get_cached_response(cache_key, ttl=5)
        if cached is not None:
            return cached

        df = _enrich_df_cached(pair, INTERVAL_PRIMARY)
        latest = df.iloc[-1]
        analysis_price = round(float(latest["close"]), 5)
        analysis_timestamp = str(df.index[-1])

        live_quote = None
        try:
            live_quote = fetch_live_quote(symbol=get_symbol_for_pair(pair))
        except Exception as exc:
            logger.warning(f"Live quote fetch failed for {pair}: {exc}")

        response = {
            "pair": get_pair_label(pair),
            "display_name": get_pair_display_name(pair),
            "current_price": round(live_quote["price"], 5) if live_quote else analysis_price,
            "live_price": round(live_quote["price"], 5) if live_quote else analysis_price,
            "live_price_time": live_quote["timestamp"] if live_quote else analysis_timestamp,
            "live_price_source": live_quote.get("source_name", "yfinance_1m") if live_quote else "analysis_fallback",
            "analysis_price": analysis_price,
            "analysis_time": analysis_timestamp,
            "live_analysis_delta": round((live_quote["price"] - analysis_price), 6) if live_quote else 0.0,
            "current_session": str(latest.get("session_primary", "unknown")),
            "liquidity_zone": str(latest.get("liquidity_zone", "unknown")),
            "trend_primary": str(latest.get("trend_primary", "unknown")),
            "rsi": round(float(latest.get("rsi_14", 0)), 2),
            "macd_histogram": round(float(latest.get("macd_histogram", 0)), 6),
            "atr": round(float(latest.get("atr", 0)), 6),
            "volatility_regime": str(latest.get("volatility_regime", "unknown")),
            "volume_condition": str(latest.get("vol_condition", "unknown")),
            "event_window": bool(latest.get("event_window", False)),
            "model_loaded": _get_model_bundle(pair) is not None,
            "recommended_live_setup": get_recommended_live_setup(pair),
            "last_updated": analysis_timestamp,
            "volume_disclaimer": get_volume_disclaimer_for_pair(pair),
        }
        _set_cached_response(cache_key, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Optimization ─────────────────────────────────────────────────────────────

@app.get("/optimization/results", tags=["optimization"])
async def optimization_results(pair: str = PAIR_LABEL):
    """Return cached optimization results if available."""
    import json
    from backend.core.config import BASE_DIR
    pair = normalize_pair(pair)
    safe_pair = pair.replace("/", "_")
    results_path = BASE_DIR / "reports" / "optimization" / f"optimization_results_{safe_pair}.json"
    if pair == PAIR_LABEL and not results_path.exists():
        legacy_path = BASE_DIR / "reports" / "optimization" / "optimization_results.json"
        if legacy_path.exists():
            results_path = legacy_path
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return {
            "status": "cached",
            "pair": get_pair_label(pair),
            "total_combos": len(data),
            "results": data,
            "best": data[0] if data else None,
        }
    raise HTTPException(status_code=404, detail="No optimization results. Run optimizer first.")


@app.post("/optimization/run", tags=["optimization"])
async def run_optimization_endpoint(req: OptimizationRequest):
    """Run multi-TF multi-RR optimization (takes several minutes)."""
    try:
        pair = normalize_pair(req.pair)
        results = run_optimization(pair=pair, timeframes=req.timeframes, score_mode=req.score_mode)
        best = get_best_config(results)
        return {
            "status": "complete",
            "pair": get_pair_label(pair),
            "score_mode": req.score_mode,
            "total_combos": len(results),
            "best": best,
        }
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
