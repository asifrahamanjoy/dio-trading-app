"""
Dio Trading App - Core Configuration
All application-wide settings, constants, and environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for _dir in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─── Market Constants ─────────────────────────────────────────────────────────
SYMBOL = "EURUSD=X"          # Yahoo Finance ticker for EUR/USD
FUTURES_SYMBOL = "6E=F"      # EUR/USD futures (CME) — used as volume proxy
PAIR_LABEL = "EUR/USD"
LOOKBACK_YEARS = 5
INTERVAL_PRIMARY = "1d"      # Primary analysis timeframe (optimizer-selected)
INTERVAL_DAILY = "1d"
INTERVAL_4H = "4h"

# All timeframes to analyse during optimization
ALL_TIMEFRAMES = ["5m", "15m", "1h", "2h", "4h", "1d", "1wk", "1mo"]

# All risk:reward ratios to test during optimization
ALL_RR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]

# IMPORTANT DISCLAIMER — embedded in all signal outputs
VOLUME_DISCLAIMER = (
    "EUR/USD spot forex has NO centralized volume data. "
    "Volume shown is tick volume (proxy) or CME futures volume. "
    "It approximates but does NOT equal true spot market volume."
)

MARKET_DEFINITIONS = {
    "EUR/USD": {
        "symbol": SYMBOL,
        "futures_symbol": FUTURES_SYMBOL,
        "pair_label": "EUR/USD",
        "display_name": "Euro / US Dollar",
        "volume_disclaimer": VOLUME_DISCLAIMER,
    },
    "XAU/USD": {
        "symbol": "GC=F",
        "futures_symbol": "GC=F",
        "pair_label": "XAU/USD",
        "display_name": "Gold / US Dollar",
        "volume_disclaimer": (
            "Gold analysis uses COMEX gold futures data (GC=F) as a liquid proxy for XAU/USD price and volume. "
            "It is highly informative but not identical to OTC spot XAU/USD pricing."
        ),
    },
    "GBP/USD": {
        "symbol": "GBPUSD=X",
        "futures_symbol": "6B=F",
        "pair_label": "GBP/USD",
        "display_name": "British Pound / US Dollar",
        "volume_disclaimer": (
            "GBP/USD spot forex has NO centralized volume data. "
            "Volume shown is tick volume (proxy) or CME futures volume (6B=F). "
            "It approximates but does NOT equal true spot market volume."
        ),
    },
    "JPY/USD": {
        "symbol": "JPY=X",
        "futures_symbol": "6J=F",
        "pair_label": "JPY/USD",
        "display_name": "Japanese Yen / US Dollar",
        "volume_disclaimer": (
            "JPY/USD analysis uses the available JPY market feed and has NO centralized spot volume data. "
            "Volume shown is tick volume (proxy) or CME futures volume (6J=F). "
            "It approximates but does NOT equal true spot market volume."
        ),
    },
}
SUPPORTED_PAIRS = list(MARKET_DEFINITIONS.keys())

# ─── Session Windows (UTC) ────────────────────────────────────────────────────
SESSIONS = {
    "tokyo":    {"start": "00:00", "end": "09:00"},
    "london":   {"start": "07:00", "end": "16:00"},
    "new_york": {"start": "12:00", "end": "21:00"},
    "overlap_london_ny": {"start": "12:00", "end": "16:00"},
}

# ─── Signal Thresholds ────────────────────────────────────────────────────────
MIN_CONFIDENCE_SCORE = 60.0        # 0–100; signals below this are suppressed
MIN_WIN_RATE_THRESHOLD = 0.35      # Historical win rate floor
MIN_PROFIT_FACTOR = 1.1            # Backtest profit factor floor
RISK_REWARD_RATIO = 1.0            # Optimizer-selected: 1:1 R:R on 1D

# ATR multipliers for SL/TP
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 1.5            # SL × RR to maintain 1:1 R:R

# Optimizer results summary (auto-updated)
# Best:  1D + 1:1 RR — WR 58.3%, PF 1.18, Sharpe 1.30, Return +32.4%
# Alt:   1D + 1:2.5 RR — WR 37.5%, PF 1.20, Sharpe 1.36, Return +33.0%
# Alt:   1H + 1:2 RR — WR 37.2%, PF 1.02, Sharpe 0.26, Return +81.2% (high DD)

# ─── Multi-Timeframe Signal Configs (risk-annotated) ─────────────────────────
# Each config has the best R:R ratio from optimization + backtest risk metrics.
MULTI_TF_CONFIGS = {
    "1d": {
        "rr_ratio": 1.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 1.5,          # 1.5 × 1.0 RR
        "risk_level": "LOW",          # Best valid high-accuracy result
        "backtest_win_rate": 0.667,
        "backtest_profit_factor": 1.70,
        "backtest_sharpe": 4.25,
        "backtest_max_dd_pct": 10.9,
        "backtest_return_pct": 18.6,
        "data_years": 5.0,
        "total_trades": 6,
    },
    "1h": {
        "rr_ratio": 1.0,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 1.5,
        "risk_level": "HIGH",         # Better on BUY side only, still weak overall
        "backtest_win_rate": 0.518,
        "backtest_profit_factor": 0.97,
        "backtest_sharpe": -0.35,
        "backtest_max_dd_pct": 37.6,
        "backtest_return_pct": -3.5,
        "data_years": 2.0,
        "total_trades": 27,
    },
    "15m": {
        "rr_ratio": 1.5,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.25,         # 1.5 × 1.5 RR
        "risk_level": "VERY_HIGH",    # High win rate but too few trades / too little data
        "backtest_win_rate": 0.750,
        "backtest_profit_factor": 3.79,
        "backtest_sharpe": 110.0,
        "backtest_max_dd_pct": 10.0,
        "backtest_return_pct": 36.9,
        "data_years": 0.16,           # ~60 days
        "total_trades": 4,
    },
}

PAIR_MULTI_TF_CONFIGS = {
    "EUR/USD": MULTI_TF_CONFIGS,
    "XAU/USD": {
        "1d": {
            "rr_ratio": 1.5,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.25,
            "risk_level": "HIGH",
            "backtest_win_rate": 0.50,
            "backtest_profit_factor": 1.00,
            "backtest_sharpe": 0.00,
            "backtest_max_dd_pct": 25.0,
            "backtest_return_pct": 0.0,
            "data_years": 0.0,
            "total_trades": 0,
        },
        "1h": {
            "rr_ratio": 1.5,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.25,
            "risk_level": "HIGH",
            "backtest_win_rate": 0.50,
            "backtest_profit_factor": 1.00,
            "backtest_sharpe": 0.00,
            "backtest_max_dd_pct": 25.0,
            "backtest_return_pct": 0.0,
            "data_years": 0.0,
            "total_trades": 0,
        },
        "15m": {
            "rr_ratio": 1.5,
            "sl_atr_mult": 1.5,
            "tp_atr_mult": 2.25,
            "risk_level": "VERY_HIGH",
            "backtest_win_rate": 0.50,
            "backtest_profit_factor": 1.00,
            "backtest_sharpe": 0.00,
            "backtest_max_dd_pct": 30.0,
            "backtest_return_pct": 0.0,
            "data_years": 0.0,
            "total_trades": 0,
        },
    },
    "GBP/USD": MULTI_TF_CONFIGS,
    "JPY/USD": MULTI_TF_CONFIGS,
}

# Risk level thresholds (used for dynamic risk scoring)
RISK_LEVEL_THRESHOLDS = {
    "LOW":       {"max_dd": 50,  "min_pf": 1.10, "min_sharpe": 0.8, "min_data_years": 2.0},
    "MEDIUM":    {"max_dd": 65,  "min_pf": 1.00, "min_sharpe": 0.3, "min_data_years": 1.0},
    "HIGH":      {"max_dd": 85,  "min_pf": 0.90, "min_sharpe": 0.0, "min_data_years": 0.5},
    "VERY_HIGH": {"max_dd": 100, "min_pf": 0.00, "min_sharpe": -10, "min_data_years": 0.0},
}

# Directional bias from backtest results.
# 1D SELL setups materially outperformed BUY setups, so conservative mode uses SELL_ONLY.
TIMEFRAME_DIRECTION_BIAS = {
    "1d": "BOTH",
    "1h": "BOTH",
    "15m": "BOTH",
}
PAIR_TIMEFRAME_DIRECTION_BIAS = {
    "EUR/USD": TIMEFRAME_DIRECTION_BIAS,
    "XAU/USD": {
        "1d": "BOTH",
        "1h": "BOTH",
        "15m": "BOTH",
    },
    "GBP/USD": TIMEFRAME_DIRECTION_BIAS,
    "JPY/USD": TIMEFRAME_DIRECTION_BIAS,
}

DEFAULT_SIGNAL_MODE = "high_accuracy"
SIGNAL_MODES = ["balanced", "high_accuracy"]
DEFAULT_MULTI_TF_SCAN = ["15m", "1h", "1d"]
MAX_SCAN_BARS = 20
CONFIRMATION_ONLY_TIMEFRAMES = []
RECOMMENDED_LIVE_SETUP = {
    "primary_timeframe": "1d",
    "signal_mode": "high_accuracy",
    "scan_bars_min": 3,
    "scan_bars_max": 5,
    "confirmation_timeframes": [],
    "message": "Trade 1D signals in High Accuracy mode for best reliability, while 1H and 15M remain available as standalone signal views.",
}
PAIR_RECOMMENDED_LIVE_SETUP = {
    "EUR/USD": RECOMMENDED_LIVE_SETUP,
    "XAU/USD": {
        "primary_timeframe": "1d",
        "signal_mode": "balanced",
        "scan_bars_min": 2,
        "scan_bars_max": 4,
        "confirmation_timeframes": [],
        "message": "Gold is enabled with pair-specific analysis, ML probability, and backtest metrics. Run a fresh backtest and model training for XAU/USD to establish validated win-rate baselines.",
    },
    "GBP/USD": RECOMMENDED_LIVE_SETUP,
    "JPY/USD": RECOMMENDED_LIVE_SETUP,
}
TIMEFRAME_SIGNAL_EXPIRY_BARS = {
    "15m": 2,
    "1h": 2,
    "1d": 3,
}

SIGNAL_MODE_THRESHOLDS = {
    "balanced": {
        "min_confidence": MIN_CONFIDENCE_SCORE,
        "min_win_rate": MIN_WIN_RATE_THRESHOLD,
        "min_model_probability": 0.50,
    },
    "high_accuracy": {
        "min_confidence": 70.0,
        "min_win_rate": 0.55,
        "min_model_probability": 0.55,
    },
}

PAIR_SIGNAL_MODE_THRESHOLDS = {
    "EUR/USD": SIGNAL_MODE_THRESHOLDS,
    "JPY/USD": SIGNAL_MODE_THRESHOLDS,
    "GBP/USD": {
        "balanced": {
            "min_confidence": 56.0,
            "min_win_rate": 0.32,
            "min_model_probability": 0.49,
        },
        "high_accuracy": {
            "min_confidence": 67.0,
            "min_win_rate": 0.50,
            "min_model_probability": 0.54,
        },
    },
    "XAU/USD": {
        "balanced": {
            "min_confidence": 54.0,
            "min_win_rate": 0.30,
            "min_model_probability": 0.48,
        },
        "high_accuracy": {
            "min_confidence": 66.0,
            "min_win_rate": 0.48,
            "min_model_probability": 0.53,
        },
    },
}

# ─── Technical Indicator Defaults ────────────────────────────────────────────
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 9
EMA_MED = 21
EMA_LONG = 50
SMA_200 = 200
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
MOMENTUM_PERIOD = 10

# ─── Backtesting ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0   # USD
RISK_PER_TRADE_PCT = 0.01    # 1% of capital per trade

# ─── ML Model ─────────────────────────────────────────────────────────────────
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_N_ESTIMATORS = 200
FEATURE_IMPORTANCE_TOP_N = 20

# ─── Alert System ────────────────────────────────────────────────────────────
ALERT_CHECK_INTERVAL_SECONDS = 300   # Check every 5 minutes
ALERT_EMAIL_ENABLED = False          # Set True and configure .env to enable

# ─── Database ────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/data/dio_trading.db"
SYNC_DATABASE_URL = f"sqlite:///{BASE_DIR}/data/dio_trading.db"


class Settings(BaseSettings):
    """Runtime settings loadable from .env"""
    app_name: str = "Dio Trading App"
    version: str = "1.0.0"
    debug: bool = False

    # Email alerts (optional)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""

    # API keys (optional — for premium data sources)
    newsapi_key: str = ""
    alpha_vantage_key: str = ""

    class Config:
        env_file = str(BASE_DIR / ".env")
        extra = "ignore"


settings = Settings()


def normalize_pair(pair: str | None) -> str:
    if not pair:
        return PAIR_LABEL
    normalized = pair.upper().replace(" ", "")
    aliases = {
        "EUR/USD": "EUR/USD",
        "EURUSD": "EUR/USD",
        "XAU/USD": "XAU/USD",
        "XAUUSD": "XAU/USD",
        "GOLD/USD": "XAU/USD",
        "GOLD": "XAU/USD",
        "GBP/USD": "GBP/USD",
        "GBPUSD": "GBP/USD",
        "JPY/USD": "JPY/USD",
        "JPYUSD": "JPY/USD",
        "USD/JPY": "JPY/USD",
        "USDJPY": "JPY/USD",
    }
    resolved = aliases.get(normalized, pair)
    if resolved not in MARKET_DEFINITIONS:
        raise ValueError(f"Unsupported pair: {pair}")
    return resolved


def get_market_config(pair: str | None = None) -> dict:
    return MARKET_DEFINITIONS[normalize_pair(pair)]


def get_symbol_for_pair(pair: str | None = None) -> str:
    return get_market_config(pair)["symbol"]


def get_futures_symbol_for_pair(pair: str | None = None) -> str:
    return get_market_config(pair)["futures_symbol"]


def get_pair_label(pair: str | None = None) -> str:
    return get_market_config(pair)["pair_label"]


def get_pair_display_name(pair: str | None = None) -> str:
    return get_market_config(pair)["display_name"]


def get_volume_disclaimer_for_pair(pair: str | None = None) -> str:
    return get_market_config(pair)["volume_disclaimer"]


def get_multi_tf_configs(pair: str | None = None) -> dict:
    return PAIR_MULTI_TF_CONFIGS.get(normalize_pair(pair), MULTI_TF_CONFIGS)


def get_timeframe_config(pair: str | None, timeframe: str) -> dict:
    configs = get_multi_tf_configs(pair)
    return configs.get(timeframe, {
        "rr_ratio": RISK_REWARD_RATIO,
        "sl_atr_mult": SL_ATR_MULTIPLIER,
        "tp_atr_mult": TP_ATR_MULTIPLIER,
        "risk_level": "HIGH",
        "backtest_win_rate": 0.50,
        "backtest_profit_factor": 1.00,
        "backtest_sharpe": 0.00,
        "backtest_max_dd_pct": 25.0,
        "backtest_return_pct": 0.0,
        "data_years": 0.0,
        "total_trades": 0,
    })


def get_direction_bias(pair: str | None, timeframe: str) -> str:
    pair_bias = PAIR_TIMEFRAME_DIRECTION_BIAS.get(normalize_pair(pair), TIMEFRAME_DIRECTION_BIAS)
    return pair_bias.get(timeframe, "BOTH")


def get_recommended_live_setup(pair: str | None = None) -> dict:
    return PAIR_RECOMMENDED_LIVE_SETUP.get(normalize_pair(pair), RECOMMENDED_LIVE_SETUP)


def get_signal_thresholds(pair: str | None, signal_mode: str) -> dict:
    mode = signal_mode if signal_mode in SIGNAL_MODES else DEFAULT_SIGNAL_MODE
    pair_cfg = PAIR_SIGNAL_MODE_THRESHOLDS.get(normalize_pair(pair), SIGNAL_MODE_THRESHOLDS)
    return pair_cfg.get(mode, SIGNAL_MODE_THRESHOLDS[mode])
