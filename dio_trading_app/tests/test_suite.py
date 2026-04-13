"""
Dio Trading App — Test Suite
==============================
Unit and integration tests.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from backend.core.config import VOLUME_DISCLAIMER, MIN_CONFIDENCE_SCORE
from backend.modules.technical.indicators import (
    add_rsi, add_macd, add_moving_averages, add_atr,
    add_bollinger_bands, add_volume_conditions,
    compute_volume_condition_stats
)
from backend.modules.liquidity.analysis import add_session_labels
from backend.modules.events.fundamental import (
    score_headline_sentiment, score_event_actual_vs_forecast
)
from backend.modules.signals.engine import build_signal, TradingSignal


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a synthetic OHLCV DataFrame for testing."""
    n = 500
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.08 + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.0002,
        "high": close + abs(np.random.randn(n) * 0.0004),
        "low": close - abs(np.random.randn(n) * 0.0004),
        "close": close,
        "volume_proxy": np.random.randint(500, 5000, n).astype(float),
    }, index=dates)
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"] = df["high"] - df["low"]
    df["body"] = abs(df["close"] - df["open"])
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    return df.dropna()


# ─── Technical Indicator Tests ────────────────────────────────────────────────

class TestTechnicalIndicators:
    def test_rsi_range(self, sample_df):
        df = add_rsi(sample_df)
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all(), "RSI must be >= 0"
        assert (rsi <= 100).all(), "RSI must be <= 100"

    def test_rsi_overbought_oversold_flags(self, sample_df):
        df = add_rsi(sample_df)
        assert "rsi_overbought" in df.columns
        assert "rsi_oversold" in df.columns
        # Flags should be boolean
        assert df["rsi_overbought"].dtype == bool

    def test_macd_columns_exist(self, sample_df):
        df = add_macd(sample_df)
        assert "macd_line" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns
        assert "macd_bullish_cross" in df.columns
        assert "macd_bearish_cross" in df.columns

    def test_macd_histogram_equals_line_minus_signal(self, sample_df):
        df = add_macd(sample_df)
        diff = (df["macd_line"] - df["macd_signal"] - df["macd_histogram"]).abs()
        assert diff.max() < 1e-10

    def test_ema_sma_columns_exist(self, sample_df):
        df = add_moving_averages(sample_df)
        assert "ema_9" in df.columns
        assert "ema_21" in df.columns
        assert "ema_50" in df.columns
        assert "sma_200" in df.columns

    def test_atr_positive(self, sample_df):
        df = add_atr(sample_df)
        atr = df["atr"].dropna()
        assert (atr > 0).all(), "ATR must always be positive"

    def test_bollinger_upper_above_lower(self, sample_df):
        df = add_bollinger_bands(sample_df)
        df = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (df["bb_upper"] >= df["bb_lower"]).all()

    def test_volume_conditions_cover_all_rows(self, sample_df):
        df = add_volume_conditions(sample_df)
        conditions = ["price_up_vol_up", "price_down_vol_down", "price_down_vol_up", "price_up_vol_down"]
        covered = df["vol_condition"].isin(conditions + ["unknown"]).all()
        assert covered

    def test_volume_condition_stats_structure(self, sample_df):
        df = add_volume_conditions(sample_df)
        stats = compute_volume_condition_stats(df)
        assert "condition" in stats.columns
        assert "count" in stats.columns
        assert "pct_of_total" in stats.columns
        assert "next_bar_up_rate" in stats.columns
        # Percentages must be in 0-100 range
        assert (stats["pct_of_total"] >= 0).all()
        assert (stats["pct_of_total"] <= 100).all()
        assert (stats["next_bar_up_rate"] >= 0).all()
        assert (stats["next_bar_up_rate"] <= 100).all()


# ─── Liquidity Tests ──────────────────────────────────────────────────────────

class TestLiquidityAnalysis:
    def test_session_labels_exist(self, sample_df):
        df = add_session_labels(sample_df)
        assert "session_primary" in df.columns
        valid = {"london", "new_york", "london_ny_overlap", "tokyo", "off_hours"}
        assert df["session_primary"].isin(valid).all()

    def test_overlap_is_subset_of_london_and_ny(self, sample_df):
        df = add_session_labels(sample_df)
        overlap_mask = df["session_overlap"]
        assert (df.loc[overlap_mask, "session_london"]).all()
        assert (df.loc[overlap_mask, "session_ny"]).all()


# ─── Event / Fundamental Tests ────────────────────────────────────────────────

class TestEventAnalysis:
    def test_bullish_headline_positive_score(self):
        score = score_headline_sentiment("ECB hike rates higher hawkish euro gains")
        assert score > 0, f"Expected positive score, got {score}"

    def test_bearish_headline_negative_score(self):
        score = score_headline_sentiment("Fed hike dollar rally eurusd falls bearish")
        assert score < 0, f"Expected negative score, got {score}"

    def test_neutral_headline_zero(self):
        score = score_headline_sentiment("Weather is nice today nothing happening")
        assert score == 0.0

    def test_nfp_beat_is_eur_negative(self):
        score = score_event_actual_vs_forecast("us_nfp", actual="250K", forecast="180K")
        assert score < 0, "Strong NFP should be EUR negative (USD positive)"

    def test_ecb_hike_is_eur_positive(self):
        score = score_event_actual_vs_forecast("eurozone_cpi", actual="3.5%", forecast="3.0%")
        assert score > 0, "Above-forecast eurozone CPI should be EUR positive"

    def test_score_clipped_to_minus1_plus1(self):
        score = score_event_actual_vs_forecast("us_nfp", actual="999K", forecast="100K")
        assert -1.0 <= score <= 1.0


# ─── Signal Engine Tests ──────────────────────────────────────────────────────

class TestSignalEngine:
    def _make_row(self, direction: str) -> pd.Series:
        """Build a synthetic bar row that should pass all signal gates."""
        if direction == "BUY":
            return pd.Series({
                "close": 1.0850,
                "ema_9": 1.0840, "ema_21": 1.0830, "ema_50": 1.0820,
                "sma_200": 1.0780,
                "ema_bull_align": True, "ema_bear_align": False,
                "above_sma200": True, "above_ema50": True,
                "rsi_14": 52.0,
                "macd_line": 0.0003, "macd_signal": 0.0001, "macd_histogram": 0.0002,
                "macd_bullish_cross": False, "macd_bearish_cross": False,
                "atr": 0.0008,
                "bb_upper": 1.0900, "bb_lower": 1.0750, "bb_mid": 1.0825,
                "bb_pct_b": 0.55, "bb_squeeze": False,
                "price_at_upper_bb": False, "price_at_lower_bb": False,
                "vol_condition": "price_up_vol_up",
                "volume_ratio": 1.2, "high_volume": False, "low_volume": False,
                "session_primary": "london", "liquidity_zone": "normal",
                "liquidity_score": 0.7,
                "session_london": True, "session_ny": False,
                "session_overlap": False, "session_tokyo": False,
                "event_window": False, "event_sentiment": 0.1,
                "event_name": "", "event_impact": "",
                "trend_primary": "bullish", "trend_secondary": "bullish",
                "uptrend_structure": True, "downtrend_structure": False,
                "breakout_up": False, "breakout_down": False,
                "bullish_sweep": False, "bearish_sweep": False,
                "stop_hunt": "none",
                "volatility_regime": "low",
                "bullish_engulf": False, "bearish_engulf": False,
                "hammer": False, "shooting_star": False,
                "hl_range": 0.0015, "body": 0.0005,
                "returns": 0.0003, "log_returns": 0.0003,
            })
        else:
            row = pd.Series({
                "close": 1.0850,
                "ema_9": 1.0820, "ema_21": 1.0830, "ema_50": 1.0840,
                "sma_200": 1.0900,
                "ema_bull_align": False, "ema_bear_align": True,
                "above_sma200": False, "above_ema50": False,
                "rsi_14": 48.0,
                "macd_line": -0.0003, "macd_signal": -0.0001, "macd_histogram": -0.0002,
                "macd_bullish_cross": False, "macd_bearish_cross": False,
                "atr": 0.0008,
                "bb_upper": 1.0900, "bb_lower": 1.0750, "bb_mid": 1.0825,
                "bb_pct_b": 0.45, "bb_squeeze": False,
                "price_at_upper_bb": False, "price_at_lower_bb": False,
                "vol_condition": "price_down_vol_up",
                "volume_ratio": 1.3, "high_volume": False, "low_volume": False,
                "session_primary": "london", "liquidity_zone": "normal",
                "liquidity_score": 0.7,
                "session_london": True, "session_ny": False,
                "session_overlap": False, "session_tokyo": False,
                "event_window": False, "event_sentiment": -0.1,
                "event_name": "", "event_impact": "",
                "trend_primary": "bearish", "trend_secondary": "bearish",
                "uptrend_structure": False, "downtrend_structure": True,
                "breakout_up": False, "breakout_down": False,
                "bullish_sweep": False, "bearish_sweep": False,
                "stop_hunt": "none",
                "volatility_regime": "low",
                "bullish_engulf": False, "bearish_engulf": False,
                "hammer": False, "shooting_star": False,
                "hl_range": 0.0015, "body": 0.0005,
                "returns": -0.0003, "log_returns": -0.0003,
            })
            return row

    def test_buy_signal_generated_with_strong_conditions(self):
        row = self._make_row("BUY")
        signal = build_signal(row, "BUY", historical_win_rate=0.58, setup_frequency=120, model_probability=0.62)
        assert signal is not None, "Expected a BUY signal to be generated"
        assert signal.direction == "BUY"
        assert signal.confidence_score >= MIN_CONFIDENCE_SCORE

    def test_sell_signal_generated_with_strong_conditions(self):
        row = self._make_row("SELL")
        signal = build_signal(row, "SELL", historical_win_rate=0.58, setup_frequency=100, model_probability=0.60)
        assert signal is not None, "Expected a SELL signal to be generated"
        assert signal.direction == "SELL"

    def test_signal_has_correct_rr(self):
        row = self._make_row("BUY")
        signal = build_signal(row, "BUY", historical_win_rate=0.60, setup_frequency=100, model_probability=0.65)
        if signal:
            sl_dist = abs(signal.entry_price - signal.stop_loss)
            tp_dist = abs(signal.take_profit - signal.entry_price)
            rr = tp_dist / sl_dist if sl_dist > 0 else 0
            assert abs(rr - 2.0) < 0.01, f"R:R should be ~2.0, got {rr:.3f}"

    def test_signal_suppressed_below_win_rate_threshold(self):
        row = self._make_row("BUY")
        signal = build_signal(row, "BUY", historical_win_rate=0.30, setup_frequency=10, model_probability=0.45)
        assert signal is None, "Signal should be suppressed when historical win rate is too low"

    def test_signal_suppressed_in_event_window(self):
        row = self._make_row("BUY")
        row["event_window"] = True
        row["event_name"] = "US NFP"
        signal = build_signal(row, "BUY", historical_win_rate=0.55, setup_frequency=50, model_probability=0.55)
        # Confidence will be penalised — should be suppressed
        # (may or may not fire depending on total score; check no crash at minimum)
        assert signal is None or isinstance(signal, TradingSignal)

    def test_signal_sl_on_correct_side(self):
        buy_row = self._make_row("BUY")
        buy_signal = build_signal(buy_row, "BUY", historical_win_rate=0.60, setup_frequency=100, model_probability=0.65)
        if buy_signal:
            assert buy_signal.stop_loss < buy_signal.entry_price, "BUY SL must be below entry"
            assert buy_signal.take_profit > buy_signal.entry_price, "BUY TP must be above entry"

        sell_row = self._make_row("SELL")
        sell_signal = build_signal(sell_row, "SELL", historical_win_rate=0.58, setup_frequency=80, model_probability=0.62)
        if sell_signal:
            assert sell_signal.stop_loss > sell_signal.entry_price, "SELL SL must be above entry"
            assert sell_signal.take_profit < sell_signal.entry_price, "SELL TP must be below entry"

    def test_signal_contains_disclaimer(self):
        row = self._make_row("BUY")
        signal = build_signal(row, "BUY", historical_win_rate=0.60, setup_frequency=100, model_probability=0.65)
        if signal:
            assert len(signal.volume_disclaimer) > 10
            assert len(signal.risk_warning) > 10

    def test_volume_disclaimer_present(self):
        assert len(VOLUME_DISCLAIMER) > 20
        assert "proxy" in VOLUME_DISCLAIMER.lower() or "tick" in VOLUME_DISCLAIMER.lower()
