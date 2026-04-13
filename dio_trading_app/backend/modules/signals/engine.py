"""
Dio Trading App — Signal Engine
=================================
Generates high-quality BUY/SELL signals for EUR/USD by combining:
  1. Technical indicator confluence
  2. Volume condition analysis (proxy volume)
  3. Liquidity & session context
  4. Fundamental / event risk
  5. Historical backtesting win rates
  6. ML model probability

A signal is only output when ALL of these gates pass:
  - Minimum confidence score (configurable threshold)
  - Minimum historical win rate for this setup
  - Not in a high-impact event window
  - Adequate liquidity (not off-hours)
  - Acceptable volatility regime

Every signal includes:
  - Direction, entry, SL, TP
  - Risk:reward (always 1:2)
  - Confidence score (0–100)
  - Winning probability (based on historical + ML)
  - Explanation of why signal fired
  - Explanation of what invalidates the trade

DISCLAIMER: This is not financial advice. All signals are probabilistic.
No signal guarantees profit. Always apply your own risk management.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backend.core.config import (
    MIN_CONFIDENCE_SCORE, MIN_WIN_RATE_THRESHOLD,
    RISK_REWARD_RATIO, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
    VOLUME_DISCLAIMER, DEFAULT_MULTI_TF_SCAN, TIMEFRAME_SIGNAL_EXPIRY_BARS, CONFIRMATION_ONLY_TIMEFRAMES,
    get_pair_label, get_timeframe_config, get_direction_bias, get_multi_tf_configs,
    get_volume_disclaimer_for_pair, get_signal_thresholds,
)


# ─── Signal Data Model ────────────────────────────────────────────────────────

@dataclass
class TradingSignal:
    # Identity
    generated_at: str = ""
    pair: str = "EUR/USD"
    timeframe: str = "1h"

    # Direction
    direction: str = ""       # "BUY" | "SELL"

    # Prices
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 2.0

    # Confidence & Probability
    confidence_score: float = 0.0          # 0–100 composite score
    win_probability: float = 0.0           # Final combined win probability
    historical_win_rate: float = 0.0       # From backtesting
    model_probability: float = 0.0         # From ML model
    setup_frequency: int = 0               # How often this setup appeared historically

    # Context
    session: str = ""
    market_condition: str = ""
    trend_direction: str = ""
    liquidity_zone: str = ""
    volatility_regime: str = ""
    event_window: bool = False

    # Explanation
    signal_reason: str = ""
    invalidation_reason: str = ""
    contributing_factors: list = field(default_factory=list)

    # Risk Assessment (multi-TF)
    risk_level: str = "LOW"            # LOW / MEDIUM / HIGH / VERY_HIGH
    risk_score: float = 0.0            # 0–100 (higher = riskier)
    risk_details: dict = field(default_factory=dict)
    risk_notification: str = ""        # Human-readable risk warning
    age_bars: int = 0
    expires_after_bars: int = 0
    is_fresh: bool = True
    confluence_passed: bool = True
    confluence_with: list = field(default_factory=list)
    stale_reason: str = ""

    # Metadata
    volume_disclaimer: str = VOLUME_DISCLAIMER
    risk_warning: str = (
        "This signal is probabilistic and NOT financial advice. "
        "Trading this market involves significant risk of loss. "
        "Never risk more than you can afford to lose."
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Condition Scoring ────────────────────────────────────────────────────────

def _score_technical_conditions(row: pd.Series, direction: str) -> tuple[float, list[str]]:
    """
    Score the strength of technical conditions for a direction.
    Returns (score 0–40, list of contributing factors).
    """
    score = 0.0
    factors = []

    if direction == "BUY":
        # EMA alignment
        if row.get("ema_bull_align", False):
            score += 10
            factors.append("EMA 9>21>50 bullish alignment")

        # RSI zone
        rsi = row.get("rsi_14", 50)
        if 40 <= rsi <= 60:
            score += 8
            factors.append(f"RSI at neutral-positive zone ({rsi:.1f})")
        elif 30 <= rsi < 40:
            score += 6
            factors.append(f"RSI recovering from oversold ({rsi:.1f})")

        # MACD
        if row.get("macd_line", 0) > row.get("macd_signal", 0):
            score += 6
            factors.append("MACD line above signal — bullish momentum")
        if row.get("macd_bullish_cross", False):
            score += 4
            factors.append("MACD bullish crossover just fired")

        # Price above SMA200
        if row.get("above_sma200", False):
            score += 6
            factors.append("Price above SMA200 — macro bullish")

        # BB position
        if 0.4 <= row.get("bb_pct_b", 0.5) <= 0.7:
            score += 4
            factors.append("Price in upper-mid Bollinger zone")

        # Trend structure
        if row.get("uptrend_structure", False):
            score += 4
            factors.append("Higher highs and higher lows structure")

        # Sweep reversal boost
        if row.get("stop_hunt") == "bullish_sweep":
            score += 6
            factors.append("Bullish liquidity sweep detected — potential smart money reversal")

    else:  # SELL
        if row.get("ema_bear_align", False):
            score += 10
            factors.append("EMA 9<21<50 bearish alignment")

        rsi = row.get("rsi_14", 50)
        if 40 <= rsi <= 60:
            score += 8
            factors.append(f"RSI at neutral-negative zone ({rsi:.1f})")
        elif rsi > 60 and rsi <= 70:
            score += 6
            factors.append(f"RSI retreating from overbought ({rsi:.1f})")

        if row.get("macd_line", 0) < row.get("macd_signal", 0):
            score += 6
            factors.append("MACD line below signal — bearish momentum")
        if row.get("macd_bearish_cross", False):
            score += 4
            factors.append("MACD bearish crossover just fired")

        if not row.get("above_sma200", True):
            score += 6
            factors.append("Price below SMA200 — macro bearish")

        if 0.3 <= row.get("bb_pct_b", 0.5) <= 0.6:
            score += 4
            factors.append("Price in lower-mid Bollinger zone")

        if row.get("downtrend_structure", False):
            score += 4
            factors.append("Lower highs and lower lows structure")

        if row.get("stop_hunt") == "bearish_sweep":
            score += 6
            factors.append("Bearish liquidity sweep detected — potential smart money reversal")

    return min(score, 40.0), factors


def _score_volume_conditions(row: pd.Series, direction: str) -> tuple[float, list[str]]:
    """Score volume-price alignment. Max 20 points."""
    score = 0.0
    factors = []

    vol_cond = row.get("vol_condition", "unknown")
    high_vol = row.get("high_volume", False)

    if direction == "BUY":
        if vol_cond == "price_up_vol_up":
            score += 15
            factors.append("Price up + proxy volume up — bullish confirmation [proxy vol only]")
        elif vol_cond == "price_down_vol_down":
            score += 10
            factors.append("Price down + proxy volume down — weak sell, possible base building")
        if high_vol:
            score += 5
            factors.append("Above-average proxy volume — increased conviction")
    else:  # SELL
        if vol_cond == "price_down_vol_up":
            score += 15
            factors.append("Price down + proxy volume up — bearish confirmation [proxy vol only]")
        elif vol_cond == "price_up_vol_down":
            score += 10
            factors.append("Price up + proxy volume down — weak buy, possible exhaustion")
        if high_vol:
            score += 5
            factors.append("Above-average proxy volume — increased conviction")

    return min(score, 20.0), factors


def _score_liquidity(row: pd.Series) -> tuple[float, list[str]]:
    """Score liquidity context. Max 20 points."""
    score = 0.0
    factors = []

    session = row.get("session_primary", "off_hours")
    liq_zone = row.get("liquidity_zone", "normal")

    session_scores = {
        "london_ny_overlap": 20,
        "london": 16,
        "new_york": 14,
        "tokyo": 8,
        "off_hours": 4,
    }
    session_score = session_scores.get(session, 10)
    score += session_score
    factors.append(f"Session: {session} (liquidity score {session_score}/20)")

    if liq_zone == "low":
        score = max(score - 10, 0)
        factors.append("Low liquidity zone detected — signal confidence reduced")
    elif liq_zone == "high":
        score = min(score + 5, 20)
        factors.append("High liquidity zone — favorable execution conditions")

    return min(score, 20.0), factors


def _score_fundamental(row: pd.Series, direction: str) -> tuple[float, list[str]]:
    """Score fundamental / event alignment. Max 20 points."""
    score = 10.0  # neutral baseline
    factors = []

    if row.get("event_window", False):
        score -= 8
        event_name = row.get("event_name", "High-impact event")
        factors.append(f"WARNING: Inside {event_name} event window — high risk")

    sentiment = row.get("event_sentiment", 0.0)
    if direction == "BUY" and sentiment > 0.3:
        score += 5
        factors.append(f"Fundamental sentiment supports EUR bullish (score: {sentiment:.2f})")
    elif direction == "SELL" and sentiment < -0.3:
        score += 5
        factors.append(f"Fundamental sentiment supports EUR bearish (score: {sentiment:.2f})")
    elif direction == "BUY" and sentiment < -0.3:
        score -= 5
        factors.append(f"Fundamental sentiment opposes BUY — caution (score: {sentiment:.2f})")
    elif direction == "SELL" and sentiment > 0.3:
        score -= 5
        factors.append(f"Fundamental sentiment opposes SELL — caution (score: {sentiment:.2f})")

    return max(min(score, 20.0), 0.0), factors


# ─── Risk Calculation ─────────────────────────────────────────────────────────

def calculate_timeframe_risk(timeframe: str, pair: str = "EUR/USD") -> tuple[str, float, dict, str]:
    """
    Calculate risk level for a given timeframe based on backtest metrics.

    Returns (risk_level, risk_score 0-100, risk_details dict, risk_notification str).
    Higher risk_score = MORE risky.
    """
    cfg = get_timeframe_config(pair, timeframe)
    if not cfg:
        # Unknown timeframe — default to high risk
        return "HIGH", 75.0, {"note": "No backtest data for this timeframe"}, (
            f"⚠ HIGH RISK: No optimized backtest data available for {timeframe}. "
            "Trade with extreme caution."
        )

    # ── Component scores (each 0–25, higher = riskier) ──
    # 1. Max drawdown risk (0-25)
    dd = cfg["backtest_max_dd_pct"]
    dd_score = min(dd / 100 * 25, 25)

    # 2. Profit factor risk (0-25): PF < 1 = very risky, PF > 1.5 = low risk
    pf = cfg["backtest_profit_factor"]
    if pf >= 1.5:
        pf_score = 0
    elif pf >= 1.0:
        pf_score = (1.5 - pf) / 0.5 * 15  # 0–15 for PF 1.0–1.5
    else:
        pf_score = 15 + (1.0 - pf) / 0.5 * 10  # 15–25 for PF 0.5–1.0

    # 3. Data limitation risk (0-25): less data = more risk
    data_yrs = cfg["data_years"]
    if data_yrs >= 3.0:
        data_score = 0
    elif data_yrs >= 1.0:
        data_score = (3.0 - data_yrs) / 2.0 * 12  # 0–12
    else:
        data_score = 12 + (1.0 - data_yrs) * 13  # 12–25

    # 4. Return/Sharpe risk (0-25)
    sharpe = cfg["backtest_sharpe"]
    ret = cfg["backtest_return_pct"]
    sharpe_score = 0
    if sharpe < 0:
        sharpe_score += 15
    elif sharpe < 0.5:
        sharpe_score += 10
    elif sharpe < 1.0:
        sharpe_score += 5

    if ret < 0:
        sharpe_score += 10
    elif ret < 20:
        sharpe_score += 5
    sharpe_score = min(sharpe_score, 25)

    # Total risk score 0–100
    risk_score = round(dd_score + pf_score + data_score + sharpe_score, 1)

    # Determine level
    if risk_score <= 25:
        risk_level = "LOW"
    elif risk_score <= 50:
        risk_level = "MEDIUM"
    elif risk_score <= 75:
        risk_level = "HIGH"
    else:
        risk_level = "VERY_HIGH"

    # Override from config if it's worse
    config_level = cfg["risk_level"]
    level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "VERY_HIGH": 3}
    if level_order.get(config_level, 0) > level_order.get(risk_level, 0):
        risk_level = config_level

    risk_details = {
        "timeframe": timeframe,
        "risk_score": risk_score,
        "drawdown_risk": round(dd_score, 1),
        "profit_factor_risk": round(pf_score, 1),
        "data_limitation_risk": round(data_score, 1),
        "sharpe_return_risk": round(sharpe_score, 1),
        "backtest_max_dd_pct": dd,
        "backtest_profit_factor": pf,
        "backtest_sharpe": cfg["backtest_sharpe"],
        "backtest_return_pct": ret,
        "backtest_win_rate": cfg["backtest_win_rate"],
        "data_years": data_yrs,
        "total_trades": cfg["total_trades"],
        "rr_ratio": cfg["rr_ratio"],
    }

    # Build human-readable notification
    level_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "VERY_HIGH": "⛔"}
    emoji = level_emoji.get(risk_level, "⚠")

    if risk_level == "LOW":
        notification = (
            f"{emoji} LOW RISK ({timeframe}): Well-tested on {data_yrs:.1f} years of data. "
            f"Win rate {cfg['backtest_win_rate']:.1%}, Profit Factor {pf:.2f}, "
            f"Max DD {dd:.1f}%, Sharpe {cfg['backtest_sharpe']:.2f}. "
            f"This is the recommended primary timeframe."
        )
    elif risk_level == "MEDIUM":
        notification = (
            f"{emoji} MEDIUM RISK ({timeframe}): Moderate risk profile. "
            f"Win rate {cfg['backtest_win_rate']:.1%}, Profit Factor {pf:.2f}, "
            f"Max DD {dd:.1f}%. Use with standard position sizing."
        )
    elif risk_level == "HIGH":
        notification = (
            f"{emoji} HIGH RISK ({timeframe}): This timeframe showed {dd:.1f}% max drawdown "
            f"in backtesting with only {pf:.2f} profit factor (barely break-even). "
            f"Win rate {cfg['backtest_win_rate']:.1%}, Sharpe {cfg['backtest_sharpe']:.2f}. "
            f"REDUCE position size to ≤0.5% risk per trade. "
            f"Only {data_yrs:.1f} years of backtest data available."
        )
    else:  # VERY_HIGH
        if ret < 0 or pf < 1.0:
            notification = (
                f"{emoji} VERY HIGH RISK ({timeframe}): EXTREME CAUTION! "
                f"Backtest showed weak or negative performance (Return {ret:.1f}%, PF {pf:.2f}) "
                f"with {dd:.1f}% max drawdown. "
                f"Only {data_yrs:.1f} years ({int(data_yrs * 365)} days) of data available. "
                f"Risk Score: {risk_score}/100. "
                f"This signal is INFORMATIONAL ONLY — trading this timeframe is NOT recommended."
            )
        else:
            notification = (
                f"{emoji} VERY HIGH RISK ({timeframe}): EXTREME CAUTION! "
                f"This setup may look strong (Return {ret:.1f}%, PF {pf:.2f}, WR {cfg['backtest_win_rate']:.1%}) "
                f"but the sample is too small to trust. "
                f"Only {cfg['total_trades']} trades across {data_yrs:.1f} years "
                f"({int(data_yrs * 365)} days) of data are available. "
                f"Risk Score: {risk_score}/100. "
                f"This signal is INFORMATIONAL ONLY — trading this timeframe is NOT recommended."
            )

    return risk_level, risk_score, risk_details, notification


# ─── Signal Builder ───────────────────────────────────────────────────────────

def build_signal(
    row: pd.Series,
    direction: str,
    historical_win_rate: float,
    setup_frequency: int,
    model_probability: float,
    pair: str = "EUR/USD",
    timeframe: str = "1h",
    signal_mode: str = "balanced",
) -> Optional[TradingSignal]:
    """
    Build a complete TradingSignal from a bar row.
    Returns None if confidence or probability gates are not met.
    """

    # ── Compute component scores ──────────────────────────────────
    tech_score, tech_factors = _score_technical_conditions(row, direction)
    vol_score, vol_factors = _score_volume_conditions(row, direction)
    liq_score, liq_factors = _score_liquidity(row)
    fund_score, fund_factors = _score_fundamental(row, direction)

    # Confidence = weighted sum (max 100)
    confidence = tech_score + vol_score + liq_score + fund_score
    all_factors = tech_factors + vol_factors + liq_factors + fund_factors

    # ── Win probability: weighted average of historical + model ───
    # Give more weight to historical backtest (more reliable than ML for forex)
    win_probability = (historical_win_rate * 0.60) + (model_probability * 0.40)

    # ── Confidence boost/penalty from trend alignment ─────────────
    # Reward strong multi-indicator confluence
    if tech_score >= 30:
        confidence = min(confidence + 5, 100)
        all_factors.append("Strong technical confluence bonus (+5)")
    # Penalize weak setups
    if tech_score < 15:
        confidence -= 10
        all_factors.append("Weak technical setup penalty (-10)")

    # ── Gate checks ───────────────────────────────────────────────
    direction_bias = get_direction_bias(pair, timeframe)
    if direction_bias == "SELL_ONLY" and direction == "BUY":
        logger.debug(f"Signal suppressed — {timeframe} is configured SELL_ONLY for higher historical accuracy")
        return None
    if direction_bias == "BUY_ONLY" and direction == "SELL":
        logger.debug(f"Signal suppressed — {timeframe} is configured BUY_ONLY for higher historical accuracy")
        return None

    thresholds = get_signal_thresholds(pair, signal_mode)
    min_confidence = thresholds.get("min_confidence", MIN_CONFIDENCE_SCORE)
    min_win_rate = thresholds.get("min_win_rate", MIN_WIN_RATE_THRESHOLD)
    min_model_probability = thresholds.get("min_model_probability", 0.50)

    if confidence < min_confidence:
        logger.debug(f"Signal suppressed — confidence {confidence:.1f} < {min_confidence}")
        return None

    if historical_win_rate < min_win_rate:
        logger.debug(f"Signal suppressed — historical win rate {historical_win_rate:.2%} too low")
        return None

    if model_probability < min_model_probability:
        logger.debug(
            f"Signal suppressed — model probability {model_probability:.2%} below {min_model_probability:.2%}"
        )
        return None

    if row.get("event_window", False) and confidence < 75:
        logger.debug("Signal suppressed — event window risk too high for this confidence level")
        return None

    if row.get("liquidity_zone") == "low" and confidence < 80:
        logger.debug("Signal suppressed — low liquidity + insufficient confidence")
        return None

    if str(row.get("volatility_regime", "")) == "high" and confidence < 75:
        logger.debug("Signal suppressed — high volatility regime requires higher confidence")
        return None

    # ── Compute prices (use per-TF config if available) ─────────
    entry = row["close"]
    atr = row.get("atr", 0.0005)

    tf_cfg = get_timeframe_config(pair, timeframe)
    sl_mult = tf_cfg.get("sl_atr_mult", SL_ATR_MULTIPLIER)
    tp_mult = tf_cfg.get("tp_atr_mult", TP_ATR_MULTIPLIER)
    rr = tf_cfg.get("rr_ratio", RISK_REWARD_RATIO)

    if direction == "BUY":
        sl = entry - atr * sl_mult
        tp = entry + atr * tp_mult
    else:
        sl = entry + atr * sl_mult
        tp = entry - atr * tp_mult

    # ── Build invalidation reason ─────────────────────────────────
    move_unit = "pips" if get_pair_label(pair) == "EUR/USD" else "USD"
    move_scale = 10_000 if move_unit == "pips" else 1

    if direction == "BUY":
        invalidation = (
            f"Trade is invalidated if price closes below stop loss at {sl:.5f} "
            f"(−{atr * sl_mult * move_scale:.1f} {move_unit}). "
            "Also invalidated by: break below EMA50, RSI drop below 35, "
            "unexpected bearish high-impact news, or session closes without reaching TP."
        )
    else:
        invalidation = (
            f"Trade is invalidated if price closes above stop loss at {sl:.5f} "
            f"(+{atr * sl_mult * move_scale:.1f} {move_unit}). "
            "Also invalidated by: break above EMA50, RSI rising above 65, "
            "unexpected bullish high-impact news, or session closes without reaching TP."
        )

    reason = (
        f"Multi-condition {direction} signal on {get_pair_label(pair)} [{timeframe}]. "
        f"Technical score: {tech_score:.0f}/40, Volume score: {vol_score:.0f}/20, "
        f"Liquidity score: {liq_score:.0f}/20, Fundamental score: {fund_score:.0f}/20. "
        f"Historical win rate for similar setups: {historical_win_rate:.1%} "
        f"({setup_frequency} occurrences). "
        f"ML model probability: {model_probability:.1%}. "
        f"Combined win probability: {win_probability:.1%}. "
        f"Key factors: {'; '.join(all_factors[:4])}."
    )

    # ── Calculate risk for this timeframe ────────────────────────
    risk_level, risk_score, risk_details, risk_notification = calculate_timeframe_risk(timeframe, pair=pair)

    # Append risk info to reason for risky timeframes
    if risk_level in ("HIGH", "VERY_HIGH"):
        reason += f" ⚠ RISK WARNING: {risk_notification}"

    signal = TradingSignal(
        generated_at=datetime.utcnow().isoformat() + "Z",
        pair=get_pair_label(pair),
        timeframe=timeframe,
        direction=direction,
        entry_price=round(entry, 5),
        stop_loss=round(sl, 5),
        take_profit=round(tp, 5),
        risk_reward=rr,
        confidence_score=round(confidence, 2),
        win_probability=round(win_probability, 4),
        historical_win_rate=round(historical_win_rate, 4),
        model_probability=round(model_probability, 4),
        setup_frequency=setup_frequency,
        session=str(row.get("session_primary", "")),
        market_condition=str(row.get("vol_condition", "")),
        trend_direction=str(row.get("trend_primary", "")),
        liquidity_zone=str(row.get("liquidity_zone", "")),
        volatility_regime=str(row.get("volatility_regime", "")),
        event_window=bool(row.get("event_window", False)),
        signal_reason=reason,
        invalidation_reason=invalidation,
        contributing_factors=all_factors,
        risk_level=risk_level,
        risk_score=risk_score,
        risk_details=risk_details,
        risk_notification=risk_notification,
        volume_disclaimer=get_volume_disclaimer_for_pair(pair),
    )

    return signal


def diagnose_signal_setup(
    row: pd.Series,
    direction: str,
    historical_win_rate: float,
    setup_frequency: int,
    model_probability: float,
    pair: str = "EUR/USD",
    timeframe: str = "1h",
    signal_mode: str = "balanced",
) -> dict:
    """Return diagnostics explaining why a setup passed or failed on the latest bar."""
    tech_score, _ = _score_technical_conditions(row, direction)
    vol_score, _ = _score_volume_conditions(row, direction)
    liq_score, _ = _score_liquidity(row)
    fund_score, _ = _score_fundamental(row, direction)

    confidence = tech_score + vol_score + liq_score + fund_score
    if tech_score >= 30:
        confidence = min(confidence + 5, 100)
    if tech_score < 15:
        confidence -= 10

    blockers = []
    direction_bias = get_direction_bias(pair, timeframe)
    thresholds = get_signal_thresholds(pair, signal_mode)
    min_confidence = thresholds.get("min_confidence", MIN_CONFIDENCE_SCORE)
    min_win_rate = thresholds.get("min_win_rate", MIN_WIN_RATE_THRESHOLD)
    min_model_probability = thresholds.get("min_model_probability", 0.50)

    if direction_bias == "SELL_ONLY" and direction == "BUY":
        blockers.append("Directional bias allows SELL only")
    if direction_bias == "BUY_ONLY" and direction == "SELL":
        blockers.append("Directional bias allows BUY only")
    if confidence < min_confidence:
        blockers.append(f"Confidence too low ({confidence:.1f} < {min_confidence:.1f})")
    if historical_win_rate < min_win_rate:
        blockers.append(
            f"Historical win rate too low ({historical_win_rate:.1%} < {min_win_rate:.1%})"
        )
    if model_probability < min_model_probability:
        blockers.append(
            f"Model probability too low ({model_probability:.1%} < {min_model_probability:.1%})"
        )
    if row.get("event_window", False) and confidence < 75:
        blockers.append("Event window active and confidence is below 75")
    if row.get("liquidity_zone") == "low" and confidence < 80:
        blockers.append("Low liquidity with insufficient confidence")
    if str(row.get("volatility_regime", "")) == "high" and confidence < 75:
        blockers.append("High volatility regime needs higher confidence")

    return {
        "direction": direction,
        "eligible": len(blockers) == 0,
        "confidence": round(confidence, 2),
        "historical_win_rate": round(historical_win_rate, 4),
        "model_probability": round(model_probability, 4),
        "setup_frequency": setup_frequency,
        "component_scores": {
            "technical": round(tech_score, 2),
            "volume": round(vol_score, 2),
            "liquidity": round(liq_score, 2),
            "fundamental": round(fund_score, 2),
        },
        "blockers": blockers,
    }


def timeframe_signal_diagnostics(
    df: pd.DataFrame,
    model_bundle: Optional[dict],
    pair: str,
    timeframe: str,
    signal_mode: str = "balanced",
) -> dict:
    """Return latest-bar BUY/SELL diagnostics for one timeframe."""
    from backend.modules.prediction.model import predict_probability

    latest = df.iloc[-1]
    tf_cfg = get_timeframe_config(pair, timeframe)
    historical_wr = tf_cfg.get("backtest_win_rate", 0.50)
    setup_frequency = max(tf_cfg.get("total_trades", 0) // 2, 0)

    bull_probability = 0.50
    bear_probability = 0.50
    if model_bundle:
        try:
            pred = predict_probability(latest, model_bundle)
            bull_probability = pred.get("bull_probability", 0.50)
            bear_probability = pred.get("bear_probability", 0.50)
        except Exception as exc:
            logger.warning(f"Model prediction failed during diagnostics for {timeframe}: {exc}")

    buy_diag = diagnose_signal_setup(
        row=latest,
        direction="BUY",
        historical_win_rate=historical_wr,
        setup_frequency=setup_frequency,
        model_probability=bull_probability,
        pair=pair,
        timeframe=timeframe,
        signal_mode=signal_mode,
    )
    sell_diag = diagnose_signal_setup(
        row=latest,
        direction="SELL",
        historical_win_rate=historical_wr,
        setup_frequency=setup_frequency,
        model_probability=bear_probability,
        pair=pair,
        timeframe=timeframe,
        signal_mode=signal_mode,
    )

    return {
        "timeframe": timeframe,
        "latest_bar_time": str(df.index[-1]),
        "buy": buy_diag,
        "sell": sell_diag,
        "summary": "Signal available"
        if buy_diag["eligible"] or sell_diag["eligible"]
        else (buy_diag["blockers"] + sell_diag["blockers"])[0],
    }


# ─── Signal Scanner ───────────────────────────────────────────────────────────

def scan_for_signals(
    df: pd.DataFrame,
    backtest_win_rates: dict,
    model_bundle: Optional[dict],
    pair: str = "EUR/USD",
    timeframe: str = "1h",
    last_n_bars: int = 3,
    signal_mode: str = "balanced",
) -> list[TradingSignal]:
    """
    Scan the most recent bars of a feature-enriched DataFrame for signals.

    backtest_win_rates: dict with keys "BUY" and "SELL" from BacktestResult.
    model_bundle: loaded ML model dict or None.
    last_n_bars: how many recent bars to scan (typically 1–3).

    Returns a list of TradingSignal objects (may be empty).
    """
    from backend.modules.prediction.model import predict_probability

    signals = []
    scan_rows = df.tail(last_n_bars)
    expiry_bars = TIMEFRAME_SIGNAL_EXPIRY_BARS.get(timeframe, 2)

    for age_bars, (_, row) in enumerate(scan_rows.iloc[::-1].iterrows()):
        for direction in ["BUY", "SELL"]:
            # Get historical win rate from backtest
            hist_wr = backtest_win_rates.get(direction, 0.50)
            setup_freq = backtest_win_rates.get(f"{direction}_count", 0)

            # Get model probability
            if model_bundle:
                try:
                    pred = predict_probability(row, model_bundle)
                    if direction == "BUY":
                        model_prob = pred["bull_probability"]
                    else:
                        model_prob = pred["bear_probability"]
                except Exception as e:
                    logger.warning(f"Model prediction failed: {e}")
                    model_prob = 0.50
            else:
                model_prob = 0.50  # no model: use neutral

            signal = build_signal(
                row=row,
                direction=direction,
                historical_win_rate=hist_wr,
                setup_frequency=setup_freq,
                model_probability=model_prob,
                pair=pair,
                timeframe=timeframe,
                signal_mode=signal_mode,
            )
            if signal:
                signal.age_bars = age_bars
                signal.expires_after_bars = expiry_bars
                signal.is_fresh = age_bars <= expiry_bars
                if not signal.is_fresh:
                    signal.stale_reason = (
                        f"Signal is {age_bars} bars old on {timeframe} and expired after {expiry_bars} bars."
                    )
                    logger.info(
                        f"STALE SIGNAL DROPPED: {direction} on {timeframe} | age={age_bars} bars > {expiry_bars}"
                    )
                    continue
                logger.info(
                    f"SIGNAL GENERATED: {direction} @ {signal.entry_price:.5f} | "
                    f"Confidence: {signal.confidence_score:.1f} | "
                    f"Win prob: {signal.win_probability:.1%}"
                )
                signals.append(signal)

    return signals


# ─── Multi-Timeframe Signal Scanner ──────────────────────────────────────────

def scan_multi_tf_signals(
    enrich_fn,
    model_bundle: Optional[dict],
    pair: str = "EUR/USD",
    timeframes: list[str] | None = None,
    last_n_bars: int = 3,
    signal_mode: str = "balanced",
) -> list[TradingSignal]:
    """
    Scan multiple timeframes for signals, each with its risk classification.

    enrich_fn: callable(timeframe) -> enriched DataFrame
    model_bundle: loaded ML model dict or None
    timeframes: list of TF strings (defaults to all in MULTI_TF_CONFIGS)
    last_n_bars: how many recent bars to scan per TF

    Returns a combined list of TradingSignal objects sorted by risk_level then confidence.
    """
    if timeframes is None:
        timeframes = DEFAULT_MULTI_TF_SCAN

    signals_by_tf: dict[str, list[TradingSignal]] = {}

    for tf in timeframes:
        tf_cfg = get_timeframe_config(pair, tf)
        logger.info(f"Scanning {tf} (risk: {tf_cfg.get('risk_level', 'UNKNOWN')})...")

        try:
            df = enrich_fn(pair, tf)
        except Exception as e:
            logger.warning(f"Failed to load data for {tf}: {e}")
            continue

        # Use per-TF backtest win rates from config
        win_rates = {
            "BUY": tf_cfg.get("backtest_win_rate", 0.50),
            "SELL": tf_cfg.get("backtest_win_rate", 0.50),
            "BUY_count": tf_cfg.get("total_trades", 0) // 2,
            "SELL_count": tf_cfg.get("total_trades", 0) // 2,
        }

        signals = scan_for_signals(
            df,
            backtest_win_rates=win_rates,
            model_bundle=model_bundle,
            pair=pair,
            timeframe=tf,
            last_n_bars=last_n_bars,
            signal_mode=signal_mode,
        )

        logger.info(f"  {tf}: {len(signals)} signal(s) found")
        signals_by_tf[tf] = signals

    # Top-down confluence.
    # High-accuracy mode requires lower-timeframe signals to align with the higher timeframe.
    daily_directions = {s.direction for s in signals_by_tf.get("1d", []) if s.is_fresh}
    hourly_directions = {s.direction for s in signals_by_tf.get("1h", []) if s.is_fresh}

    all_signals: list[TradingSignal] = []
    for tf in timeframes:
        for signal in signals_by_tf.get(tf, []):
            confluence_with = []
            confluence_passed = True

            if tf == "1h":
                if daily_directions:
                    confluence_with.append("1d")
                    confluence_passed = signal.direction in daily_directions
                elif signal_mode == "high_accuracy":
                    confluence_passed = False
                    signal.stale_reason = "1H signal rejected: no 1D anchor signal for confluence."

            elif tf == "15m":
                if hourly_directions:
                    confluence_with.append("1h")
                    confluence_passed = signal.direction in hourly_directions
                    if confluence_passed and daily_directions:
                        confluence_with.append("1d")
                        confluence_passed = signal.direction in daily_directions
                elif daily_directions:
                    confluence_with.append("1d")
                    confluence_passed = signal.direction in daily_directions
                elif signal_mode == "high_accuracy":
                    confluence_passed = False
                    signal.stale_reason = "15m signal rejected: no higher-timeframe anchor signal for confluence."

            signal.confluence_with = confluence_with
            signal.confluence_passed = confluence_passed

            if tf in CONFIRMATION_ONLY_TIMEFRAMES:
                logger.info(f"CONFIRMATION-ONLY SIGNAL HIDDEN: {signal.direction} on {tf}")
                continue

            if signal_mode == "high_accuracy" and not confluence_passed:
                logger.info(
                    f"CONFLUENCE NOT MET: {signal.direction} on {tf} | anchors={confluence_with or ['none']}"
                )

            all_signals.append(signal)

    # Sort: LOW risk first, then by confidence descending
    level_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "VERY_HIGH": 3}
    all_signals.sort(
        key=lambda s: (level_order.get(s.risk_level, 9), -s.confidence_score)
    )

    return all_signals
