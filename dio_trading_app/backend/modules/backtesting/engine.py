"""
Dio Trading App — Backtesting Engine
======================================
Simulates trades from historical signals and computes:
  buy/sell counts, win/loss rates, profit factor, max drawdown,
  Sharpe ratio, expected value, and full equity curve.

Rules enforced:
  - Every trade has SL and TP
  - Risk/reward is fixed at 1:2
  - Position size = (capital × risk_pct) / (SL distance in price)
  - No look-ahead bias: signals use only data available at bar close

DISCLAIMER:
  Backtesting results are hypothetical and do not guarantee
  future performance. Past win rates are probabilistic guides only.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger

from backend.core.config import (
    INITIAL_CAPITAL, RISK_PER_TRADE_PCT,
    RISK_REWARD_RATIO, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER,
    VOLUME_DISCLAIMER, TIMEFRAME_DIRECTION_BIAS
)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    bar_index: int
    entry_time: pd.Timestamp
    direction: str          # "BUY" | "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float = 2.0
    confidence: float = 0.0
    session: str = ""
    original_sl_distance: float = 0.0  # stored at entry for position sizing

    # Filled on close
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None   # "WIN" | "LOSS"
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    position_size: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    timeframe: str
    start_date: str
    end_date: str

    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    wins: int = 0
    losses: int = 0
    buy_wins: int = 0
    sell_wins: int = 0

    buy_win_rate: float = 0.0
    sell_win_rate: float = 0.0
    total_win_rate: float = 0.0
    total_loss_rate: float = 0.0

    gross_profit_usd: float = 0.0
    gross_loss_usd: float = 0.0
    profit_pct: float = 0.0
    loss_pct: float = 0.0
    expected_value: float = 0.0
    profit_factor: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    initial_capital: float = INITIAL_CAPITAL
    final_capital: float = INITIAL_CAPITAL
    total_return_pct: float = 0.0

    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0

    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    analysis_scope: str = "full_history"   # full_history | recent_signals
    recent_signals_requested: int = 0
    recent_signals_used: int = 0

    volume_disclaimer: str = VOLUME_DISCLAIMER


# ─── Signal Generator Stub ────────────────────────────────────────────────────

def _generate_signals_for_backtest(
    df: pd.DataFrame,
    timeframe: str = "1h",
    direction_bias: str | None = None,
) -> pd.DataFrame:
    """
    Generate raw candidate signals from the feature-enriched DataFrame.
    Returns a copy of df with a 'signal' column: +1=BUY, -1=SELL, 0=none.

    This is a multi-condition rule-based approach with strict confluence.
    All conditions must be True simultaneously for a signal to fire.

    This function is used ONLY for backtesting. Production signals
    come from the full SignalEngine which adds ML probability and
    confidence scoring before output.
    """
    df = df.copy()
    df["signal"] = 0

    required_cols = [
        "rsi_14", "macd_line", "macd_signal", "ema_9", "ema_21",
        "ema_bull_align", "ema_bear_align", "atr",
        "session_primary", "liquidity_zone", "event_window",
        "trend_primary", "buy_confluence", "sell_confluence",
    ]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing column {col} — backtest signal generation skipped.")
            return df

    # ── Momentum confirmation ─────────────────────────────────────
    # Require price moving in our direction over last 3 bars
    momentum_up = df["close"] > df["close"].shift(3)
    momentum_down = df["close"] < df["close"].shift(3)

    # ── Volatility filter: skip very low ATR periods (choppy) ────
    atr_sma = df["atr"].rolling(20).mean()
    adequate_vol = df["atr"] > atr_sma * 0.7

    # ── Trend strength: require EMAs to be sufficiently spread ────
    if "trend_strength" in df.columns:
        strong_trend = df["trend_strength"] > 0.1  # EMA spread > 0.1% of price
    else:
        strong_trend = pd.Series(True, index=df.index)

    # ── Market regime filter: skip ranging/choppy markets ─────────
    if "market_regime" in df.columns:
        not_ranging = df["market_regime"] != "ranging"
    else:
        not_ranging = pd.Series(True, index=df.index)

    # ── RSI divergence boost: track RSI trend vs price ────────────
    rsi_rising = df["rsi_14"] > df["rsi_14"].shift(3)
    rsi_falling = df["rsi_14"] < df["rsi_14"].shift(3)

    # ── BUY conditions (stricter) ─────────────────────────────────
    buy_mask = (
        # Trend alignment: EMAs bullish + primary trend bullish
        df["ema_bull_align"] &
        (df["trend_primary"] == "bullish") &
        # RSI: in sweet spot (not overbought or deep oversold)
        (df["rsi_14"] > 40) & (df["rsi_14"] < 62) &
        # RSI should be rising (momentum confirmation)
        rsi_rising &
        # MACD bullish and histogram expanding
        (df["macd_line"] > df["macd_signal"]) &
        (df.get("macd_histogram", pd.Series(0, index=df.index)) > 0) &
        # Higher confluence requirement
        (df["buy_confluence"] >= 3) &
        # Momentum: price moving up
        momentum_up &
        # Trend filter
        strong_trend &
        not_ranging &
        # Volatility filter
        adequate_vol &
        # Good liquidity session
        (df["liquidity_zone"].isin(["high", "normal"])) &
        ~df["event_window"] &
        ~df["price_at_upper_bb"] &
        df["above_ema50"] &
        # Price above SMA200 for macro confirmation
        df.get("above_sma200", pd.Series(True, index=df.index))
    )

    # ── SELL conditions (stricter) ────────────────────────────────
    sell_mask = (
        df["ema_bear_align"] &
        (df["trend_primary"] == "bearish") &
        (df["rsi_14"] < 60) & (df["rsi_14"] > 38) &
        rsi_falling &
        (df["macd_line"] < df["macd_signal"]) &
        (df.get("macd_histogram", pd.Series(0, index=df.index)) < 0) &
        (df["sell_confluence"] >= 3) &
        momentum_down &
        strong_trend &
        not_ranging &
        adequate_vol &
        (df["liquidity_zone"].isin(["high", "normal"])) &
        ~df["event_window"] &
        ~df["price_at_lower_bb"] &
        ~df["above_ema50"] &
        # Price below SMA200 for macro confirmation
        ~df.get("above_sma200", pd.Series(False, index=df.index))
    )

    resolved_bias = direction_bias or TIMEFRAME_DIRECTION_BIAS.get(timeframe, "BOTH")
    if resolved_bias == "SELL_ONLY":
        buy_mask = buy_mask & False
    elif resolved_bias == "BUY_ONLY":
        sell_mask = sell_mask & False

    # Avoid consecutive signals (min gap: 8 bars for less noise)
    signal_col = df["signal"].copy()
    last_signal_bar = -10
    for i, idx in enumerate(df.index):
        if i - last_signal_bar < 8:
            continue
        if buy_mask.iloc[i]:
            signal_col.iloc[i] = 1
            last_signal_bar = i
        elif sell_mask.iloc[i]:
            signal_col.iloc[i] = -1
            last_signal_bar = i

    df["signal"] = signal_col
    return df


# ─── Trade Simulator ──────────────────────────────────────────────────────────

def simulate_trades(
    df: pd.DataFrame,
    capital: float = INITIAL_CAPITAL,
    risk_pct: float = RISK_PER_TRADE_PCT,
    rr: float = RISK_REWARD_RATIO,
    sl_atr_mult: float = SL_ATR_MULTIPLIER,
    tp_atr_mult: float | None = None,
) -> tuple[list[Trade], list[float]]:
    """
    Walk forward through the DataFrame, opening and closing trades.
    Returns (trade_list, equity_curve).

    Trade management:
    - SL = entry ± ATR × sl_atr_mult
    - TP = entry ± ATR × (sl_atr_mult × rr)   [or tp_atr_mult if given]
    - One trade at a time (no pyramiding)
    - Trade closed when price hits SL or TP on a future bar
    """
    if tp_atr_mult is None:
        tp_atr_mult = sl_atr_mult * rr

    trades = []
    equity = [capital]
    current_capital = capital
    open_trade: Optional[Trade] = None

    pip_value = 10.0  # USD per pip per standard lot (EUR/USD)

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # ── Check if open trade hits SL or TP ────────────────────
        if open_trade is not None:
            hit_tp = hit_sl = False

            if open_trade.direction == "BUY":
                if row["low"] <= open_trade.stop_loss:
                    hit_sl = True
                elif row["high"] >= open_trade.take_profit:
                    hit_tp = True
            else:  # SELL
                if row["high"] >= open_trade.stop_loss:
                    hit_sl = True
                elif row["low"] <= open_trade.take_profit:
                    hit_tp = True

            if hit_tp or hit_sl:
                if hit_tp:
                    exit_price = open_trade.take_profit
                    open_trade.outcome = "WIN"
                    pips = abs(exit_price - open_trade.entry_price) * 10_000
                else:
                    exit_price = open_trade.stop_loss
                    open_trade.outcome = "LOSS"
                    pips = -abs(exit_price - open_trade.entry_price) * 10_000

                open_trade.exit_price = exit_price
                open_trade.exit_time = df.index[i]
                open_trade.pnl_pips = round(pips, 2)

                # PnL in USD based on position size
                sl_dist_price = open_trade.original_sl_distance
                if sl_dist_price > 0:
                    position_size = (current_capital * risk_pct) / (sl_dist_price * 10_000)
                    open_trade.position_size = round(position_size, 4)
                    open_trade.pnl_usd = round(pips * pip_value * open_trade.position_size, 2)
                else:
                    open_trade.pnl_usd = 0.0

                current_capital += open_trade.pnl_usd
                equity.append(current_capital)
                trades.append(open_trade)
                open_trade = None
            else:
                equity.append(current_capital)

        # ── Open a new trade if signal fires ─────────────────────
        elif prev_row.get("signal", 0) != 0 and open_trade is None:
            direction = "BUY" if prev_row["signal"] == 1 else "SELL"
            entry = row["open"]
            atr = prev_row.get("atr", 0.0005)

            if direction == "BUY":
                sl = entry - atr * sl_atr_mult
                tp = entry + atr * tp_atr_mult
            else:
                sl = entry + atr * sl_atr_mult
                tp = entry - atr * tp_atr_mult

            open_trade = Trade(
                bar_index=i,
                entry_time=df.index[i],
                direction=direction,
                entry_price=entry,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                risk_reward=rr,
                confidence=prev_row.get("confidence_score", 0.0),
                session=prev_row.get("session_primary", ""),
                original_sl_distance=abs(entry - sl),
            )
        else:
            equity.append(current_capital)

    # Close any still-open trade at market at end
    if open_trade is not None:
        last_price = df.iloc[-1]["close"]
        open_trade.exit_price = last_price
        open_trade.exit_time = df.index[-1]
        open_trade.outcome = "OPEN"
        trades.append(open_trade)

    return trades, equity


# ─── Performance Metrics ──────────────────────────────────────────────────────

def compute_metrics(
    trades: list[Trade],
    equity: list[float],
    strategy_name: str,
    timeframe: str,
    df: pd.DataFrame,
) -> BacktestResult:
    """Compute all performance statistics from completed trades."""

    result = BacktestResult(
        strategy_name=strategy_name,
        timeframe=timeframe,
        start_date=str(df.index.min()),
        end_date=str(df.index.max()),
        initial_capital=INITIAL_CAPITAL,
        final_capital=equity[-1] if equity else INITIAL_CAPITAL,
        equity_curve=equity,
    )

    closed = [t for t in trades if t.outcome in ("WIN", "LOSS")]
    if not closed:
        logger.warning("No closed trades in backtest — metrics will be zero.")
        return result

    buys = [t for t in closed if t.direction == "BUY"]
    sells = [t for t in closed if t.direction == "SELL"]
    wins = [t for t in closed if t.outcome == "WIN"]
    losses = [t for t in closed if t.outcome == "LOSS"]

    result.total_signals = len(closed)
    result.buy_signals = len(buys)
    result.sell_signals = len(sells)
    result.wins = len(wins)
    result.losses = len(losses)
    result.buy_wins = len([t for t in buys if t.outcome == "WIN"])
    result.sell_wins = len([t for t in sells if t.outcome == "WIN"])

    result.buy_win_rate = round(result.buy_wins / len(buys), 4) if buys else 0.0
    result.sell_win_rate = round(result.sell_wins / len(sells), 4) if sells else 0.0
    result.total_win_rate = round(len(wins) / len(closed), 4)
    result.total_loss_rate = round(1 - result.total_win_rate, 4)

    result.gross_profit_usd = sum(t.pnl_usd for t in wins)
    result.gross_loss_usd = abs(sum(t.pnl_usd for t in losses))

    result.profit_pct = round(result.gross_profit_usd / INITIAL_CAPITAL * 100, 4) if INITIAL_CAPITAL else 0.0
    result.loss_pct = round(result.gross_loss_usd / INITIAL_CAPITAL * 100, 4) if INITIAL_CAPITAL else 0.0

    result.profit_factor = (
        round(result.gross_profit_usd / result.gross_loss_usd, 3)
        if result.gross_loss_usd > 0 else float("inf")
    )

    avg_win = np.mean([t.pnl_usd for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl_usd for t in losses])) if losses else 0
    result.expected_value = round(
        result.total_win_rate * avg_win - result.total_loss_rate * avg_loss, 4
    )

    result.avg_win_pips = round(np.mean([t.pnl_pips for t in wins]), 2) if wins else 0.0
    result.avg_loss_pips = round(np.mean([abs(t.pnl_pips) for t in losses]), 2) if losses else 0.0

    # Max Drawdown
    eq_arr = np.array(equity)
    rolling_max = np.maximum.accumulate(eq_arr)
    drawdowns = rolling_max - eq_arr
    result.max_drawdown = round(float(drawdowns.max()), 4)
    result.max_drawdown_pct = round(float((drawdowns / rolling_max).max() * 100), 4)

    # Sharpe Ratio (annualised, based on timeframe)
    pnl_series = np.array([t.pnl_usd for t in closed])
    if pnl_series.std() > 0:
        bars_per_year_map = {
            "5m": 252 * 24 * 12,  # ~72k
            "15m": 252 * 24 * 4,  # ~24k
            "1h": 252 * 8,        # ~2016
            "2h": 252 * 4,        # ~1008
            "4h": 252 * 2,        # ~504
            "1d": 252,
            "1wk": 52,
            "1mo": 12,
        }
        bars_per_year = bars_per_year_map.get(timeframe, 252 * 8)
        result.sharpe_ratio = round(
            float(pnl_series.mean() / pnl_series.std() * np.sqrt(bars_per_year)), 4
        )

    result.total_return_pct = round(
        (result.final_capital - result.initial_capital) / result.initial_capital * 100, 4
    )
    result.trades = [asdict(t) for t in trades]

    return result


# ─── Main Entry Point ────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy_name: str = "DioMultiCondition_v1",
    timeframe: str = "1h",
    rr: float = RISK_REWARD_RATIO,
    sl_atr_mult: float = SL_ATR_MULTIPLIER,
    direction_bias: str | None = None,
    recent_signals: int | None = None,
) -> BacktestResult:
    """
    Full backtest pipeline:
    1. Generate signals from feature-enriched df
    2. Simulate trades with given R:R
    3. Compute performance metrics
    """
    logger.info(f"Starting backtest: {strategy_name} on {timeframe} RR=1:{rr} ({len(df)} bars)")

    df_signals = _generate_signals_for_backtest(
        df,
        timeframe=timeframe,
        direction_bias=direction_bias,
    )
    n_signals = (df_signals["signal"] != 0).sum()
    logger.info(f"Generated {n_signals} candidate signals")

    trades, equity = simulate_trades(
        df_signals, rr=rr, sl_atr_mult=sl_atr_mult,
    )

    if recent_signals and recent_signals > 0:
        closed_trades = [t for t in trades if t.outcome in ("WIN", "LOSS")]
        scoped_trades = closed_trades[-recent_signals:]

        scoped_equity = [INITIAL_CAPITAL]
        running_capital = INITIAL_CAPITAL
        for trade in scoped_trades:
            running_capital += trade.pnl_usd
            scoped_equity.append(running_capital)

        result = compute_metrics(scoped_trades, scoped_equity, strategy_name, timeframe, df)
        result.analysis_scope = "recent_signals"
        result.recent_signals_requested = int(recent_signals)
        result.recent_signals_used = len(scoped_trades)
    else:
        result = compute_metrics(trades, equity, strategy_name, timeframe, df)

    logger.success(
        f"Backtest complete — Signals: {result.total_signals} | "
        f"Win rate: {result.total_win_rate:.1%} | "
        f"Profit factor: {result.profit_factor:.2f} | "
        f"Max DD: {result.max_drawdown_pct:.1f}%"
    )
    return result
