"""
Dio Trading App — Multi-Timeframe & Multi-RR Optimizer
========================================================
Runs the full feature pipeline + backtest across every combination
of timeframe and risk:reward ratio, then ranks them by a composite
score (profit factor × Sharpe × expected value) and selects the best.

Supported timeframes: 5m, 15m, 1h, 2h, 4h, 1d, 1wk, 1mo
Supported R:R ratios: 1:1, 1:1.5, 1:2, 1:2.5, 1:3

DISCLAIMER: Past backtest results are hypothetical. They help rank
configurations but do not guarantee future returns.
"""

import json
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from backend.core.config import (
    PAIR_LABEL, ALL_TIMEFRAMES, ALL_RR_RATIOS,
    SL_ATR_MULTIPLIER, INITIAL_CAPITAL, BASE_DIR,
    get_symbol_for_pair, normalize_pair,
)
from backend.modules.data_ingestion.downloader import fetch_ohlcv
from backend.modules.preprocessing.cleaner import preprocess
from backend.modules.technical.indicators import compute_all_indicators
from backend.modules.liquidity.analysis import compute_all_liquidity
from backend.modules.events.fundamental import load_event_calendar, flag_event_windows
from backend.modules.features.engineer import engineer_all_features
from backend.modules.backtesting.engine import run_backtest


RESULTS_DIR = BASE_DIR / "reports" / "optimization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _results_paths(pair: str) -> tuple[Path, Path]:
    safe_pair = normalize_pair(pair).replace("/", "_")
    return (
        RESULTS_DIR / f"optimization_results_{safe_pair}.csv",
        RESULTS_DIR / f"optimization_results_{safe_pair}.json",
    )


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class OptimizationRow:
    timeframe: str
    rr_ratio: float
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    buy_win_rate: float
    sell_win_rate: float
    profit_factor: float
    sharpe_ratio: float
    expected_value: float
    max_drawdown_pct: float
    total_return_pct: float
    final_capital: float
    avg_win_pips: float
    avg_loss_pips: float
    composite_score: float = 0.0


# ─── Pipeline per timeframe ──────────────────────────────────────────────────

def _prepare_df(timeframe: str, pair: str = PAIR_LABEL) -> pd.DataFrame:
    """Download, preprocess, and add all features for a given timeframe."""
    logger.info(f"[Optimizer] Preparing data for {timeframe}")
    df = fetch_ohlcv(symbol=get_symbol_for_pair(pair), interval=timeframe, force_refresh=False)
    df = preprocess(df)
    df = compute_all_indicators(df)
    df = compute_all_liquidity(df)
    events = load_event_calendar()
    df = flag_event_windows(df, events)
    df = engineer_all_features(df)
    logger.info(f"[Optimizer] {timeframe}: {len(df)} bars, {len(df.columns)} columns")
    return df


# ─── Composite scoring ───────────────────────────────────────────────────────

def _composite_score(row: OptimizationRow) -> float:
    """
    Rank configurations by a weighted composite of key metrics.
    Higher is better.  Penalises configs with very few trades or
    huge drawdown.

    Components (normalised within the optimiser after all runs):
      40% — profit factor  (capped at 5)
      25% — Sharpe ratio   (capped at 5)
      20% — expected value  (per trade, in $)
      10% — win rate
       5% — trade count bonus (enough trades = more confidence)
    Penalty: −20% for max_dd > 15%, −50% for max_dd > 25%
    """
    if row.total_trades < 5:
        return -999.0  # not enough data

    pf = min(row.profit_factor, 5.0) if row.profit_factor != float("inf") else 5.0
    sr = min(max(row.sharpe_ratio, -3), 5.0)
    ev = row.expected_value
    wr = row.win_rate

    trade_bonus = min(row.total_trades / 100, 1.0)

    score = (
        0.40 * pf +
        0.25 * sr +
        0.20 * (ev / max(abs(ev), 1)) +
        0.10 * (wr * 10) +
        0.05 * trade_bonus
    )

    # Drawdown penalty
    if row.max_drawdown_pct > 25:
        score *= 0.50
    elif row.max_drawdown_pct > 15:
        score *= 0.80

    return round(score, 4)


def _win_rate_score(row: OptimizationRow) -> float:
    """
    Rank configurations with win rate as the primary objective.
    Still rewards quality via profit factor, drawdown, and enough trades.
    """
    if row.total_trades < 5:
        return -999.0

    pf = min(row.profit_factor, 3.0) if row.profit_factor != float("inf") else 3.0
    dd_penalty = max(0.0, 1.0 - (row.max_drawdown_pct / 100))
    trade_bonus = min(row.total_trades / 50, 1.0)
    sharpe_bonus = min(max(row.sharpe_ratio, 0.0), 2.0) / 2.0

    score = (
        0.60 * row.win_rate +
        0.20 * (pf / 3.0) +
        0.10 * dd_penalty +
        0.05 * trade_bonus +
        0.05 * sharpe_bonus
    )

    if row.profit_factor < 1.0:
        score *= 0.70
    if row.total_return_pct < 0:
        score *= 0.75

    return round(score, 4)


# ─── Main optimizer ──────────────────────────────────────────────────────────

def run_optimization(
    pair: str = PAIR_LABEL,
    timeframes: list[str] | None = None,
    rr_ratios: list[float] | None = None,
    sl_atr_mult: float = SL_ATR_MULTIPLIER,
    score_mode: str = "composite",
    save_results: bool = True,
) -> list[OptimizationRow]:
    """
    Run backtest for every (timeframe, rr_ratio) combination.
    Returns list of OptimizationRow sorted by composite_score descending.
    """
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES
    if rr_ratios is None:
        rr_ratios = ALL_RR_RATIOS

    pair = normalize_pair(pair)

    results: list[OptimizationRow] = []
    df_cache: dict[str, pd.DataFrame] = {}

    total_combos = len(timeframes) * len(rr_ratios)
    logger.info(f"[Optimizer] Starting {total_combos} combinations "
                f"({len(timeframes)} TFs × {len(rr_ratios)} RRs), score_mode={score_mode}")

    combo_num = 0
    for tf in timeframes:
        # Prepare data once per timeframe (reuse across R:R ratios)
        if tf not in df_cache:
            try:
                df_cache[tf] = _prepare_df(tf, pair=pair)
            except Exception as e:
                logger.error(f"[Optimizer] Failed to prepare {tf}: {e}")
                traceback.print_exc()
                # Record empty row for each RR so it's visible in results
                for rr in rr_ratios:
                    combo_num += 1
                    results.append(OptimizationRow(
                        timeframe=tf, rr_ratio=rr,
                        total_trades=0, wins=0, losses=0,
                        win_rate=0, buy_win_rate=0, sell_win_rate=0,
                        profit_factor=0, sharpe_ratio=0, expected_value=0,
                        max_drawdown_pct=0, total_return_pct=0,
                        final_capital=INITIAL_CAPITAL, avg_win_pips=0,
                        avg_loss_pips=0, composite_score=-999,
                    ))
                continue

        df = df_cache[tf]

        for rr in rr_ratios:
            combo_num += 1
            logger.info(f"[Optimizer] [{combo_num}/{total_combos}] "
                        f"TF={tf}  RR=1:{rr}")
            try:
                bt = run_backtest(
                    df,
                    strategy_name=f"Dio_{tf}_RR{rr}",
                    timeframe=tf,
                    rr=rr,
                    sl_atr_mult=sl_atr_mult,
                )
                row = OptimizationRow(
                    timeframe=tf,
                    rr_ratio=rr,
                    total_trades=bt.total_signals,
                    wins=bt.wins,
                    losses=bt.losses,
                    win_rate=bt.total_win_rate,
                    buy_win_rate=bt.buy_win_rate,
                    sell_win_rate=bt.sell_win_rate,
                    profit_factor=bt.profit_factor,
                    sharpe_ratio=bt.sharpe_ratio,
                    expected_value=bt.expected_value,
                    max_drawdown_pct=bt.max_drawdown_pct,
                    total_return_pct=bt.total_return_pct,
                    final_capital=bt.final_capital,
                    avg_win_pips=bt.avg_win_pips,
                    avg_loss_pips=bt.avg_loss_pips,
                )
                row.composite_score = (
                    _win_rate_score(row)
                    if score_mode == "win_rate"
                    else _composite_score(row)
                )
                results.append(row)
            except Exception as e:
                logger.error(f"[Optimizer] Backtest failed for {tf} RR={rr}: {e}")
                traceback.print_exc()
                results.append(OptimizationRow(
                    timeframe=tf, rr_ratio=rr,
                    total_trades=0, wins=0, losses=0,
                    win_rate=0, buy_win_rate=0, sell_win_rate=0,
                    profit_factor=0, sharpe_ratio=0, expected_value=0,
                    max_drawdown_pct=0, total_return_pct=0,
                    final_capital=INITIAL_CAPITAL, avg_win_pips=0,
                    avg_loss_pips=0, composite_score=-999,
                ))

    # Sort by composite score descending
    results.sort(key=lambda r: r.composite_score, reverse=True)

    if save_results:
        _save_results(results, pair=pair)

    _print_results_table(results)
    return results


# ─── Reporting helpers ────────────────────────────────────────────────────────

def _save_results(results: list[OptimizationRow], pair: str = PAIR_LABEL) -> None:
    """Save optimisation results to CSV and JSON."""
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)

    csv_path, json_path = _results_paths(pair)

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    logger.success(f"Results saved for {pair} to {csv_path}")
    logger.success(f"Results saved for {pair} to {json_path}")


def _print_results_table(results: list[OptimizationRow]) -> None:
    """Print a ranked summary table to the console."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Optimization Results — All Timeframe × R:R Combinations",
                      show_lines=True)

        table.add_column("Rank", justify="right", style="bold")
        table.add_column("Timeframe", style="cyan")
        table.add_column("R:R", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("PF", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("EV ($)", justify="right")
        table.add_column("Return%", justify="right")
        table.add_column("MaxDD%", justify="right")
        table.add_column("Score", justify="right", style="bold green")

        for rank, r in enumerate(results, 1):
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "∞"
            style = "bold green" if rank == 1 else ("green" if rank <= 3 else "")
            table.add_row(
                str(rank),
                r.timeframe,
                f"1:{r.rr_ratio}",
                str(r.total_trades),
                f"{r.win_rate:.1%}",
                pf_str,
                f"{r.sharpe_ratio:.2f}",
                f"{r.expected_value:.2f}",
                f"{r.total_return_pct:.1f}%",
                f"{r.max_drawdown_pct:.1f}%",
                f"{r.composite_score:.3f}",
                style=style,
            )

        console.print()
        console.print(table)

        # Best configuration summary
        best = results[0] if results else None
        if best and best.composite_score > -999:
            console.print()
            console.print(f"[bold green]🏆 BEST CONFIG: {best.timeframe} with "
                          f"1:{best.rr_ratio} R:R[/bold green]")
            console.print(f"   Win rate: {best.win_rate:.1%} | "
                          f"Profit factor: {pf_str} | "
                          f"Sharpe: {best.sharpe_ratio:.2f} | "
                          f"Return: {best.total_return_pct:.1f}% | "
                          f"Max DD: {best.max_drawdown_pct:.1f}%")
            console.print()
            console.print("[dim]⚠  Past results are hypothetical. They do not "
                          "guarantee future performance.[/dim]")
        console.print()

    except ImportError:
        # Fallback if rich isn't installed
        print("\n=== OPTIMIZATION RESULTS ===")
        for rank, r in enumerate(results, 1):
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
            print(f"#{rank:2d}  {r.timeframe:5s}  RR=1:{r.rr_ratio}  "
                  f"Trades={r.total_trades:3d}  WR={r.win_rate:.1%}  "
                  f"PF={pf_str}  Sharpe={r.sharpe_ratio:.2f}  "
                  f"EV=${r.expected_value:.2f}  Return={r.total_return_pct:.1f}%  "
                  f"MaxDD={r.max_drawdown_pct:.1f}%  Score={r.composite_score:.3f}")


def get_best_config(results: list[OptimizationRow]) -> dict:
    """Return the best (timeframe, rr_ratio) combination as a dict."""
    valid = [r for r in results if r.composite_score > -999 and r.total_trades >= 5]
    if not valid:
        logger.warning("No valid optimization results. Falling back to defaults.")
        return {"timeframe": "1h", "rr_ratio": 2.0, "sl_atr_mult": SL_ATR_MULTIPLIER}

    best = max(valid, key=lambda r: r.composite_score)
    return {
        "timeframe": best.timeframe,
        "rr_ratio": best.rr_ratio,
        "sl_atr_mult": SL_ATR_MULTIPLIER,
        "composite_score": best.composite_score,
        "win_rate": best.win_rate,
        "buy_win_rate": best.buy_win_rate,
        "sell_win_rate": best.sell_win_rate,
        "profit_factor": best.profit_factor,
        "sharpe_ratio": best.sharpe_ratio,
        "total_return_pct": best.total_return_pct,
        "total_trades": best.total_trades,
    }
