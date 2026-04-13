"""
Dio Trading App — Reporting Module
=====================================
Generates structured performance reports from backtest results
and signal history. Outputs to console (rich), CSV, and JSON.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from loguru import logger

from backend.core.config import LOGS_DIR, VOLUME_DISCLAIMER


console = Console(force_terminal=False, legacy_windows=False)


def print_backtest_report(result) -> None:
    """Pretty-print a BacktestResult to the console using rich."""

    console.print(Panel.fit(
        f"[bold cyan]DIO TRADING APP — BACKTEST REPORT[/bold cyan]\n"
        f"Strategy: {result.strategy_name} | Timeframe: {result.timeframe}\n"
        f"{result.start_date[:10]} → {result.end_date[:10]}",
        border_style="cyan"
    ))

    # Performance table
    table = Table(title="Performance Metrics", border_style="dim")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    win_color = "green" if result.total_win_rate >= 0.55 else "yellow" if result.total_win_rate >= 0.50 else "red"
    pf_color = "green" if result.profit_factor >= 1.5 else "yellow" if result.profit_factor >= 1.2 else "red"

    rows = [
        ("Total Signals",       str(result.total_signals)),
        ("Buy Signals",         str(result.buy_signals)),
        ("Sell Signals",        str(result.sell_signals)),
        ("Wins",                f"[green]{result.wins}[/green]"),
        ("Losses",              f"[red]{result.losses}[/red]"),
        ("Buy Win Rate",        f"[{win_color}]{result.buy_win_rate:.1%}[/{win_color}]"),
        ("Sell Win Rate",       f"[{win_color}]{result.sell_win_rate:.1%}[/{win_color}]"),
        ("Total Win Rate",      f"[{win_color}]{result.total_win_rate:.1%}[/{win_color}]"),
        ("Profit Factor",       f"[{pf_color}]{result.profit_factor:.3f}[/{pf_color}]"),
        ("Expected Value",      f"${result.expected_value:.2f}"),
        ("Max Drawdown",        f"[red]{result.max_drawdown_pct:.1f}%[/red]"),
        ("Sharpe Ratio",        f"{result.sharpe_ratio:.3f}"),
        ("Initial Capital",     f"${result.initial_capital:,.2f}"),
        ("Final Capital",       f"${result.final_capital:,.2f}"),
        ("Total Return",        f"{'[green]' if result.total_return_pct > 0 else '[red]'}{result.total_return_pct:.1f}%{'[/green]' if result.total_return_pct > 0 else '[/red]'}"),
        ("Avg Win (pips)",      f"{result.avg_win_pips:.1f}"),
        ("Avg Loss (pips)",     f"{result.avg_loss_pips:.1f}"),
    ]
    for metric, val in rows:
        table.add_row(metric, val)

    console.print(table)
    console.print(f"\n[yellow dim]⚠ {VOLUME_DISCLAIMER}[/yellow dim]")
    console.print("[yellow dim]⚠ Backtest results are hypothetical. Past performance does not guarantee future results.[/yellow dim]\n")


def export_backtest_report(result, output_dir: Path = LOGS_DIR) -> dict[str, Path]:
    """Export backtest result to JSON and CSV files."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"backtest_{result.strategy_name}_{ts}"

    # JSON summary
    json_path = base.parent / (base.name + "_summary.json")
    summary = {k: v for k, v in result.__dict__.items() if k not in ("trades", "equity_curve")}
    summary["volume_disclaimer"] = VOLUME_DISCLAIMER
    summary["report_disclaimer"] = "Hypothetical results only."
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Trade log CSV
    csv_path = base.parent / (base.name + "_trades.csv")
    if result.trades:
        df_trades = pd.DataFrame(result.trades)
        df_trades.to_csv(csv_path, index=False)

    # Equity curve CSV
    eq_path = base.parent / (base.name + "_equity.csv")
    pd.Series(result.equity_curve, name="equity").to_csv(eq_path, index=True)

    logger.success(f"Backtest report exported to {output_dir}")
    return {"summary": json_path, "trades": csv_path, "equity": eq_path}


def print_signal_report(signals: list) -> None:
    """Pretty-print a list of TradingSignal objects."""
    if not signals:
        console.print("[yellow]No signals to display.[/yellow]")
        return

    for sig in signals:
        direction_color = "green" if sig.direction == "BUY" else "red"
        console.print(Panel(
            f"[bold {direction_color}]{sig.direction}[/bold {direction_color}] "
            f"@ {sig.entry_price:.5f}  |  Confidence: {sig.confidence_score:.1f}/100  |  "
            f"Win Prob: {sig.win_probability * 100:.1f}%\n"
            f"SL: {sig.stop_loss:.5f}  TP: {sig.take_profit:.5f}  R:R 1:{sig.risk_reward:.0f}\n"
            f"Session: {sig.session}  |  Liquidity: {sig.liquidity_zone}\n\n"
            f"[dim]WHY: {sig.signal_reason[:120]}...[/dim]\n"
            f"[yellow dim]INVALIDATION: {sig.invalidation_reason[:100]}...[/yellow dim]",
            title=f"EUR/USD Signal — {sig.generated_at[:16]}",
            border_style=direction_color,
        ))

    console.print(f"\n[yellow dim]⚠ Not financial advice. Probabilistic signals only.[/yellow dim]")
    console.print(f"[yellow dim]⚠ {VOLUME_DISCLAIMER}[/yellow dim]\n")
