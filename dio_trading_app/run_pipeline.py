"""Run the full Dio Trading App pipeline and save results."""
import json
import sys

from backend.pipeline import run_full_pipeline

result = run_full_pipeline(timeframe="1h", run_training=True)

# Summarize
bt = result["backtest"]
summary = {
    "data_bars": len(result["df"]),
    "signals_generated": len(result["signals"]),
    "backtest": {
        "total_signals": bt.total_signals,
        "buy_signals": bt.buy_signals,
        "sell_signals": bt.sell_signals,
        "wins": bt.wins,
        "losses": bt.losses,
        "buy_win_rate": round(bt.buy_win_rate, 4),
        "sell_win_rate": round(bt.sell_win_rate, 4),
        "total_win_rate": round(bt.total_win_rate, 4),
        "profit_factor": round(bt.profit_factor, 3),
        "max_drawdown_pct": round(bt.max_drawdown_pct, 2),
        "sharpe_ratio": round(bt.sharpe_ratio, 4),
        "total_return_pct": round(bt.total_return_pct, 2),
        "initial_capital": bt.initial_capital,
        "final_capital": round(bt.final_capital, 2),
    },
    "signals": [
        {
            "direction": s.direction,
            "entry": s.entry_price,
            "sl": s.stop_loss,
            "tp": s.take_profit,
            "confidence": s.confidence_score,
            "win_prob": round(s.win_probability * 100, 1),
            "session": s.session,
        }
        for s in result["signals"]
    ],
}
print(json.dumps(summary, indent=2))
