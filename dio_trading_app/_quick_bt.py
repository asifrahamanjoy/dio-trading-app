import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from backend.modules.data_ingestion.downloader import fetch_ohlcv
from backend.modules.preprocessing.cleaner import preprocess
from backend.modules.technical.indicators import compute_all_indicators
from backend.modules.liquidity.analysis import compute_all_liquidity
from backend.modules.events.fundamental import load_event_calendar, flag_event_windows
from backend.modules.features.engineer import engineer_all_features
from backend.modules.backtesting.engine import run_backtest
import backend.core.config as cfg

df = fetch_ohlcv()
df = preprocess(df)
df = compute_all_indicators(df)
df = compute_all_liquidity(df)
events = load_event_calendar()
df = flag_event_windows(df, events)
df = engineer_all_features(df)

# Test multiple SL/TP configs
configs = [
    (1.5, 3.0, "1.5/3.0 (current)"),
    (2.0, 4.0, "2.0/4.0 (wider)"),
    (1.2, 2.4, "1.2/2.4 (tighter)"),
    (1.5, 4.5, "1.5/4.5 (1:3 R:R)"),
]
dollar = "$"
for sl_m, tp_m, label in configs:
    cfg.SL_ATR_MULTIPLIER = sl_m
    cfg.TP_ATR_MULTIPLIER = tp_m
    r = run_backtest(df)
    print(f"\n=== {label} ===")
    print(f"Signals: {r.total_signals} | WR: {r.total_win_rate:.1%} | PF: {r.profit_factor:.3f}")
    print(f"Return: {r.total_return_pct:.1f}% | Final: {dollar}{r.final_capital:.2f} | EV: {dollar}{r.expected_value:.2f}")
    print(f"Avg Win: {r.avg_win_pips:.1f} pips | Avg Loss: {r.avg_loss_pips:.1f} pips | DD: {r.max_drawdown_pct:.1f}%")
