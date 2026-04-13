# Dio Trading App — Makefile

.PHONY: install backend frontend backtest train help

help:
	@echo ""
	@echo "Dio Trading App — Commands"
	@echo "--------------------------"
	@echo "make install    Install all dependencies"
	@echo "make backend    Start FastAPI backend (port 8000)"
	@echo "make frontend   Start Streamlit dashboard (port 8501)"
	@echo "make backtest   Run backtest from CLI"
	@echo "make train      Train prediction model from CLI"
	@echo ""

install:
	pip install -r requirements.txt

backend:
	uvicorn backend.api.main:app --reload --port 8000 --log-level info

frontend:
	streamlit run frontend/app.py --server.port 8501

backtest:
	python -c "\
from backend.modules.data_ingestion.downloader import fetch_ohlcv; \
from backend.modules.technical.indicators import compute_all_indicators; \
from backend.modules.liquidity.analysis import compute_all_liquidity; \
from backend.modules.events.fundamental import load_event_calendar, flag_event_windows; \
from backend.modules.backtesting.engine import run_backtest; \
from rich.console import Console; \
from rich.table import Table; \
import json; \
c = Console(); \
c.print('[bold cyan]Running backtest...[/bold cyan]'); \
df = fetch_ohlcv(); \
df = compute_all_indicators(df); \
df = compute_all_liquidity(df); \
ev = load_event_calendar(); \
df = flag_event_windows(df, ev); \
r = run_backtest(df); \
c.print(f'[green]Win Rate: {r.total_win_rate:.1%}[/green]'); \
c.print(f'Profit Factor: {r.profit_factor:.2f}'); \
c.print(f'Max Drawdown: {r.max_drawdown_pct:.1f}%'); \
c.print(f'Sharpe: {r.sharpe_ratio:.3f}'); \
c.print(f'Total Return: {r.total_return_pct:.1f}%'); \
"

train:
	python -c "\
from backend.modules.data_ingestion.downloader import fetch_ohlcv; \
from backend.modules.technical.indicators import compute_all_indicators; \
from backend.modules.liquidity.analysis import compute_all_liquidity; \
from backend.modules.events.fundamental import load_event_calendar, flag_event_windows; \
from backend.modules.prediction.model import build_training_dataset, train_model; \
from rich.console import Console; \
c = Console(); \
c.print('[bold cyan]Training model...[/bold cyan]'); \
df = fetch_ohlcv(); \
df = compute_all_indicators(df); \
df = compute_all_liquidity(df); \
ev = load_event_calendar(); \
df = flag_event_windows(df, ev); \
X, y = build_training_dataset(df); \
r = train_model(X, y); \
c.print(f'[green]CV AUC: {r[\"cv_mean_auc\"]:.4f}[/green]'); \
c.print(f'Test AUC: {r[\"test_auc\"]:.4f}'); \
"
