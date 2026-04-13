# Dio Trading App

**EUR/USD market analysis, signal generation, backtesting, and probability-based prediction platform.**

> ⚠ **DISCLAIMER**: This software is for educational and research purposes only. All signals are probabilistic estimates. Nothing in this software constitutes financial advice. Trading forex involves substantial risk of loss. Past performance does not guarantee future results. Volume data is proxy/tick only — EUR/USD spot forex has no centralised true volume.

---

## Overview

Dio Trading App is a professional, modular trading analysis platform built in Python. It studies 5 years of EUR/USD data and generates buy/sell signals only after combining:

- Technical indicator confluence (RSI, MACD, EMA, ATR, Bollinger Bands, momentum)
- Volume-price condition analysis (4 conditions, with proxy volume caveat)
- Liquidity & session analysis (London, NY, Tokyo, overlaps, stop hunts)
- Fundamental & event risk (ECB, Fed, CPI, NFP, GDP)
- News sentiment analysis
- Historical backtesting statistics
- ML model probability (XGBoost + Logistic Regression ensemble)

---

## Architecture

```
dio_trading_app/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI REST API (all endpoints)
│   ├── core/
│   │   ├── config.py            # All settings, thresholds, constants
│   │   └── database.py          # SQLAlchemy models + session management
│   ├── modules/
│   │   ├── data_ingestion/
│   │   │   └── downloader.py    # yfinance EUR/USD + futures volume proxy
│   │   ├── preprocessing/
│   │   │   └── cleaner.py       # OHLCV cleaning, gap fill, outlier removal
│   │   ├── features/
│   │   │   └── engineer.py      # Feature engineering (confluence, regime)
│   │   ├── technical/
│   │   │   └── indicators.py    # RSI, MACD, EMA, ATR, BB, S/R, trend
│   │   ├── liquidity/
│   │   │   └── analysis.py      # Sessions, sweeps, liquidity zones
│   │   ├── events/
│   │   │   └── fundamental.py   # Economic events, news sentiment
│   │   ├── backtesting/
│   │   │   └── engine.py        # Trade simulator + performance metrics
│   │   ├── prediction/
│   │   │   └── model.py         # XGBoost + LR ensemble, TimeSeriesCV
│   │   ├── signals/
│   │   │   └── engine.py        # Multi-gate signal generator
│   │   ├── alerts/
│   │   │   └── scheduler.py     # APScheduler + email/log dispatch
│   │   └── reporting/
│   │       └── reporter.py      # Rich console + CSV/JSON export
│   └── pipeline.py              # Full pipeline orchestrator
├── frontend/
│   └── app.py                   # Streamlit multi-page dashboard
├── data/
│   ├── raw/                     # Original downloaded data
│   ├── processed/               # Cleaned + feature-enriched data
│   └── cache/                   # Parquet cache files
├── models/                      # Saved ML model artifacts
├── logs/                        # Backtest reports, alert logs
├── tests/
│   └── test_suite.py            # Pytest unit + integration tests
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
cd dio_trading_app
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env to set email alerts, API keys, etc.
```

### 3. Start the backend API

```bash
make backend
# or:
uvicorn backend.api.main:app --reload --port 8000
```

### 4. Start the dashboard

```bash
make frontend
# or:
streamlit run frontend/app.py
```

### 5. Run the full pipeline from CLI

```bash
python -m backend.pipeline
```

### 6. Run backtest from CLI

```bash
make backtest
```

### 7. Train ML model from CLI

```bash
make train
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | App status and model info |
| GET | `/health` | Health check |
| GET | `/data/summary` | EUR/USD data summary |
| GET | `/data/ohlcv` | Recent OHLCV bars |
| GET | `/data/volume-conditions` | 4-condition volume stats |
| GET | `/analysis/full` | Full technical + liquidity analysis |
| GET | `/analysis/session-stats` | Per-session statistics |
| POST | `/backtest/run` | Run full backtest |
| GET | `/backtest/trades` | Detailed trade log |
| POST | `/model/train` | Train ML prediction model |
| GET | `/model/predict` | Current direction probability |
| GET | `/signals/latest` | Scan for active signals |
| GET | `/dashboard/summary` | Dashboard summary stats |

Interactive docs at: `http://localhost:8000/docs`

---

## Signal Quality Gates

A signal is only generated when ALL of the following pass:

| Gate | Threshold |
|------|-----------|
| Composite confidence score | ≥ 60 / 100 |
| Historical win rate (backtest) | ≥ 52% |
| Profit factor | ≥ 1.3 |
| Not in high-impact event window | Required |
| Adequate liquidity (not off-hours) | Required |
| Volatility not extreme | Unless confidence ≥ 75 |

---

## Signal Output Format

Every signal includes:

```json
{
  "direction": "BUY",
  "entry_price": 1.08500,
  "stop_loss": 1.08380,
  "take_profit": 1.08740,
  "risk_reward": 2.0,
  "confidence_score": 72.5,
  "win_probability": 0.587,
  "historical_win_rate": 0.561,
  "model_probability": 0.623,
  "setup_frequency": 148,
  "session": "london",
  "liquidity_zone": "high",
  "signal_reason": "Multi-condition BUY signal...",
  "invalidation_reason": "Trade invalidated if price closes below...",
  "contributing_factors": [...],
  "volume_disclaimer": "EUR/USD spot forex has NO centralised volume...",
  "risk_warning": "This signal is probabilistic and NOT financial advice..."
}
```

---

## Volume Disclaimer (Important)

EUR/USD spot forex does **not** have centralised, reliable volume data. This application uses:

1. **Tick volume** from Yahoo Finance (`EURUSD=X`) — counts price ticks per bar, not actual traded lots
2. **CME futures volume** (`6E=F`) — daily volume from EUR/USD futures as an institutional proxy

All volume-based analysis is therefore approximate. Volume conditions (price_up_vol_up, etc.) are directional indicators, not absolute confirmations. This disclaimer is embedded in all signal outputs and the codebase.

---

## Backtesting Metrics Explained

| Metric | Formula | What it means |
|--------|---------|---------------|
| Win Rate | wins / total trades | % of trades that hit TP |
| Profit Factor | gross profit / gross loss | > 1.0 = net profitable |
| Expected Value | (win_rate × avg_win) − (loss_rate × avg_loss) | Average $ per trade |
| Max Drawdown | peak-to-trough equity drop | Largest losing streak impact |
| Sharpe Ratio | mean PnL / std PnL × √(bars/yr) | Risk-adjusted return |

---

## Volume-Price Market Conditions

The system classifies every bar into one of 4 conditions and tracks next-bar direction rates:

| Condition | Interpretation |
|-----------|---------------|
| Price Up + Volume Up | Bullish confirmation — strong buying |
| Price Down + Volume Down | Weak sell — potential base building |
| Price Down + Volume Up | Bearish confirmation — strong selling |
| Price Up + Volume Down | Weak buy — potential exhaustion |

---

## Trading Sessions (UTC)

| Session | Hours (UTC) | EUR/USD Liquidity |
|---------|-------------|-------------------|
| Tokyo | 00:00 – 09:00 | Low |
| London | 07:00 – 16:00 | High |
| New York | 12:00 – 21:00 | High |
| London/NY Overlap | 12:00 – 16:00 | Highest (preferred) |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Development Roadmap

- [x] Data ingestion (yfinance, futures proxy)
- [x] Technical indicators (RSI, MACD, EMA, ATR, BB, S/R, trend)
- [x] Liquidity analysis (sessions, sweeps, zones)
- [x] Event & news analysis
- [x] Backtesting engine with full metrics
- [x] ML prediction model (XGBoost + LR ensemble)
- [x] Multi-gate signal engine with confidence scoring
- [x] FastAPI backend
- [x] Streamlit dashboard
- [x] Alert system (log + email)
- [x] Test suite
- [ ] WebSocket live price feed
- [ ] PostgreSQL migration (currently SQLite)
- [ ] Telegram/Discord alert channel
- [ ] Multi-pair support (GBP/USD, USD/JPY)
- [ ] Docker deployment

---

## Configuration

Key thresholds in `backend/core/config.py`:

```python
MIN_CONFIDENCE_SCORE = 60.0     # Minimum signal confidence (0–100)
MIN_WIN_RATE_THRESHOLD = 0.52   # Historical win rate floor
RISK_REWARD_RATIO = 2.0         # Always 1:2
SL_ATR_MULTIPLIER = 1.5         # SL = entry ± ATR × 1.5
TP_ATR_MULTIPLIER = 3.0         # TP = entry ± ATR × 3.0
ALERT_CHECK_INTERVAL_SECONDS = 300  # 5 minutes
```

---

## License

MIT License. For educational purposes. Not for live trading without independent validation.
