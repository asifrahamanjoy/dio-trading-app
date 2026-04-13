# Dio Trading App

Dio Trading App is a professional Python scaffold for EUR/USD market analysis, signal generation, alerting, and probability-based prediction.

## Scope

This project is designed to grow into a data-driven trading platform that combines:

- technical analysis
- proxy volume analysis for forex
- liquidity and session analysis
- event and macro analysis
- news sentiment analysis
- backtesting and performance reporting
- probability-based signal generation
- FastAPI backend and Streamlit dashboard

## Important market limitation

Spot forex does not provide centralized true volume. For EUR/USD, this project uses proxy volume such as tick volume or futures volume and keeps that limitation explicit in the code and documentation.

## Initial scaffold

The current scaffold includes:

- FastAPI application entrypoint
- Streamlit dashboard entrypoint
- modular packages for ingestion, analytics, backtesting, prediction, signals, alerts, and reporting
- placeholder schemas and routes
- documentation and test skeletons

## Planned development order

1. Build data ingestion for EUR/USD OHLC and proxy volume
2. Build technical, liquidity, and event feature pipelines
3. Build backtesting and performance metrics
4. Build prediction and signal probability engine
5. Build alerting and dashboard workflows

## Getting started

```powershell
cd "d:\AI Model\dio trading app"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
uvicorn dio_trading_app.main:app --reload
streamlit run src/dio_trading_app/dashboard.py
```

## Project layout

```text
src/dio_trading_app/
  api/
  analytics/
  data_ingestion/
  backtesting/
  prediction/
  signals/
  alerts/
  reporting/
  schemas/
docs/
tests/
data/
reports/
notebooks/
```

## Current behavior

The scaffold does not emit live trading advice yet. Placeholder endpoints and dashboards deliberately avoid fake certainty until the real data, features, backtests, and probability models are implemented.
