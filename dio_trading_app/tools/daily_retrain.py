from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from loguru import logger

from backend.api.main import app
from backend.core.config import SUPPORTED_PAIRS


TRAIN_TIMEFRAMES = ["1d", "1h"]
BACKTEST_TIMEFRAME = "1d"


def _train_pair(client: TestClient, pair: str, timeframe: str) -> None:
    payload = {
        "pair": pair,
        "timeframe": timeframe,
        "force_refresh": False,
    }
    response = client.post("/model/train", json=payload, timeout=900)
    response.raise_for_status()
    data = response.json()
    logger.info(
        f"TRAIN {pair} {timeframe}: test_auc={data.get('test_auc')} cv_auc={data.get('cv_mean_auc')} samples={data.get('samples')}"
    )


def _refresh_backtest(client: TestClient, pair: str) -> None:
    payload = {
        "pair": pair,
        "timeframe": BACKTEST_TIMEFRAME,
        "strategy_name": "DailyRetrainRefresh",
        "rr": 1.0,
        "force_refresh": False,
        "recent_signals": 0,
    }
    response = client.post("/backtest/run", json=payload, timeout=900)
    response.raise_for_status()
    data = response.json()
    logger.info(
        f"BACKTEST {pair}: WR={round(data.get('total_win_rate', 0) * 100, 1)}% PF={data.get('profit_factor')} signals={data.get('total_signals')}"
    )


def main() -> None:
    start_time = datetime.utcnow().isoformat()
    logger.info(f"Daily retrain started at {start_time} UTC")

    with TestClient(app) as client:
        for pair in SUPPORTED_PAIRS:
            logger.info(f"=== {pair} ===")
            for timeframe in TRAIN_TIMEFRAMES:
                try:
                    _train_pair(client, pair, timeframe)
                except Exception as exc:
                    logger.warning(f"Train failed for {pair} {timeframe}: {exc}")
            try:
                _refresh_backtest(client, pair)
            except Exception as exc:
                logger.warning(f"Backtest refresh failed for {pair}: {exc}")

    logger.info("Daily retrain completed")


if __name__ == "__main__":
    main()
