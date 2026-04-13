"""
Dio Trading App — Optimizer Runner
====================================
Runs the multi-timeframe, multi-R:R optimization and prints
which combination yields the best backtest results.

Usage:
    cd "d:\AI Model\dio trading app\dio_trading_app"
    python run_optimizer.py
"""

import sys
import json
from loguru import logger

from backend.modules.backtesting.optimizer import run_optimization, get_best_config


def main():
    logger.info("=" * 60)
    logger.info("DIO TRADING APP — MULTI-TF / MULTI-RR OPTIMIZER")
    logger.info("=" * 60)

    results = run_optimization()

    best = get_best_config(results)

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"Best config: {json.dumps(best, indent=2, default=str)}")
    logger.info("=" * 60)

    return best


if __name__ == "__main__":
    best = main()
