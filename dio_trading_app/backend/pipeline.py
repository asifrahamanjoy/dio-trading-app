"""
Dio Trading App — Analysis Pipeline Orchestrator
==================================================
Single entry point that runs the complete analysis pipeline:
  Data → Preprocess → Indicators → Liquidity → Events →
  Features → Backtest → Train Model → Generate Signals

Use this for:
  - First-time setup
  - Full refresh of all analysis
  - Scheduled nightly runs
"""

from loguru import logger
from backend.modules.data_ingestion.downloader import fetch_ohlcv, load_multi_timeframe
from backend.modules.preprocessing.cleaner import preprocess
from backend.modules.technical.indicators import compute_all_indicators
from backend.modules.liquidity.analysis import compute_all_liquidity
from backend.modules.events.fundamental import load_event_calendar, flag_event_windows
from backend.modules.features.engineer import engineer_all_features
from backend.modules.backtesting.engine import run_backtest
from backend.modules.prediction.model import build_training_dataset, train_model, load_model
from backend.modules.signals.engine import scan_for_signals
from backend.modules.reporting.reporter import print_backtest_report, print_signal_report
from backend.modules.backtesting.optimizer import run_optimization, get_best_config


def run_full_pipeline(
    timeframe: str = "1h",
    force_refresh: bool = False,
    run_training: bool = True,
    rr: float = 2.0,
    optimize_first: bool = False,
) -> dict:
    """
    Execute the complete Dio Trading App pipeline.

    Returns a dict containing:
      - df: enriched DataFrame
      - backtest: BacktestResult
      - signals: list[TradingSignal]
      - model_bundle: trained model (if run_training=True)
    """
    logger.info("=" * 60)
    logger.info("DIO TRADING APP — FULL PIPELINE START")
    logger.info("=" * 60)

    # 0. Optional: run optimizer to find best TF + R:R before proceeding
    optimization_results = None
    if optimize_first:
        logger.info("Step 0: Running multi-TF / multi-RR optimization")
        optimization_results = run_optimization()
        best = get_best_config(optimization_results)
        timeframe = best["timeframe"]
        rr = best["rr_ratio"]
        logger.success(f"Optimizer selected: {timeframe} with 1:{rr} R:R")

    # 1. Data ingestion
    logger.info("Step 1: Data ingestion")
    df = fetch_ohlcv(interval=timeframe, force_refresh=force_refresh)

    # 2. Preprocessing
    logger.info("Step 2: Preprocessing")
    df = preprocess(df)

    # 3. Technical indicators
    logger.info("Step 3: Technical indicators")
    df = compute_all_indicators(df)

    # 4. Liquidity analysis
    logger.info("Step 4: Liquidity analysis")
    df = compute_all_liquidity(df)

    # 5. Event/fundamental analysis
    logger.info("Step 5: Event & fundamental analysis")
    events = load_event_calendar()
    df = flag_event_windows(df, events)

    # 6. Feature engineering
    logger.info("Step 6: Feature engineering")
    df = engineer_all_features(df)

    # 7. Backtesting
    logger.info("Step 7: Backtesting")
    backtest_result = run_backtest(df, timeframe=timeframe, rr=rr)
    print_backtest_report(backtest_result)

    win_rates = {
        "BUY": backtest_result.buy_win_rate,
        "SELL": backtest_result.sell_win_rate,
        "BUY_count": backtest_result.buy_signals,
        "SELL_count": backtest_result.sell_signals,
    }

    # 8. Model training
    model_bundle = None
    if run_training:
        logger.info("Step 8: Training prediction model")
        try:
            X, y = build_training_dataset(df)
            train_result = train_model(X, y, save=True)
            model_bundle = train_result["model_bundle"]
            logger.success(
                f"Model trained — CV AUC: {train_result['cv_mean_auc']:.4f} | "
                f"Test AUC: {train_result['test_auc']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Model training failed: {e}. Using pre-trained model if available.")
            model_bundle = load_model()
    else:
        model_bundle = load_model()

    # 9. Signal generation
    logger.info("Step 9: Scanning for signals")
    signals = scan_for_signals(
        df,
        backtest_win_rates=win_rates,
        model_bundle=model_bundle,
        timeframe=timeframe,
        last_n_bars=5,
    )
    print_signal_report(signals)

    logger.info("=" * 60)
    logger.success(f"PIPELINE COMPLETE — {len(signals)} signal(s) generated")
    logger.info("=" * 60)

    return {
        "df": df,
        "backtest": backtest_result,
        "signals": signals,
        "model_bundle": model_bundle,
        "events": events,
        "optimization": optimization_results,
        "config": {"timeframe": timeframe, "rr": rr},
    }


if __name__ == "__main__":
    run_full_pipeline(timeframe="1h", run_training=True)
