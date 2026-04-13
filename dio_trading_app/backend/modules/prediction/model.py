"""
Dio Trading App — Prediction Engine
======================================
Builds and applies machine learning models to estimate the
probability of EUR/USD price direction on the next N bars.

Models:
  1. XGBoost classifier (primary)
  2. Logistic Regression (calibration baseline)
  3. Ensemble probability averaging

Output is a probability score [0, 1] where:
  > 0.55 = mild bullish lean
  > 0.65 = strong bullish lean
  < 0.45 = mild bearish lean
  < 0.35 = strong bearish lean

CRITICAL DISCLAIMER:
  These are PROBABILISTIC estimates, NOT predictions.
  No model can reliably predict forex price direction.
  Probabilities above 60% are considered 'moderate confidence'.
  Never treat any output as financial advice or guaranteed profit.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from backend.core.config import (
    ML_TEST_SIZE, ML_RANDOM_STATE, ML_N_ESTIMATORS,
    FEATURE_IMPORTANCE_TOP_N, MODELS_DIR, normalize_pair
)

warnings.filterwarnings("ignore")


def _pair_model_path(pair: str) -> Path:
    safe_pair = normalize_pair(pair).replace("/", "_").replace(" ", "")
    return MODELS_DIR / f"direction_model_{safe_pair}.pkl"


# ─── Feature Selection ────────────────────────────────────────────────────────

FEATURE_COLS = [
    # RSI
    "rsi_14",
    # MACD
    "macd_line", "macd_signal", "macd_histogram",
    # MAs
    "ema_9", "ema_21", "ema_50",
    # Positioning
    "above_ema9", "above_ema21", "above_ema50", "above_sma200",
    "ema_bull_align", "ema_bear_align",
    # ATR / Volatility
    "atr", "atr_pct", "volatility_20",
    # Bollinger
    "bb_width", "bb_pct_b", "bb_squeeze",
    # Momentum
    "momentum_pct", "roc",
    # Volume (proxy)
    "volume_ratio", "high_volume", "low_volume",
    # Breakout/Reversal
    "breakout_up", "breakout_down",
    "bullish_engulf", "bearish_engulf",
    "hammer", "shooting_star",
    # Trend structure
    "uptrend_structure", "downtrend_structure",
    # Liquidity
    "liquidity_score",
    # Stop hunts
    "bullish_sweep", "bearish_sweep",
    # Session (encoded)
    "session_london", "session_ny", "session_overlap", "session_tokyo",
    # Event
    "event_window", "event_sentiment",
    # HL range
    "hl_range", "body",
    # Returns
    "returns", "log_returns",
    # Multi-bar context & momentum (new)
    "return_3bar", "return_5bar", "return_10bar",
    "rsi_roc_3", "rsi_roc_5",
    "atr_ratio",
    "dist_ema_9_atr", "dist_ema_21_atr", "dist_ema_50_atr",
    "bull_streak_3", "bear_streak_3",
    "body_atr_ratio",
    "trend_strength",
    # Confluence scores
    "buy_confluence", "sell_confluence",
]

TARGET_HORIZON = 4  # Predict direction N bars ahead
TARGET_HORIZON_BY_TIMEFRAME = {
    "15m": 16,  # ~4 hours ahead
    "1h": 8,    # ~8 hours ahead
    "4h": 6,    # ~1 day ahead
    "1d": 3,    # ~3 days ahead
}


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_training_dataset(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    timeframe: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construct features (X) and binary target (y) for model training.

    Target: 1 if close[t+horizon] > close[t], else 0
    Features: all available technical, liquidity, and event columns.

    Uses only columns present in df to avoid errors on partial data.
    """
    df = df.copy()

    if timeframe:
        horizon = TARGET_HORIZON_BY_TIMEFRAME.get(timeframe, horizon)

    # Build target
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)

    # Select available features
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.debug(f"Missing {len(missing)} feature columns (will be skipped): {missing[:5]}...")

    df_feat = df[available].copy()

    # Convert boolean columns to int
    bool_cols = df_feat.select_dtypes(include=bool).columns
    df_feat[bool_cols] = df_feat[bool_cols].astype(int)

    # Convert categorical columns
    cat_cols = df_feat.select_dtypes(include="category").columns
    for col in cat_cols:
        df_feat[col] = df_feat[col].cat.codes

    # Drop rows with NaN — use a threshold to keep rows where most features exist
    df_feat["__target__"] = df["target"]
    # First drop any columns that are >50% NaN (e.g. unavailable volume data)
    nan_pct = df_feat.isna().mean()
    bad_cols = nan_pct[nan_pct > 0.5].index.tolist()
    if bad_cols:
        logger.warning(f"Dropping {len(bad_cols)} columns with >50% NaN: {bad_cols}")
        df_feat = df_feat.drop(columns=bad_cols)
    df_feat = df_feat.dropna()

    y = df_feat.pop("__target__")
    X = df_feat

    logger.info(f"Dataset built: {len(X)} samples × {len(X.columns)} features | horizon={horizon}")
    return X, y


# ─── Model Training ───────────────────────────────────────────────────────────

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    pair: str = "EUR/USD",
    timeframe: str = "1d",
    save: bool = True,
) -> dict:
    """
    Train a time-series-aware ensemble and return calibrated probabilities.

    Uses TimeSeriesSplit to avoid lookahead bias in cross-validation.
    Final model is calibrated with Platt scaling for reliable probabilities.

    Returns:
        {
          "model": trained pipeline,
          "features": list of feature names,
          "cv_scores": list of AUC per fold,
          "test_report": classification report dict,
          "feature_importance": top-N feature importances,
        }
    """
    n_samples = len(X)
    test_size = max(int(n_samples * ML_TEST_SIZE), 100)
    train_size = n_samples - test_size

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    logger.info(f"Training: {train_size} samples | Test: {test_size} samples")

    # Dynamic class weighting to avoid majority-class bias.
    class_counts = y_train.value_counts().to_dict()
    pos_count = float(class_counts.get(1, 1.0))
    neg_count = float(class_counts.get(0, 1.0))
    pos_weight = neg_count / max(pos_count, 1.0)
    sample_weight = np.where(y_train.values == 1, pos_weight, 1.0)

    # ── Gradient Boosting (primary) ───────────────────────────────
    gb = GradientBoostingClassifier(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.75,
        min_samples_leaf=30,
        max_features=0.7,
        random_state=ML_RANDOM_STATE,
    )

    # ── Random Forest (diversity in ensemble) ─────────────────────
    rf = RandomForestClassifier(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=5,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=ML_RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    # ── Logistic Regression (calibration baseline) ────────────────
    lr = LogisticRegression(
        C=0.5, max_iter=1000, random_state=ML_RANDOM_STATE, class_weight="balanced"
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    gb.fit(X_train_sc, y_train, sample_weight=sample_weight)
    rf.fit(X_train_sc, y_train)
    lr.fit(X_train_sc, y_train, sample_weight=sample_weight)

    # Calibrate probabilities (sigmoid calibration)
    # scikit-learn 1.6+ removed cv="prefit"; use a 3-fold calibration instead
    gb_cal = CalibratedClassifierCV(gb, method="sigmoid", cv=3)
    gb_cal.fit(X_train_sc, y_train)

    rf_cal = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
    rf_cal.fit(X_train_sc, y_train)

    # Ensemble prediction is done via module-level function (picklable)

    # ── CV Evaluation ─────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        gb_cv = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=ML_RANDOM_STATE
        )
        gb_cv.fit(X_tr_s, y_tr)
        probs = gb_cv.predict_proba(X_te_s)[:, 1]
        auc = roc_auc_score(y_te, probs)
        cv_aucs.append(round(auc, 4))
        logger.debug(f"Fold {fold+1} AUC: {auc:.4f}")

    # ── Test Set Evaluation ───────────────────────────────────────
    test_bundle = {"gb_cal": gb_cal, "rf_cal": rf_cal, "lr": lr}
    test_proba = ensemble_predict_proba(test_bundle, X_test_sc)
    test_pred = (test_proba >= 0.5).astype(int)
    try:
        test_auc = roc_auc_score(y_test, test_proba)
    except Exception:
        test_auc = 0.0

    report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    logger.info(f"Test AUC: {test_auc:.4f} | Accuracy: {report.get('accuracy', 0):.4f}")
    logger.info(f"CV AUC scores: {cv_aucs} | Mean: {np.mean(cv_aucs):.4f}")

    # ── Feature Importance ────────────────────────────────────────
    importances = pd.Series(gb.feature_importances_, index=X.columns)
    top_features = importances.nlargest(FEATURE_IMPORTANCE_TOP_N).round(4).to_dict()

    # ── Save model artifacts ──────────────────────────────────────
    bundle = {
        "gb": gb,
        "gb_cal": gb_cal,
        "rf": rf,
        "rf_cal": rf_cal,
        "lr": lr,
        "scaler": scaler,
        "features": list(X.columns),
        "pair": normalize_pair(pair),
        "timeframe": timeframe,
    }
    if save:
        model_path = _pair_model_path(pair)
        joblib.dump(bundle, model_path)
        logger.success(f"Model saved: {model_path}")

    return {
        "model_bundle": bundle,
        "features": list(X.columns),
        "cv_auc_scores": cv_aucs,
        "cv_mean_auc": round(float(np.mean(cv_aucs)), 4),
        "test_auc": round(test_auc, 4),
        "test_report": report,
        "feature_importance": top_features,
    }


# ─── Ensemble Prediction (module-level for pickling) ──────────────────────────

def ensemble_predict_proba(bundle: dict, X_arr) -> np.ndarray:
    """Average GB calibrated + RF calibrated + LR probabilities (0.5 / 0.3 / 0.2 weight)."""
    p_gb = bundle["gb_cal"].predict_proba(X_arr)[:, 1]
    p_lr = bundle["lr"].predict_proba(X_arr)[:, 1]
    if "rf_cal" in bundle:
        p_rf = bundle["rf_cal"].predict_proba(X_arr)[:, 1]
        return 0.5 * p_gb + 0.3 * p_rf + 0.2 * p_lr
    return 0.7 * p_gb + 0.3 * p_lr


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model(pair: str = "EUR/USD") -> Optional[dict]:
    model_path = _pair_model_path(pair)
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            logger.info(f"Loaded trained model from disk for {normalize_pair(pair)}.")
            return bundle
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return None


def predict_probability(
    df_row: pd.Series,
    model_bundle: dict,
) -> dict:
    """
    Given a single DataFrame row (latest bar), return direction probability.

    Returns:
        {
          "bull_probability": float,   # P(next N bars close higher)
          "bear_probability": float,
          "direction_lean": "BULLISH" | "BEARISH" | "NEUTRAL",
          "confidence_label": "low" | "moderate" | "high",
          "disclaimer": str,
        }
    """
    features = model_bundle["features"]
    scaler = model_bundle["scaler"]

    # Build input vector
    row_dict = {}
    for f in features:
        val = df_row.get(f, 0)
        if pd.isna(val):
            val = 0
        if isinstance(val, bool):
            val = int(val)
        row_dict[f] = val

    X_input = pd.DataFrame([row_dict])[features]

    # Convert categoricals
    for col in X_input.select_dtypes(include="category").columns:
        X_input[col] = X_input[col].cat.codes

    X_scaled = scaler.transform(X_input)
    bull_prob = float(ensemble_predict_proba(model_bundle, X_scaled)[0])
    bear_prob = 1.0 - bull_prob

    if bull_prob > 0.60:
        lean = "BULLISH"
        conf = "high" if bull_prob > 0.70 else "moderate"
    elif bull_prob < 0.40:
        lean = "BEARISH"
        conf = "high" if bull_prob < 0.30 else "moderate"
    else:
        lean = "NEUTRAL"
        conf = "low"

    return {
        "bull_probability": round(bull_prob, 4),
        "bear_probability": round(bear_prob, 4),
        "direction_lean": lean,
        "confidence_label": conf,
        "disclaimer": (
            "This probability is a statistical estimate only. "
            "It does NOT constitute a trade signal or financial advice. "
            "Forex markets are inherently unpredictable."
        ),
    }
