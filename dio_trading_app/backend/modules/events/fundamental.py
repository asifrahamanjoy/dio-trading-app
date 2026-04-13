"""
Dio Trading App — Event & News Analysis Module
================================================
Handles:
  1. Scheduled economic events (ECB, Fed, CPI, NFP, GDP, etc.)
  2. Event impact classification
  3. Simple rule-based news sentiment scoring for EUR/USD direction
  4. Event-window risk flagging (avoid signals near high-impact events)

Data sources:
  - Hardcoded known event types with expected EUR/USD directional bias
  - NewsAPI (optional, requires API key) for real-time sentiment
  - CSV-importable event calendar (e.g. from investing.com export)

DISCLAIMER: Fundamental and news analysis is probabilistic.
No guarantee of directional accuracy.
"""

import re
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from backend.core.config import settings, PROCESSED_DIR


# ─── Event Catalog ────────────────────────────────────────────────────────────

# Known high-impact EUR/USD events and their typical directional bias rules.
# bias: +1 = EUR bullish, -1 = EUR bearish, 0 = uncertain/mixed
EVENT_CATALOG = {
    # ECB
    "ecb_rate_decision":         {"impact": "HIGH",   "currency": "EUR", "typical_bias": 0},
    "ecb_press_conference":      {"impact": "HIGH",   "currency": "EUR", "typical_bias": 0},
    "ecb_monetary_policy":       {"impact": "HIGH",   "currency": "EUR", "typical_bias": 0},

    # Federal Reserve
    "fomc_rate_decision":        {"impact": "HIGH",   "currency": "USD", "typical_bias": 0},
    "fed_press_conference":      {"impact": "HIGH",   "currency": "USD", "typical_bias": 0},
    "fed_minutes":               {"impact": "MEDIUM", "currency": "USD", "typical_bias": 0},

    # US Macro
    "us_nfp":                    {"impact": "HIGH",   "currency": "USD", "typical_bias": -1},  # strong NFP → USD up → EUR down
    "us_cpi":                    {"impact": "HIGH",   "currency": "USD", "typical_bias": -1},
    "us_gdp":                    {"impact": "HIGH",   "currency": "USD", "typical_bias": -1},
    "us_retail_sales":           {"impact": "MEDIUM", "currency": "USD", "typical_bias": -1},
    "us_pce":                    {"impact": "HIGH",   "currency": "USD", "typical_bias": -1},
    "us_ism_manufacturing":      {"impact": "MEDIUM", "currency": "USD", "typical_bias": -1},
    "us_jobless_claims":         {"impact": "MEDIUM", "currency": "USD", "typical_bias": 0},

    # Euro Area Macro
    "eurozone_cpi":              {"impact": "HIGH",   "currency": "EUR", "typical_bias": 1},
    "eurozone_gdp":              {"impact": "HIGH",   "currency": "EUR", "typical_bias": 1},
    "germany_ifo":               {"impact": "MEDIUM", "currency": "EUR", "typical_bias": 1},
    "eurozone_pmi":              {"impact": "MEDIUM", "currency": "EUR", "typical_bias": 0},
    "eurozone_retail_sales":     {"impact": "MEDIUM", "currency": "EUR", "typical_bias": 1},
    "eurozone_unemployment":     {"impact": "MEDIUM", "currency": "EUR", "typical_bias": 0},

    # UK Macro / BoE
    "boe_rate_decision":         {"impact": "HIGH",   "currency": "GBP", "typical_bias": 0},
    "uk_cpi":                    {"impact": "HIGH",   "currency": "GBP", "typical_bias": 1},
    "uk_gdp":                    {"impact": "HIGH",   "currency": "GBP", "typical_bias": 1},

    # Japan Macro / BoJ
    "boj_rate_decision":         {"impact": "HIGH",   "currency": "JPY", "typical_bias": 0},
    "japan_cpi":                 {"impact": "HIGH",   "currency": "JPY", "typical_bias": 1},
    "japan_gdp":                 {"impact": "MEDIUM", "currency": "JPY", "typical_bias": 1},

    # Broad risk events that often spill into equities
    "us_cpi_release":            {"impact": "HIGH",   "currency": "USD", "typical_bias": 0},
    "us_fomc_statement":         {"impact": "HIGH",   "currency": "USD", "typical_bias": 0},
}


# ─── Keyword → Sentiment Mapping ─────────────────────────────────────────────

# For news headline sentiment: words that suggest EUR/USD direction
BULLISH_EUR_KEYWORDS = [
    "ecb hike", "ecb raises", "higher rates", "inflation surge", "hawkish ecb",
    "strong eurozone", "euro zone growth", "ecb tightening", "lagarde hawkish",
    "eur rally", "euro gains", "weaker dollar", "fed pause", "fed cut",
    "dollar falls", "usd weakens",
]

BEARISH_EUR_KEYWORDS = [
    "ecb pause", "ecb cut", "rate cut ecb", "dovish ecb", "eurozone recession",
    "germany contraction", "euro weakens", "eur falls", "fed hike", "fomc hike",
    "hawkish fed", "strong nfp", "dollar rally", "usd strengthens", "risk off",
    "geopolitical risk europe",
]


def score_headline_sentiment(headline: str) -> float:
    """
    Returns a sentiment score for EUR/USD from a news headline:
      +1.0 = strongly bullish EUR (bearish USD)
      -1.0 = strongly bearish EUR (bullish USD)
       0.0 = neutral

    Simple keyword matching — not NLP. For production, replace with
    a fine-tuned model (e.g. FinBERT).
    """
    text = headline.lower()
    bull_count = sum(1 for kw in BULLISH_EUR_KEYWORDS if kw in text)
    bear_count = sum(1 for kw in BEARISH_EUR_KEYWORDS if kw in text)
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return round((bull_count - bear_count) / total, 3)


def score_event_actual_vs_forecast(
    event_type: str,
    actual: str,
    forecast: str,
    previous: str = "",
) -> float:
    """
    Given an economic event's actual vs forecast, estimate sentiment.
    Returns score: +1 = EUR bullish, -1 = EUR bearish, 0 = neutral.

    Logic:
    - If actual > forecast and event is EUR-positive → +score
    - If actual > forecast and event is USD-positive → -score
    - Magnitude scaled 0–1 based on relative surprise
    """
    meta = EVENT_CATALOG.get(event_type.lower(), {})
    currency = meta.get("currency", "")
    default_bias = meta.get("typical_bias", 0)

    try:
        act_val = float(re.sub(r"[^0-9.\-]", "", actual))
        fct_val = float(re.sub(r"[^0-9.\-]", "", forecast))
    except (ValueError, TypeError):
        return 0.0

    surprise = act_val - fct_val
    if fct_val != 0:
        surprise_pct = abs(surprise / fct_val)
    else:
        surprise_pct = abs(surprise) * 0.1

    magnitude = min(surprise_pct, 1.0)
    direction = 1.0 if surprise > 0 else -1.0

    # Adjust by currency bias
    if currency == "EUR":
        score = direction * magnitude
    elif currency == "USD":
        score = -direction * magnitude  # USD positive = EUR negative
    else:
        score = default_bias * magnitude

    return round(score, 3)


# ─── Event Calendar ───────────────────────────────────────────────────────────

def load_event_calendar(csv_path: str = None) -> pd.DataFrame:
    """
    Load an economic event calendar.
    Accepts CSV with columns: date, name, currency, impact, actual, forecast, previous

    If no CSV provided, generates a synthetic stub for demonstration.
    In production: export from investing.com, Forex Factory, or use an API.
    """
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        logger.info(f"Loaded {len(df)} economic events from {csv_path}")
        return df

    # Generate synthetic example events with both recent history and upcoming events
    logger.warning("No event calendar CSV provided. Using synthetic stub events.")
    events = []
    now = datetime.utcnow()
    # Cover a near-future window so dashboard can surface upcoming events.
    for offset_days in range(-90, 121, 7):
        dt = now + timedelta(days=offset_days)
        if dt.weekday() == 4:  # Fridays: NFP-ish
            events.append({
                "date": dt.replace(hour=13, minute=30),
                "name": "US Non-Farm Payrolls",
                "event_type": "us_nfp",
                "currency": "USD",
                "impact": "HIGH",
                "actual": "",
                "forecast": "",
                "previous": "",
            })
        if dt.weekday() == 3 and dt.day <= 14:  # ~monthly ECB
            events.append({
                "date": dt.replace(hour=12, minute=15),
                "name": "ECB Rate Decision",
                "event_type": "ecb_rate_decision",
                "currency": "EUR",
                "impact": "HIGH",
                "actual": "",
                "forecast": "",
                "previous": "",
            })
        if dt.weekday() == 3 and dt.day > 14:  # ~monthly BoE
            events.append({
                "date": dt.replace(hour=11, minute=0),
                "name": "BoE Rate Decision",
                "event_type": "boe_rate_decision",
                "currency": "GBP",
                "impact": "HIGH",
                "actual": "",
                "forecast": "",
                "previous": "",
            })
        if dt.weekday() == 2 and dt.day <= 14:  # ~monthly BoJ
            events.append({
                "date": dt.replace(hour=3, minute=0),
                "name": "BoJ Rate Decision",
                "event_type": "boj_rate_decision",
                "currency": "JPY",
                "impact": "HIGH",
                "actual": "",
                "forecast": "",
                "previous": "",
            })

    df = pd.DataFrame(events)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.sort_values("date", inplace=True)
    return df


def flag_event_windows(
    df: pd.DataFrame,
    events: pd.DataFrame,
    pre_hours: int = 4,
    post_hours: int = 2,
) -> pd.DataFrame:
    """
    Mark bars that fall within a high-impact event window.
    Signals generated in event windows are flagged as high-risk
    and suppressed unless confidence is very high.

    pre_hours: hours before event to start flagging
    post_hours: hours after event to continue flagging
    """
    df["event_window"] = False
    df["event_name"] = ""
    df["event_impact"] = ""
    df["event_sentiment"] = 0.0

    high_impact = events[events.get("impact", "LOW") == "HIGH"] if not events.empty else pd.DataFrame()

    for _, ev in high_impact.iterrows():
        ev_time = pd.Timestamp(ev["date"])
        window_start = ev_time - timedelta(hours=pre_hours)
        window_end = ev_time + timedelta(hours=post_hours)

        mask = (df.index >= window_start) & (df.index <= window_end)
        df.loc[mask, "event_window"] = True
        df.loc[mask, "event_name"] = ev.get("name", "")
        df.loc[mask, "event_impact"] = ev.get("impact", "")

        sentiment = score_event_actual_vs_forecast(
            ev.get("event_type", ""),
            str(ev.get("actual", "")),
            str(ev.get("forecast", "")),
        )
        df.loc[mask, "event_sentiment"] = sentiment

    return df


def get_event_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Scalar risk score per bar: 1.0 = high risk (near event), 0 = no event.
    Used to penalize signal confidence during risky windows.
    """
    return df["event_window"].astype(float)


# ─── NewsAPI Integration (Optional) ──────────────────────────────────────────

def fetch_news_sentiment(query: str = "EUR USD forex", max_articles: int = 10) -> float:
    """
    Fetch recent news from NewsAPI and compute aggregate sentiment.
    Returns a float in [-1, +1] representing EUR/USD directional bias.

    Requires NEWSAPI_KEY in .env. Returns 0.0 if key not set.
    """
    if not settings.newsapi_key:
        logger.debug("NewsAPI key not configured. Skipping news sentiment.")
        return 0.0

    try:
        import requests
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": settings.newsapi_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            return 0.0

        scores = [score_headline_sentiment(a.get("title", "") + " " + a.get("description", ""))
                  for a in articles]
        return round(float(np.mean(scores)), 3)

    except Exception as e:
        logger.warning(f"News sentiment fetch failed: {e}")
        return 0.0


def _pair_relevant_currencies(pair: str | None) -> set[str]:
    if not pair:
        return set()
    normalized = pair.upper().replace(" ", "")
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
        return {base, quote}
    if len(normalized) >= 6:
        return {normalized[:3], normalized[3:6]}
    return set()


def get_upcoming_high_impact_events(
    pair: str | None = None,
    hours_ahead: int = 168,
    max_items: int = 8,
    csv_path: str | None = None,
) -> list[dict]:
    """Return upcoming HIGH-impact events, optionally filtered for a pair."""
    events = load_event_calendar(csv_path=csv_path)
    if events.empty:
        return []

    events = events.copy()
    events["date"] = pd.to_datetime(events["date"], utc=True, errors="coerce")
    events = events.dropna(subset=["date"])
    events["impact"] = events.get("impact", "").astype(str).str.upper()
    events["currency"] = events.get("currency", "").astype(str).str.upper()

    now = pd.Timestamp.utcnow()
    horizon = now + pd.Timedelta(hours=max(1, hours_ahead))
    filtered = events[
        (events["impact"] == "HIGH")
        & (events["date"] >= now)
        & (events["date"] <= horizon)
    ]

    relevant_ccy = _pair_relevant_currencies(pair)
    if relevant_ccy:
        filtered = filtered[filtered["currency"].isin(relevant_ccy)]

    filtered = filtered.sort_values("date").head(max_items)
    results: list[dict] = []
    for _, row in filtered.iterrows():
        results.append(
            {
                "date": row["date"].isoformat(),
                "name": str(row.get("name", "Unknown Event")),
                "event_type": str(row.get("event_type", "")),
                "currency": str(row.get("currency", "")),
                "impact": str(row.get("impact", "HIGH")),
            }
        )
    return results


def fetch_market_news_headlines(query: str = "stock market OR central bank OR inflation", max_articles: int = 5) -> list[dict]:
    """Fetch latest market headlines when NewsAPI key is configured."""
    if not settings.newsapi_key:
        return []

    try:
        import requests

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": settings.newsapi_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        items = []
        for article in data.get("articles", [])[:max_articles]:
            items.append(
                {
                    "title": article.get("title", ""),
                    "source": (article.get("source") or {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                }
            )
        return items
    except Exception as exc:
        logger.warning(f"Market headline fetch failed: {exc}")
        return []
