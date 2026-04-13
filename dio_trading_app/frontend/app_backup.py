"""
Dio Trading App — Streamlit Dashboard
=======================================
Simplified dashboard for multi-market analysis, live signals, backtesting,
and ML probability checks.

Run with:
    streamlit run frontend/app.py
"""

from datetime import datetime, timezone
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


DEFAULT_API_BASE = os.getenv("DIO_API_BASE_URL", "http://localhost:8000").rstrip("/")
API_HEALTH_PATH = "/health"
API_DISCOVERY_PORTS = (8000, 8001, 8002, 8003, 8004, 8005)
SUPPORTED_PAIRS = ["EUR/USD", "JPY/USD", "XAU/USD"]
SUPPORTED_TIMEFRAMES = ["1d", "4h", "1h", "15m"]
ALLOWED_SIGNAL_TIMEFRAMES = {
    "EUR/USD": ["1d", "4h", "1h", "15m"],
    "JPY/USD": ["1d", "4h", "15m"],
    "XAU/USD": ["1d", "4h", "1h", "15m"],
}


def _allowed_timeframes_for_pair(pair_name: str) -> list[str]:
    return ALLOWED_SIGNAL_TIMEFRAMES.get(pair_name, SUPPORTED_TIMEFRAMES)


def _all_allowed_pair_timeframe_combos() -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for p in SUPPORTED_PAIRS:
        for tf in _allowed_timeframes_for_pair(p):
            combos.append((p, tf))
    return combos


st.set_page_config(
    page_title="Dio Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.disclaimer-box {
    background: #1a1a2e;
    border: 1px solid #f0a500;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #f0a500;
}
</style>
""",
    unsafe_allow_html=True,
)


def _normalise_params(params: dict | None) -> tuple[tuple[str, str], ...]:
    if not params:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in params.items()))


def _local_timezone_label() -> str:
    return datetime.now().astimezone().tzname() or "Local"


def _format_local_time(timestamp_value: str | None, fallback: str = "—") -> str:
    if not timestamp_value:
        return fallback

    try:
        normalized = timestamp_value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        local_dt = parsed.astimezone()
        return local_dt.strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return fallback


def _price_format(pair_name: str, price: float) -> str:
    if pair_name == "XAU/USD":
        decimals = 2
    elif "JPY" in pair_name:
        decimals = 3
    else:
        decimals = 5
    return f"{price:.{decimals}f}"


def _price_delta_label(pair_name: str, price_delta: float) -> str:
    if pair_name == "XAU/USD":
        return f"{price_delta:+.2f} USD"
    pip_scale = 100 if "JPY" in pair_name else 10000
    if "JPY" in pair_name:
        return f"{price_delta * pip_scale:+.1f} pips"
    return f"{price_delta * pip_scale:+.1f} pips"


def _candidate_api_bases(configured_base: str) -> list[str]:
    candidates = [configured_base]
    for port in API_DISCOVERY_PORTS:
        candidate = f"http://localhost:{port}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _tp_hit_future_search(
    pair_name: str,
    direction: str,
    take_profit: float,
    current_price: float,
) -> dict:
    """
    Detects if TP was hit and provides guidance on future entry search.
    
    Returns dict with:
    - tp_hit: bool (True if price reached or passed TP)
    - future_search_zone: price range to look for new entries
    - search_guidance: text describing where to monitor
    """
    tolerance = 0.0001  # Small tolerance for TP hit detection
    
    if direction == "BUY":
        tp_hit = current_price >= (take_profit - tolerance)
        if tp_hit:
            # After BUY TP hit, look for pullback or consolidation
            reentry_lower = take_profit * 0.98  # 2% pullback
            reentry_upper = take_profit * 1.02  # 2% extension
            return {
                "tp_hit": True,
                "future_search_zone_low": reentry_lower,
                "future_search_zone_high": reentry_upper,
                "search_guidance": (
                    f"OLD SIGNAL COMPLETED ✅ | Look for NEW BUY setup in future bars. "
                    f"Monitor consolidation zone: {_price_format(pair_name, reentry_lower)} - {_price_format(pair_name, reentry_upper)}. "
                    f"New entry may form after fresh technical alignment."
                ),
            }
    else:  # SELL
        tp_hit = current_price <= (take_profit + tolerance)
        if tp_hit:
            # After SELL TP hit, look for bounce or consolidation
            reentry_lower = take_profit * 0.98  # 2% pullback
            reentry_upper = take_profit * 1.02  # 2% extension
            return {
                "tp_hit": True,
                "future_search_zone_low": reentry_lower,
                "future_search_zone_high": reentry_upper,
                "search_guidance": (
                    f"OLD SIGNAL COMPLETED ✅ | Look for NEW SELL setup in future bars. "
                    f"Monitor consolidation zone: {_price_format(pair_name, reentry_lower)} - {_price_format(pair_name, reentry_upper)}. "
                    f"New entry may form after fresh technical alignment."
                ),
            }
    
    return {"tp_hit": False, "future_search_zone_low": None, "future_search_zone_high": None, "search_guidance": ""}


def _sl_hit_search_new_signals(
    pair_name: str,
    direction: str,
    stop_loss: float,
    current_price: float,
) -> dict:
    """
    When SL is hit, search for new future signals to enter.
    
    Returns dict with:
    - sl_hit: bool
    - new_signals: list of suggested new signals
    - search_zone: where to look for new entries
    - guidance: next steps
    """
    tolerance = 0.0001
    
    if direction == "BUY":
        sl_hit = current_price <= (stop_loss + tolerance)
    else:  # SELL
        sl_hit = current_price >= (stop_loss - tolerance)
    
    if not sl_hit:
        return {"sl_hit": False, "new_signals": [], "search_zone": None, "guidance": ""}
    
    # SL was hit - search for new signals
    try:
        new_sigs = requests.get(
            f"{get_api_base()}/signals/multi-tf",
            params={"pair": pair_name, "scan_bars": 5, "signal_mode": "balanced"},
            timeout=15,
        ).json()
        
        matching_signals = []
        if new_sigs.get("signals"):
            # Filter for matching direction or opportunistic opposite
            for sig in new_sigs.get("signals", []):
                if sig.get("direction") == direction:
                    matching_signals.append(sig)
        
        search_zone_low = stop_loss * 0.98  # 2% below SL
        search_zone_high = stop_loss * 1.02  # 2% above SL
        
        return {
            "sl_hit": True,
            "new_signals": matching_signals,
            "search_zone": (search_zone_low, search_zone_high),
            "guidance": (
                f"OLD SIGNAL STOPPED | 🔍 Scanning for NEW future {direction} opportunities... "
                f"Monitor: {_price_format(pair_name, search_zone_low)} - {_price_format(pair_name, search_zone_high)}"
            ),
        }
    except Exception:
        return {
            "sl_hit": True,
            "new_signals": [],
            "search_zone": None,
            "guidance": "OLD SIGNAL STOPPED | Searching for new signals...",
        }


def _running_signal_status(
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
) -> dict:
    """
    Determine running signal status (active, pending, danger zone).
    
    Returns dict with:
    - status: active, pending, danger_zone
    - color: green (active), yellow (pending), orange (danger)
    - label: text label with emoji
    - profit_pct: unrealized profit percentage
    """
    risk = max(abs(entry_price - stop_loss), 1e-9)

    if direction == "BUY":
        if current_price <= stop_loss:
            return {
                "status": "stopped",
                "color": "red",
                "label": "🔴 STOPPED",
                "profit_pct": -1.0 * (abs(current_price - entry_price) / risk),
            }

        profit_pct = (current_price - entry_price) / risk

        if current_price >= take_profit:
            return {
                "status": "completed",
                "color": "blue",
                "label": "🔵 TP HIT",
                "profit_pct": 1.0,
            }
        if current_price >= entry_price:
            return {
                "status": "active",
                "color": "green",
                "label": f"🟢 RUNNING +{profit_pct * 100:.1f}R",
                "profit_pct": profit_pct,
            }

        danger_pct = (entry_price - current_price) / risk
        if danger_pct > 0.7:
            return {
                "status": "danger_zone",
                "color": "orange",
                "label": f"🟠 DANGER -{danger_pct * 100:.1f}R (near SL)",
                "profit_pct": -danger_pct,
            }
        return {
            "status": "pending",
            "color": "yellow",
            "label": f"🟡 PENDING -{danger_pct * 100:.1f}R",
            "profit_pct": -danger_pct,
        }

    if current_price >= stop_loss:
        return {
            "status": "stopped",
            "color": "red",
            "label": "🔴 STOPPED",
            "profit_pct": -1.0 * (abs(current_price - entry_price) / risk),
        }

    profit_pct = (entry_price - current_price) / risk

    if current_price <= take_profit:
        return {
            "status": "completed",
            "color": "blue",
            "label": "🔵 TP HIT",
            "profit_pct": 1.0,
        }
    if current_price <= entry_price:
        return {
            "status": "active",
            "color": "green",
            "label": f"🟢 RUNNING +{profit_pct * 100:.1f}R",
            "profit_pct": profit_pct,
        }

    danger_pct = (current_price - entry_price) / risk
    if danger_pct > 0.7:
        return {
            "status": "danger_zone",
            "color": "orange",
            "label": f"🟠 DANGER -{danger_pct * 100:.1f}R (near SL)",
            "profit_pct": -danger_pct,
        }
    return {
        "status": "pending",
        "color": "yellow",
        "label": f"🟡 PENDING -{danger_pct * 100:.1f}R",
        "profit_pct": -danger_pct,
    }


def _resolve_signal_time(signal_item: dict, scan_timestamp: str | None = None) -> str:
    for key in ("timestamp", "signal_time", "created_at", "generated_at", "bar_time", "analysis_time"):
        value = signal_item.get(key)
        if value:
            return _format_local_time(str(value))
    if scan_timestamp:
        return _format_local_time(scan_timestamp)
    return datetime.now().astimezone().strftime("%Y-%m-%d %I:%M %p")


def _status_label_with_time(status_word: str, when_text: str) -> str:
    return f"{status_word} | {when_text} {_local_timezone_label()}"


def _signal_lifecycle_status_with_time(
    signal_item: dict,
    current_price: float | None,
    scan_timestamp: str | None,
) -> tuple[str, str]:
    signal_time = _resolve_signal_time(signal_item, scan_timestamp)

    if current_price is None:
        return "UPCOMING", _status_label_with_time("UPCOMING", signal_time)

    direction = str(signal_item.get("direction", "")).upper()
    entry = float(signal_item.get("entry_price", 0.0) or 0.0)
    sl = float(signal_item.get("stop_loss", 0.0) or 0.0)
    tp = float(signal_item.get("take_profit", 0.0) or 0.0)

    if direction not in ("BUY", "SELL") or entry == 0.0 or sl == 0.0 or tp == 0.0:
        return "UPCOMING", _status_label_with_time("UPCOMING", signal_time)

    running = _running_signal_status(
        current_price=float(current_price),
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        direction=direction,
    )
    if running.get("status") in ("stopped", "completed"):
        return "EXPIRED", _status_label_with_time("EXPIRED", signal_time)
    return "RUNNING", _status_label_with_time("RUNNING", signal_time)


def _signal_outcome(current_price: float, signal_item: dict) -> str:
    direction = str(signal_item.get("direction", "")).upper()
    entry = float(signal_item.get("entry_price", 0.0) or 0.0)
    sl = float(signal_item.get("stop_loss", 0.0) or 0.0)
    tp = float(signal_item.get("take_profit", 0.0) or 0.0)
    if direction not in ("BUY", "SELL") or entry == 0.0 or sl == 0.0 or tp == 0.0:
        return "unknown"

    running = _running_signal_status(
        current_price=float(current_price),
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        direction=direction,
    )
    if running.get("status") == "completed":
        return "tp_hit"
    if running.get("status") == "stopped":
        return "sl_hit"
    return "running"


def _compact_signal_explanation(sig: dict, overview_item: dict | None = None) -> str:
    tf = str(sig.get("timeframe", "?")).upper()
    direction = str(sig.get("direction", "?")).upper()
    ml = float(sig.get("model_probability", 0.0) or 0.0) * 100
    win = float(sig.get("win_probability", 0.0) or 0.0) * 100
    hist = float(sig.get("historical_win_rate", 0.0) or 0.0) * 100
    conf = float(sig.get("confidence_score", 0.0) or 0.0)

    liq = str(sig.get("liquidity_zone", "unknown"))
    vol = str(sig.get("market_condition", "unknown"))
    trend = str(sig.get("trend_direction", "unknown"))
    if overview_item:
        liq = str(overview_item.get("liquidity_zone", liq))
        trend = str(overview_item.get("trend_primary", trend))

    factors = sig.get("contributing_factors", []) or []
    candle_factor = "No clear candle trigger"
    for factor in factors:
        txt = str(factor)
        lowered = txt.lower()
        if "macd" in lowered or "sweep" in lowered or "bollinger" in lowered:
            candle_factor = txt
            break

    return (
        f"{direction} [{tf}] | Conf {conf:.1f}/100 | ML {ml:.1f}% | Win {win:.1f}% | Hist WR {hist:.1f}% | "
        f"Liquidity {liq} | Volume {vol} | Trend {trend} | Candle/Trigger: {candle_factor}"
    )


def _build_three_type_buckets(
    pair_name: str,
    sig_data: dict,
    live_price: float | None,
) -> dict:
    allowed_tfs = _allowed_timeframes_for_pair(pair_name)
    raw_signals = sig_data.get("signals", []) or []
    signals = [
        s for s in raw_signals
        if str(s.get("timeframe", "")).lower() in allowed_tfs
    ]

    raw_overview = sig_data.get("timeframe_overview", []) or []
    overview_lookup = {
        str(item.get("timeframe", "")).lower(): item
        for item in raw_overview
    }

    buckets = {"running": [], "upcoming": [], "passed": []}
    scan_ts = sig_data.get("timestamp")

    for tf in allowed_tfs:
        tf_upper = tf.upper()
        overview_item = overview_lookup.get(tf, {"timeframe": tf})
        tf_signal = next((s for s in signals if str(s.get("timeframe", "")).lower() == tf), None)

        if tf_signal:
            started_at = _resolve_signal_time(tf_signal, scan_ts)
            if live_price is None:
                buckets["upcoming"].append(
                    {
                        "pair": pair_name,
                        "timeframe": tf_upper,
                        "started_at": started_at,
                        "status": "UPCOMING",
                        "color": "purple",
                        "explanation": _compact_signal_explanation(tf_signal, overview_item),
                        "signal": tf_signal,
                    }
                )
                continue

            outcome = _signal_outcome(float(live_price), tf_signal)
            if outcome == "running":
                buckets["running"].append(
                    {
                        "pair": pair_name,
                        "timeframe": tf_upper,
                        "started_at": started_at,
                        "status": "RUNNING",
                        "color": "green",
                        "explanation": _compact_signal_explanation(tf_signal, overview_item),
                        "signal": tf_signal,
                    }
                )
            else:
                outcome_text = "TP HIT" if outcome == "tp_hit" else "SL HIT" if outcome == "sl_hit" else "CLOSED"
                buckets["passed"].append(
                    {
                        "pair": pair_name,
                        "timeframe": tf_upper,
                        "started_at": started_at,
                        "status": f"PASSED ({outcome_text})",
                        "color": "red",
                        "explanation": _compact_signal_explanation(tf_signal, overview_item),
                        "signal": tf_signal,
                    }
                )
        else:
            diagnostics = overview_item.get("signal_diagnostics", {}) or {}
            buy = diagnostics.get("buy", {})
            sell = diagnostics.get("sell", {})
            buy_conf = float(buy.get("confidence", 0) or 0)
            sell_conf = float(sell.get("confidence", 0) or 0)
            buy_prob = float(buy.get("model_probability", 0) or 0)
            sell_prob = float(sell.get("model_probability", 0) or 0)

            likely_side = "BUY" if (buy_conf + buy_prob * 100) >= (sell_conf + sell_prob * 100) else "SELL"
            likely_conf = buy_conf if likely_side == "BUY" else sell_conf
            likely_prob = buy_prob if likely_side == "BUY" else sell_prob
            label = "likely soon" if likely_conf >= 50 or likely_prob >= 0.50 else "watch"

            buckets["upcoming"].append(
                {
                    "pair": pair_name,
                    "timeframe": tf_upper,
                    "started_at": _format_local_time(scan_ts),
                    "status": f"UPCOMING ({label})",
                    "color": "purple",
                    "explanation": (
                        f"{likely_side} সম্ভাবনা | Conf {likely_conf:.1f}/100 | ML {likely_prob * 100:.1f}% | "
                        f"Liquidity {overview_item.get('liquidity_zone', 'unknown')} | "
                        f"Volume {overview_item.get('signal_diagnostics', {}).get('summary', 'monitor setup')}"
                    ),
                    "signal": None,
                }
            )

    # ── Smart-rank running signals: best entry first ──
    def _rank_score(item: dict) -> float:
        sig = item.get("signal") or {}
        conf = float(sig.get("confidence_score", 0) or 0)
        ml = float(sig.get("model_probability", 0) or 0) * 100
        wr = float(sig.get("win_probability", 0) or 0) * 100
        hist = float(sig.get("historical_win_rate", 0) or 0) * 100
        risk_score = float(sig.get("risk_score", 50) or 50)
        # Higher is better: confidence + ML + win + hist − risk penalty
        return conf * 0.30 + ml * 0.25 + wr * 0.25 + hist * 0.10 - risk_score * 0.10

    buckets["running"].sort(key=_rank_score, reverse=True)
    # Assign rank labels
    for idx, item in enumerate(buckets["running"], 1):
        item["rank"] = idx
        item["rank_score"] = round(_rank_score(item), 1)

    return buckets


def _render_three_type_sections(pair_name: str, buckets: dict) -> None:
    st.markdown("---")
    st.markdown("## 3-Type Signal Analysis")

    r_count = len(buckets.get("running", []))
    u_count = len(buckets.get("upcoming", []))
    p_count = len(buckets.get("passed", []))
    c1, c2, c3 = st.columns(3)
    c1.metric("1) Running", r_count)
    c2.metric("2) Upcoming", u_count)
    c3.metric("3) Passed", p_count)

    st.markdown("### 1) Running Signals (Best entry first)")
    if not buckets.get("running"):
        st.info("No running signal right now.")
    for item in buckets.get("running", []):
        sig = item.get("signal") or {}
        rank = item.get("rank", "?")
        rank_score = item.get("rank_score", 0)
        rank_badge = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        st.success(
            f"{rank_badge} {item.get('pair')} {item.get('timeframe')} | {item.get('status')} | "
            f"Score {rank_score} | Started: {item.get('started_at', '—')} {_local_timezone_label()}"
        )
        st.caption(item.get("explanation", ""))
        entry_p = float(sig.get("entry_price", 0) or 0)
        sl_p = float(sig.get("stop_loss", 0) or 0)
        tp_p = float(sig.get("take_profit", 0) or 0)
        st.caption(
            f"Entry {_price_format(pair_name, entry_p)} | "
            f"SL {_price_format(pair_name, sl_p)} | "
            f"TP {_price_format(pair_name, tp_p)} | "
            f"R:R 1:{float(sig.get('risk_reward', 2) or 2):.1f}"
        )

    st.markdown("### 2) Upcoming Signals (Get ready)")
    if not buckets.get("upcoming"):
        st.info("No upcoming setup detected.")
    for item in buckets.get("upcoming", []):
        st.info(
            f"🟣 {item.get('pair')} {item.get('timeframe')} | {item.get('status')} | "
            f"Time: {item.get('started_at', '—')} {_local_timezone_label()}"
        )
        st.caption(item.get("explanation", ""))

    st.markdown("### 3) Passed Signals (TP/SL reached)")
    if not buckets.get("passed"):
        st.info("No passed signal yet.")
    for item in buckets.get("passed", []):
        sig = item.get("signal") or {}
        status_text = item.get("status", "PASSED")
        is_tp = "TP HIT" in status_text
        badge = "✅ TP HIT — Profit" if is_tp else "❌ SL HIT — Loss"
        color_fn = st.success if is_tp else st.error
        color_fn(
            f"{badge} | {item.get('pair')} {item.get('timeframe')} | "
            f"Started: {item.get('started_at', '—')} {_local_timezone_label()}"
        )
        st.caption(item.get("explanation", ""))
        entry_p = float(sig.get("entry_price", 0) or 0)
        sl_p = float(sig.get("stop_loss", 0) or 0)
        tp_p = float(sig.get("take_profit", 0) or 0)
        st.caption(
            f"Entry {_price_format(pair_name, entry_p)} | "
            f"SL {_price_format(pair_name, sl_p)} | "
            f"TP {_price_format(pair_name, tp_p)}"
        )


@st.cache_data(ttl=15, show_spinner=False)
def _discover_api_base(configured_base: str) -> str:
    for base_url in _candidate_api_bases(configured_base):
        try:
            response = requests.get(f"{base_url}{API_HEALTH_PATH}", timeout=(1.5, 1.5))
            response.raise_for_status()
            return base_url.rstrip("/")
        except requests.RequestException:
            continue
    raise RuntimeError(
        "No healthy backend responded on localhost ports 8000-8005. "
        "Start the API or run start.py so the dashboard can discover it."
    )


def get_api_base(force_refresh: bool = False) -> str:
    if force_refresh:
        _discover_api_base.clear()
    resolved = _discover_api_base(DEFAULT_API_BASE)
    st.session_state["resolved_api_base"] = resolved
    return resolved


@st.cache_data(ttl=20, show_spinner=False)
def _api_get_cached(
    api_base: str,
    endpoint: str,
    params_key: tuple[tuple[str, str], ...],
    timeout: int,
) -> dict:
    response = requests.get(
        f"{api_base}{endpoint}",
        params=dict(params_key),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def render_sidebar_logo() -> None:
    st.sidebar.markdown(
        """
        <div style="padding: 10px 12px; border: 1px solid #21262d; border-radius: 10px; background: #111827;">
            <div style="font-size: 11px; letter-spacing: 0.18em; color: #9ca3af; margin-bottom: 4px;">DIO</div>
            <div style="font-size: 20px; font-weight: 700; color: #f8fafc; line-height: 1;">TRADING</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">Signal Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def api_get(endpoint: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    api_base = get_api_base()
    try:
        return _api_get_cached(api_base, endpoint, _normalise_params(params), timeout)
    except requests.RequestException:
        try:
            api_base = get_api_base(force_refresh=True)
            return _api_get_cached(api_base, endpoint, _normalise_params(params), timeout)
        except Exception as exc:
            st.error(f"API error ({endpoint}) via {api_base}: {exc}")
            return None
    except Exception as exc:
        st.error(f"API error ({endpoint}) via {api_base}: {exc}")
        return None


def api_post(endpoint: str, json: dict | None = None) -> dict | None:
    api_base = get_api_base()
    try:
        response = requests.post(f"{api_base}{endpoint}", json=json or {}, timeout=180)
        response.raise_for_status()
        _api_get_cached.clear()
        return response.json()
    except requests.RequestException:
        try:
            api_base = get_api_base(force_refresh=True)
            response = requests.post(f"{api_base}{endpoint}", json=json or {}, timeout=180)
            response.raise_for_status()
            _api_get_cached.clear()
            return response.json()
        except Exception as exc:
            st.error(f"API error ({endpoint}) via {api_base}: {exc}")
            return None
    except Exception as exc:
        st.error(f"API error ({endpoint}) via {api_base}: {exc}")
        return None


def price_move_label(pair_name: str, price_delta: float) -> str:
    if pair_name == "XAU/USD":
        return f"{price_delta:.2f} USD"
    pip_scale = 100 if "JPY" in pair_name else 10000
    return f"{price_delta * pip_scale:.1f} pips"


def _entry_reentry_guidance(
    pair_name: str,
    direction: str,
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> dict:
    """
    Provides entry/re-entry guidance with color-coded zones.
    
    Returns dict with:
    - status: entry_active, entry_missed, wait_entry, tp_reached, invalidated
    - message: Main status message
    - reentry: Re-entry zone guidance
    - reentry_zone_low/high: Exact re-entry zone prices
    - color_code: Color suggestion for display (green/yellow/red/blue)
    - next_action: What to do next (look for fresh setup, etc.)
    """
    risk_distance = abs(entry_price - stop_loss)
    zone_half_width = max(risk_distance * 0.15, 0.0001)
    zone_low = entry_price - zone_half_width
    zone_high = entry_price + zone_half_width

    if direction == "BUY":
        if current_price <= stop_loss:
            return {
                "status": "invalidated",
                "message": "🔴 STOP-LOSS HIT — Signal invalidated",
                "color_code": "red",
                "reentry": "Old signal is dead. Look for NEW fresh entry setup after price stabilizes.",
                "reentry_zone_low": stop_loss,
                "reentry_zone_high": stop_loss,
                "next_action": "Search for fresh BUY setup after price recovers above recovery zone.",
            }
        if current_price < zone_low:
            return {
                "status": "wait_entry",
                "message": "⬇️ Price below entry zone — waiting for entry pullback",
                "color_code": "yellow",
                "reentry": None,
                "reentry_zone_low": zone_low,
                "reentry_zone_high": zone_high,
                "next_action": "Wait for price to rise into green entry zone.",
            }
        if zone_low <= current_price <= zone_high:
            return {
                "status": "entry_active",
                "message": "🟢 ENTRY ZONE ACTIVE — You can enter now",
                "color_code": "green",
                "reentry": (
                    f"✅ Re-entry zone safe to enter: {_price_format(pair_name, zone_low)} to {_price_format(pair_name, zone_high)}"
                ),
                "reentry_zone_low": zone_low,
                "reentry_zone_high": zone_high,
                "next_action": "Enter BUY trade with SL at " + _price_format(pair_name, stop_loss),
            }
        if current_price < take_profit:
            return {
                "status": "entry_missed",
                "message": "⬆️ Entry missed — Price moved above entry zone",
                "color_code": "yellow",
                "reentry": (
                    f"🟢 RE-ENTRY PULLBACK ZONE: You can re-enter if price pulls back to "
                    f"{_price_format(pair_name, zone_low)} - {_price_format(pair_name, zone_high)} "
                    f"(±15% of risk from entry {_price_format(pair_name, entry_price)})"
                ),
                "reentry_zone_low": zone_low,
                "reentry_zone_high": zone_high,
                "next_action": "Watch for pullback into green re-entry zone before TP.",
            }
        return {
            "status": "tp_reached",
            "message": "🔵 TAKE-PROFIT REACHED — Old signal has closed out",
            "color_code": "blue",
            "reentry": (
                f"🆕 OLD SIGNAL OVER — Look in future for FRESH entry setup. "
                f"After TP is reached, current market conditions change. "
                f"Wait for price consolidation, then scan for new BUY signals."
            ),
            "reentry_zone_low": take_profit,
            "reentry_zone_high": take_profit,
            "next_action": "Signal job done. Monitor for new entry opportunities in upcoming bars.",
        }

    # SELL direction
    if current_price >= stop_loss:
        return {
            "status": "invalidated",
            "message": "🔴 STOP-LOSS HIT — Signal invalidated",
            "color_code": "red",
            "reentry": "Old signal is dead. Look for NEW fresh entry setup after price stabilizes.",
            "reentry_zone_low": stop_loss,
            "reentry_zone_high": stop_loss,
            "next_action": "Search for fresh SELL setup after price recovers below recovery zone.",
        }
    if current_price > zone_high:
        return {
            "status": "wait_entry",
            "message": "⬆️ Price above entry zone — waiting for entry pullback",
            "color_code": "yellow",
            "reentry": None,
            "reentry_zone_low": zone_low,
            "reentry_zone_high": zone_high,
            "next_action": "Wait for price to fall into green entry zone.",
        }
    if zone_low <= current_price <= zone_high:
        return {
            "status": "entry_active",
            "message": "🟢 ENTRY ZONE ACTIVE — You can enter now",
            "color_code": "green",
            "reentry": (
                f"✅ Re-entry zone safe to enter: {_price_format(pair_name, zone_low)} to {_price_format(pair_name, zone_high)}"
            ),
            "reentry_zone_low": zone_low,
            "reentry_zone_high": zone_high,
            "next_action": "Enter SELL trade with SL at " + _price_format(pair_name, stop_loss),
        }
    if current_price > take_profit:
        return {
            "status": "entry_missed",
            "message": "⬇️ Entry missed — Price moved below entry zone",
            "color_code": "yellow",
            "reentry": (
                f"🟢 RE-ENTRY PULLBACK ZONE: You can re-enter if price bounces back to "
                f"{_price_format(pair_name, zone_low)} - {_price_format(pair_name, zone_high)} "
                f"(±15% of risk from entry {_price_format(pair_name, entry_price)})"
            ),
            "reentry_zone_low": zone_low,
            "reentry_zone_high": zone_high,
            "next_action": "Watch for bounce back into green re-entry zone before TP.",
        }
    return {
        "status": "tp_reached",
        "message": "🔵 TAKE-PROFIT REACHED — Old signal has closed out",
        "color_code": "blue",
        "reentry": (
            f"🆕 OLD SIGNAL OVER — Look in future for FRESH entry setup. "
            f"After TP is reached, current market conditions change. "
            f"Wait for price consolidation, then scan for new SELL signals."
        ),
        "reentry_zone_low": take_profit,
        "reentry_zone_high": take_profit,
        "next_action": "Signal job done. Monitor for new entry opportunities in upcoming bars.",
    }


def render_upcoming_market_section(pair: str) -> None:
    st.subheader("Upcoming High-Impact Events & Market News")
    upcoming = api_get("/market/upcoming", {"pair": pair, "hours_ahead": 168, "limit": 8})
    if not upcoming:
        st.info("Unable to load upcoming events right now.")
        st.markdown("---")
        return

    events = upcoming.get("upcoming_events", [])
    headlines = upcoming.get("news_headlines", [])

    if events:
        st.markdown("**Next 7 days: high-impact events**")
        for event in events:
            event_time = _format_local_time(event.get("date"), fallback="—")
            st.write(
                f"- {event_time} {_local_timezone_label()} | "
                f"{event.get('currency', 'N/A')} | {event.get('name', 'Event')}"
            )
    else:
        st.caption("No high-impact events scheduled in the selected window.")

    if headlines:
        st.markdown("**Latest market headlines**")
        for item in headlines[:5]:
            title = item.get("title", "Market headline")
            source = item.get("source", "Unknown source")
            published = _format_local_time(item.get("published_at"), fallback="")
            url = item.get("url", "")
            suffix = f" ({source}{' | ' + published if published else ''})"
            if url:
                st.markdown(f"- [{title}]({url}){suffix}")
            else:
                st.write(f"- {title}{suffix}")
    else:
        st.caption("No live market headlines available. Configure NEWSAPI_KEY to enable headline feed.")

    st.markdown("---")


def render_live_signals_section(pair: str, timeframe: str, show_title: bool = True) -> None:
    if show_title:
        st.title(f"Live Trading Signals — {pair}")
    else:
        st.subheader("Live Trading Signals")

    st.warning(
        "All signals are probabilistic estimates only. "
        "They do NOT constitute financial advice. "
        "Always apply your own risk management."
    )

    _, scan_col, mode_col, refresh_col = st.columns([2, 1, 1, 1])
    with scan_col:
        scan_bars = st.slider("Scan last N bars", 1, 20, 1)
    with mode_col:
        scan_mode = st.radio("Scan Mode", ["Multi-TF", "Single TF"], horizontal=True)
    with refresh_col:
        auto_refresh = st.toggle("Auto-refresh (60s)", value=True)

    signal_mode = st.radio("Signal Mode", ["High Accuracy", "Balanced"], horizontal=True)
    signal_mode_api = "high_accuracy" if signal_mode == "High Accuracy" else "balanced"

    if auto_refresh:
        st.caption("Auto-refreshing every 60 seconds...")
        import time as _time

    if scan_mode == "Multi-TF":
        with st.spinner("Scanning all timeframes for signals..."):
            sig_data = api_get(
                "/signals/multi-tf",
                {"pair": pair, "scan_bars": scan_bars, "signal_mode": signal_mode_api},
                timeout=90,
            )
    else:
        with st.spinner("Scanning for signals..."):
            sig_data = api_get(
                "/signals/latest",
                {"pair": pair, "timeframe": timeframe, "scan_bars": scan_bars, "signal_mode": signal_mode_api},
                timeout=60,
            )

    if sig_data:
        count = sig_data.get("signal_count", 0)
        overview = sig_data.get("timeframe_overview", []) if scan_mode == "Multi-TF" else []
        st.caption(
            f"Last scan: {_format_local_time(sig_data.get('timestamp'))} {_local_timezone_label()}"
        )

        recommended_setup = sig_data.get("recommended_live_setup", {})
        if recommended_setup:
            st.caption(recommended_setup.get("message", ""))

        if scan_mode == "Multi-TF":
            best_tf = sig_data.get("best_timeframe", {})
            if best_tf:
                st.info(
                    f"Best historical timeframe: {best_tf.get('timeframe', '?')} | "
                    f"WR {best_tf.get('win_rate', 0) * 100:.1f}% | "
                    f"PF {best_tf.get('profit_factor', 0):.2f} | "
                    f"Risk {best_tf.get('risk_level', 'UNKNOWN')}"
                )

        if count == 0:
            st.info(sig_data.get("no_signal_message", "No signals found right now."))
            diagnostics_blocks = []
            if scan_mode == "Multi-TF":
                diagnostics_blocks = [
                    item.get("signal_diagnostics", {})
                    for item in sig_data.get("timeframe_overview", [])
                    if item.get("signal_diagnostics")
                ]
            elif sig_data.get("signal_diagnostics"):
                diagnostics_blocks = [sig_data.get("signal_diagnostics")]

            for diagnostics in diagnostics_blocks:
                timeframe_label = diagnostics.get("timeframe", timeframe).upper()
                st.markdown(f"**Why {timeframe_label} has no live signal**")
                st.caption(diagnostics.get("summary", "No eligible setup on the latest bar."))

                buy = diagnostics.get("buy", {})
                sell = diagnostics.get("sell", {})
                buy_col, sell_col = st.columns(2)
                with buy_col:
                    st.markdown("**BUY check**")
                    st.metric("Confidence", f"{buy.get('confidence', 0):.1f}/100")
                    st.metric("ML Probability", f"{buy.get('model_probability', 0) * 100:.1f}%")
                    for reason in buy.get("blockers", [])[:4]:
                        st.caption(f"- {reason}")
                with sell_col:
                    st.markdown("**SELL check**")
                    st.metric("Confidence", f"{sell.get('confidence', 0):.1f}/100")
                    st.metric("ML Probability", f"{sell.get('model_probability', 0) * 100:.1f}%")
                    for reason in sell.get("blockers", [])[:4]:
                        st.caption(f"- {reason}")
        else:
            st.success(f"Found {count} signal(s)")
            summary_snapshot = api_get("/dashboard/summary", {"pair": pair})
            live_price = None
            if summary_snapshot:
                live_price = summary_snapshot.get("current_price")
            for sig in sig_data.get("signals", []):
                direction = sig.get("direction", "")
                risk_level = sig.get("risk_level", "LOW")
                risk_score = sig.get("risk_score", 0)
                tf = sig.get("timeframe", "?")
                age_bars = int(sig.get("age_bars", 0) or 0)
                header = (
                    f"{direction} [{tf}] @ {_price_format(pair, sig.get('entry_price', 0))} | "
                    f"Confidence: {sig.get('confidence_score', 0):.1f}/100 | "
                    f"Risk: {risk_level} ({risk_score:.0f}/100)"
                )
                with st.expander(header, expanded=(risk_level == "LOW")):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entry", _price_format(pair, sig.get('entry_price', 0)))
                    c2.metric(
                        "Stop Loss",
                        _price_format(pair, sig.get('stop_loss', 0)),
                        delta=_price_delta_label(pair, sig.get('stop_loss', 0) - sig.get('entry_price', 0)),
                    )
                    c3.metric(
                        "Take Profit",
                        _price_format(pair, sig.get('take_profit', 0)),
                        delta=_price_delta_label(pair, sig.get('take_profit', 0) - sig.get('entry_price', 0)),
                    )
                    c4.metric("R:R", f"1:{sig.get('risk_reward', 1):.1f}")

                    c5, c6, c7 = st.columns(3)
                    c5.metric("Win Probability", f"{sig.get('win_probability', 0) * 100:.1f}%")
                    c6.metric("Historical WR", f"{sig.get('historical_win_rate', 0) * 100:.1f}%")
                    c7.metric("ML Probability", f"{sig.get('model_probability', 0) * 100:.1f}%")

                    if live_price is not None:
                        entry_gap = float(live_price) - float(sig.get("entry_price", 0.0))
                        g1, g2 = st.columns(2)
                        g1.metric("Current Price", _price_format(pair, float(live_price)))
                        g2.metric("Entry Gap", _price_delta_label(pair, entry_gap))

                        # === RUNNING SIGNAL STATUS ===
                        running_status = _running_signal_status(
                            current_price=float(live_price),
                            entry_price=float(sig.get("entry_price", 0.0)),
                            stop_loss=float(sig.get("stop_loss", 0.0)),
                            take_profit=float(sig.get("take_profit", 0.0)),
                            direction=direction,
                        )
                        
                        status_color = running_status.get("color", "gray")
                        status_label = running_status.get("label", "UNKNOWN")
                        profit_pct = running_status.get("profit_pct", 0)
                        
                        # Display status header with indicator
                        st.markdown("---")
                        if status_color == "green":
                            st.success(f"✅ **ACTIVE RUNNING SIGNAL**")
                            st.markdown(status_label)
                        elif status_color == "orange":
                            st.warning(f"⚠️ **DANGER ZONE - NEAR STOP LOSS**")
                            st.markdown(status_label)
                        elif status_color == "yellow":
                            st.info(f"⏳ **PENDING SIGNAL**")
                            st.markdown(status_label)
                        elif status_color == "blue":
                            st.info(f"🎯 **TAKE-PROFIT REACHED**")
                            st.markdown(status_label)
                        elif status_color == "red":
                            st.error(f"🛑 **STOP-LOSS HIT - SIGNAL INVALIDATED**")
                            st.markdown(status_label)
                        
                        # === ENTRY/RE-ENTRY GUIDANCE ===
                        guidance = _entry_reentry_guidance(
                            pair_name=pair,
                            direction=direction,
                            current_price=float(live_price),
                            entry_price=float(sig.get("entry_price", 0.0)),
                            stop_loss=float(sig.get("stop_loss", 0.0)),
                            take_profit=float(sig.get("take_profit", 0.0)),
                        )
                        
                        # Display guidance with color coding
                        status = guidance.get("status", "unknown")
                        color_code = guidance.get("color_code", "gray")
                        
                        if color_code == "green":
                            st.success(guidance["message"])
                        elif color_code == "red":
                            st.error(guidance["message"])
                        elif color_code == "yellow":
                            st.warning(guidance["message"])
                        elif color_code == "blue":
                            st.info(guidance["message"])
                        else:
                            st.info(guidance["message"])
                        
                        # Show re-entry zone if available
                        if guidance.get("reentry"):
                            st.markdown(f"**✨ Re-entry Info:**")
                            st.markdown(guidance["reentry"])
                        
                        # Show next action
                        if guidance.get("next_action"):
                            st.caption(f"📍 Next Action: {guidance['next_action']}")
                        
                        # === PRICE ZONE MAP ===
                        entry = float(sig.get("entry_price", 0.0))
                        sl = float(sig.get("stop_loss", 0.0))
                        tp = float(sig.get("take_profit", 0.0))
                        zone_low = guidance.get("reentry_zone_low", entry)
                        zone_high = guidance.get("reentry_zone_high", entry)
                        
                        # Create simple visual zone indicator
                        st.markdown("---")
                        st.markdown("**Price Zone Map:**")
                        zone_display = []
                        
                        if direction == "BUY":
                            if float(live_price) <= sl:
                                zone_display.append(f"🔴 Current: {_price_format(pair, float(live_price))} ← at/below SL {_price_format(pair, sl)}")
                            elif float(live_price) < zone_low:
                                zone_display.append(f"🟡 Current: {_price_format(pair, float(live_price))} ← BELOW Entry Zone")
                            elif zone_low <= float(live_price) <= zone_high:
                                zone_display.append(f"🟢 Current: {_price_format(pair, float(live_price))} ← IN Entry Zone")
                            elif float(live_price) < tp:
                                zone_display.append(f"🟡 Current: {_price_format(pair, float(live_price))} ← ABOVE Entry Zone")
                            else:
                                zone_display.append(f"🔵 Current: {_price_format(pair, float(live_price))} ← at/above TP {_price_format(pair, tp)}")
                        else:  # SELL
                            if float(live_price) >= sl:
                                zone_display.append(f"🔴 Current: {_price_format(pair, float(live_price))} ← at/above SL {_price_format(pair, sl)}")
                            elif float(live_price) > zone_high:
                                zone_display.append(f"🟡 Current: {_price_format(pair, float(live_price))} ← ABOVE Entry Zone")
                            elif zone_low <= float(live_price) <= zone_high:
                                zone_display.append(f"🟢 Current: {_price_format(pair, float(live_price))} ← IN Entry Zone")
                            elif float(live_price) > tp:
                                zone_display.append(f"🟡 Current: {_price_format(pair, float(live_price))} ← BELOW Entry Zone")
                            else:
                                zone_display.append(f"🔵 Current: {_price_format(pair, float(live_price))} ← at/below TP {_price_format(pair, tp)}")
                        
                        zone_display.append(f"🔴 Stop Loss: {_price_format(pair, sl)}")
                        zone_display.append(f"🟢 Entry Zone: {_price_format(pair, zone_low)} - {_price_format(pair, zone_high)}")
                        zone_display.append(f"🔵 Take Profit: {_price_format(pair, tp)}")
                        
                        for line in zone_display:
                            st.caption(line)
                        
                        # === SL HIT: SEARCH FOR NEW SIGNALS ===
                        sl_check = _sl_hit_search_new_signals(
                            pair_name=pair,
                            direction=direction,
                            stop_loss=float(sig.get("stop_loss", 0.0)),
                            current_price=float(live_price),
                        )
                        
                        if sl_check.get("sl_hit"):
                            st.markdown("---")
                            st.error("🛑 STOP-LOSS HIT - SIGNAL STOPPED")
                            st.error(sl_check["search_guidance"])
                            
                            # Show new signals if found
                            new_sigs = sl_check.get("new_signals", [])
                            if new_sigs:
                                st.markdown(f"🔍 **Found {len(new_sigs)} new {direction} signal(s) to consider:**")
                                for i, nsig in enumerate(new_sigs[:3], 1):
                                    st.caption(
                                        f"{i}. {nsig.get('direction')} [{nsig.get('timeframe', '?')}] "
                                        f"@ {_price_format(pair, nsig.get('entry_price', 0))}"
                                    )
                            else:
                                st.info("🔍 No matching new signals found yet. Keep monitoring for future opportunities.")
                        
                        # === TP HIT: SEARCH FOR NEW SIGNALS ===
                        tp_check = _tp_hit_future_search(
                            pair_name=pair,
                            direction=direction,
                            take_profit=float(sig.get("take_profit", 0.0)),
                            current_price=float(live_price),
                        )
                        
                        if tp_check.get("tp_hit"):
                            st.markdown("---")
                            st.success("🎯 TAKE-PROFIT REACHED!")
                            st.markdown(tp_check["search_guidance"])
                            if tp_check.get("future_search_zone_low") is not None:
                                st.caption(
                                    f"💡 Watch for new setup in: "
                                    f"{_price_format(pair, tp_check['future_search_zone_low'])} - "
                                    f"{_price_format(pair, tp_check['future_search_zone_high'])}"
                                )

                        if age_bars > 0:
                            st.warning(
                                f"This signal is from {age_bars} bar(s) ago, so current market price may differ from entry."
                            )
                    else:
                        st.caption("Fresh signal from the latest completed bar.")

                    st.info(sig.get("signal_reason", ""))
                    st.warning(sig.get("invalidation_reason", ""))
                    st.caption(sig.get("volume_disclaimer", ""))

        if overview:
            st.markdown("---")
            st.markdown("**Timeframe Status**")
            columns = st.columns(len(overview))
            for column, item in zip(columns, overview):
                column.markdown(f"**{item.get('timeframe', '?').upper()}**")
                column.metric("Trend", item.get("trend_primary", "unknown").title())
                column.metric("WR", f"{item.get('backtest_win_rate', 0) * 100:.1f}%")
                column.metric("PF", f"{item.get('profit_factor', 0):.2f}")
                if item.get("signal_diagnostics", {}).get("summary"):
                    column.caption(f"Now: {item['signal_diagnostics']['summary']}")

    if auto_refresh:
        _time.sleep(60)
        st.rerun()


render_sidebar_logo()
st.sidebar.markdown("---")

# === COMPACT SIDEBAR ORGANIZATION ===
st.sidebar.subheader("⚙️ Configuration")
pair = st.sidebar.selectbox("📊 Pair", SUPPORTED_PAIRS, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🧭 Navigation")

# Initialize persistent page state once
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

# Create columns for main navigation buttons
nav_cols = st.sidebar.columns(2)
with nav_cols[0]:
    if st.button("🏠 Home", use_container_width=True, key="btn_home"):
        st.session_state.nav_page = "Home"
        st.rerun()
with nav_cols[1]:
    if st.button("📊 Signals", use_container_width=True, key="btn_signals"):
        st.session_state.nav_page = "Live Signals"
        st.rerun()

nav_cols2 = st.sidebar.columns(2)
with nav_cols2[0]:
    if st.button("📉 Backtest", use_container_width=True, key="btn_backtest"):
        st.session_state.nav_page = "Backtest"
        st.rerun()
with nav_cols2[1]:
    if st.button("🤖 Model", use_container_width=True, key="btn_model"):
        st.session_state.nav_page = "Model & Prediction"
        st.rerun()

page = st.session_state.nav_page

st.sidebar.markdown("---")

# === QUICK INFO PANEL ===
st.sidebar.subheader("📌 Quick Info")
st.sidebar.caption(f"Pair: {pair}")
st.sidebar.caption(f"Page: {page}")

try:
    sidebar_api_base = get_api_base()
    st.sidebar.caption(f"API: ✓ {sidebar_api_base.split('/')[-1] or '8000'}")
except:
    st.sidebar.caption("API: ⚠️ unavailable")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Signals are probabilistic estimates only.")

if page == "Home":
    st.markdown(f"# 💼 {pair} Dashboard")
    
    summary = api_get("/dashboard/summary", {"pair": pair})
    if summary:
        # === COMPACT HEADER WITH KEY METRICS ===
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            price = summary.get('current_price', 0)
            price_fmt = _price_format(pair, price)
            st.metric("💰 Price", price_fmt, delta=_price_delta_label(pair, summary.get('live_analysis_delta', 0.0)))
        with col2:
            st.metric("📈 RSI(14)", f"{summary.get('rsi', 0):.1f}", label_visibility="collapsed")
        with col3:
            vol_regime = summary.get("volatility_regime", "—").title()
            st.metric("🌪️ Volatility", vol_regime, label_visibility="collapsed")
        with col4:
            session = summary.get("current_session", "—").replace("_", " ").title()
            st.metric("📍 Session", session, label_visibility="collapsed")
        
        # === RECOMMENDED SETUP ===
        recommended = summary.get("recommended_live_setup", {})
        if recommended:
            st.info(
                f"🎯 **Recommended**: {recommended.get('primary_timeframe', '1d').upper()} | "
                f"{recommended.get('signal_mode', 'high_accuracy').replace('_', ' ').title()} | "
                f"Scan {recommended.get('scan_bars_min', 3)}-{recommended.get('scan_bars_max', 5)} bars"
            )
        
        # === TABS FOR DIFFERENT VIEWS ===
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Market", "⏰ Timing", "🔍 Analysis", "⚠️ Events"])
        
        with tab1:
            st.markdown("**Market Conditions**")
            m1, m2, m3 = st.columns(3)
            m1.caption(f"Trend: **{summary.get('trend_primary', 'unknown').upper()}**")
            m2.caption(f"Liquidity: **{summary.get('liquidity_zone', '—').upper()}**")
            m3.caption(f"Volume: **{summary.get('volume_condition', '—')}**")
            m4, m5 = st.columns(2)
            m4.caption(f"Model: **{'✓ Loaded' if summary.get('model_loaded') else '✗ Not Loaded'}**")
            m5.caption(f"Events: **{'⚠️ Active' if summary.get('event_window') else '✓ Clear'}**")
        
        with tab2:
            st.markdown("**Time Information**")
            t1, t2 = st.columns(2)
            t1.caption(f"📊 Quote: {_format_local_time(summary.get('live_price_time'), '—')} {_local_timezone_label()}")
            t2.caption(f"📈 Analysis: {_format_local_time(summary.get('analysis_time'), '—')} {_local_timezone_label()}")
        
        with tab3:
            st.markdown("**Technical Analysis**")
            a1, a2 = st.columns(2)
            a1.caption(f"Entry Price: {_price_format(pair, summary.get('analysis_price', 0))}")
            a2.caption(f"Live-Analysis Gap: {_price_delta_label(pair, summary.get('live_analysis_delta', 0.0))}")
            a3, a4 = st.columns(2)
            a3.caption(f"Source: {summary.get('live_price_source', 'unknown')}")
            a4.caption(f"Last Refresh: {datetime.now().astimezone().strftime('%I:%M %p')} {_local_timezone_label()}")
        
        with tab4:
            if summary.get("event_window"):
                st.warning("⚠️ High-impact event window ACTIVE — Signals may be suppressed")
            else:
                st.success("✅ No high-impact events in current window")
            st.caption(summary.get("volume_disclaimer", ""))
        
        # === QUICK ACTION BUTTONS ===
        st.markdown("---")
        st.markdown("**Quick Actions**")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        with quick_col1:
            if st.button("📊 View Live Signals", use_container_width=True):
                st.session_state.nav_page = "Live Signals"
                st.rerun()
        with quick_col2:
            if st.button("📉 Run Backtest", use_container_width=True):
                st.session_state.nav_page = "Backtest"
                st.rerun()
        with quick_col3:
            if st.button("🤖 Check Model", use_container_width=True):
                st.session_state.nav_page = "Model & Prediction"
                st.rerun()

elif page == "Live Signals":
    st.markdown(f"# 📊 Live Signals — {pair}")

    # Pair-only flow: auto scan all allowed timeframes
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        scan_bars = st.slider("Scan Bars", 1, 20, 3)
    with filter_col2:
        signal_mode_display = st.radio("Signal Mode", ["High Accuracy", "Balanced"], horizontal=True)
    with filter_col3:
        auto_refresh = st.toggle("Auto-Refresh (60s)", value=True)

    all_pairs_3type = st.button("🌐 Analyze ALL Pairs + ALL TF (3-Type)", use_container_width=True)

    st.caption("Auto mode: all timeframes scanned together. Only pair নির্বাচন করলেই হবে.")

    signal_view = st.radio(
        "View",
        ["All-in-One", "Signal Feed", "Timeframe Status", "Diagnostics"],
        horizontal=True,
    )

    signal_mode_api = "high_accuracy" if signal_mode_display == "High Accuracy" else "balanced"
    
    if auto_refresh:
        st.caption("🔄 Auto-refreshing every 60 seconds...")
        import time as _time
    
    # === SIGNAL SCAN (always multi-timeframe) ===
    with st.spinner("Scanning all timeframes..."):
        sig_data = api_get(
            "/signals/multi-tf",
            {"pair": pair, "scan_bars": scan_bars, "signal_mode": signal_mode_api},
            timeout=90,
        )

    if sig_data:
        allowed_tfs = _allowed_timeframes_for_pair(pair)

        raw_signals = sig_data.get("signals", []) or []
        signals = [
            s for s in raw_signals
            if str(s.get("timeframe", "")).lower() in allowed_tfs
        ]

        raw_overview = sig_data.get("timeframe_overview", []) or []
        overview_lookup = {
            str(item.get("timeframe", "")).lower(): item
            for item in raw_overview
        }
        overview = [
            overview_lookup.get(tf, {"timeframe": tf, "signal_diagnostics": {}})
            for tf in allowed_tfs
        ]

        count = len(signals)

        summary_snapshot = api_get("/dashboard/summary", {"pair": pair})
        live_price_for_summary = summary_snapshot.get("current_price") if summary_snapshot else None
        three_type_buckets = _build_three_type_buckets(
            pair_name=pair,
            sig_data=sig_data,
            live_price=float(live_price_for_summary) if live_price_for_summary is not None else None,
        )

        # Quick status summary for friendly navigation
        tf_running = len(three_type_buckets.get("running", []))
        tf_upcoming = len(three_type_buckets.get("upcoming", []))
        tf_passed = len(three_type_buckets.get("passed", []))

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Active Signals", count)
        q2.metric("Running TF", tf_running)
        q3.metric("Upcoming TF", tf_upcoming)
        q4.metric("Passed TF", tf_passed)
        
        st.caption(f"Scan: {_format_local_time(sig_data.get('timestamp'))} {_local_timezone_label()}")

        recommended_setup = sig_data.get("recommended_live_setup", {})
        if recommended_setup:
            st.caption(recommended_setup.get("message", ""))

        best_tf = sig_data.get("best_timeframe", {})
        if best_tf:
            st.info(
                f"🏆 {best_tf.get('timeframe', '?').upper()} | "
                f"WR {best_tf.get('win_rate', 0) * 100:.1f}% | "
                f"PF {best_tf.get('profit_factor', 0):.2f}"
            )

        if signal_view in ("All-in-One", "Signal Feed"):
            _render_three_type_sections(pair, three_type_buckets)

        if count == 0:
            st.info(sig_data.get("no_signal_message", "No signals found."))
            diagnostics_blocks = [
                item.get("signal_diagnostics", {})
                for item in overview
                if item.get("signal_diagnostics")
            ]

            if signal_view in ("All-in-One", "Diagnostics"):
                for diagnostics in diagnostics_blocks:
                    timeframe_label = diagnostics.get("timeframe", "?").upper()
                    with st.expander(f"Why no {timeframe_label}?"):
                        st.caption(diagnostics.get("summary", "No setup."))
                        buy = diagnostics.get("buy", {})
                        sell = diagnostics.get("sell", {})

                        buy_conf = float(buy.get("confidence", 0) or 0)
                        sell_conf = float(sell.get("confidence", 0) or 0)
                        buy_prob = float(buy.get("model_probability", 0) or 0)
                        sell_prob = float(sell.get("model_probability", 0) or 0)
                        best_side = "BUY" if (buy_conf + buy_prob * 100) >= (sell_conf + sell_prob * 100) else "SELL"
                        best_conf = buy_conf if best_side == "BUY" else sell_conf
                        best_prob = buy_prob if best_side == "BUY" else sell_prob
                        if best_conf >= 50 or best_prob >= 0.50:
                            upcoming_time = _format_local_time(sig_data.get("timestamp"))
                            st.info(
                                f"🟣 {_status_label_with_time('UPCOMING', upcoming_time)} | "
                                f"Likely {best_side} setup (Conf {best_conf:.1f}/100, ML {best_prob * 100:.1f}%)"
                            )

                        b1, s1 = st.columns(2)
                        with b1:
                            st.caption("**BUY**")
                            st.caption(f"Conf: {buy.get('confidence', 0):.1f}/100")
                            st.caption(f"ML: {buy.get('model_probability', 0) * 100:.1f}%")
                        with s1:
                            st.caption("**SELL**")
                            st.caption(f"Conf: {sell.get('confidence', 0):.1f}/100")
                            st.caption(f"ML: {sell.get('model_probability', 0) * 100:.1f}%")
        else:
            st.success(f"✓ Found {count} signal(s)")
            live_price = live_price_for_summary

            if signal_view in ("All-in-One", "Signal Feed"):
                for sig in signals:
                    direction = sig.get("direction", "")
                    risk_level = sig.get("risk_level", "LOW")
                    risk_score = sig.get("risk_score", 0)
                    tf = sig.get("timeframe", "?")
                    age_bars = int(sig.get("age_bars", 0) or 0)

                    signal_time_text = _resolve_signal_time(sig, sig_data.get("timestamp"))
                    signal_state_suffix = ""
                    precomputed_running_status = None
                    if live_price is not None:
                        precomputed_running_status = _running_signal_status(
                            current_price=float(live_price),
                            entry_price=float(sig.get("entry_price", 0.0)),
                            stop_loss=float(sig.get("stop_loss", 0.0)),
                            take_profit=float(sig.get("take_profit", 0.0)),
                            direction=direction,
                        )
                        if precomputed_running_status.get("status") in {"stopped", "completed"}:
                            signal_state_suffix = f" | EXPIRED ({signal_time_text})"
                        else:
                            signal_state_suffix = f" | RUNNING ({signal_time_text})"
                    else:
                        signal_state_suffix = f" | UPCOMING ({signal_time_text})"
                    
                    # Compact header
                    header = (
                        f"{direction} [{tf}] | {_price_format(pair, sig.get('entry_price', 0))} | "
                        f"Conf: {sig.get('confidence_score', 0):.0f}/100 | Risk: {risk_level}{signal_state_suffix}"
                    )
                    with st.expander(header, expanded=(risk_level == "LOW")):
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Entry", _price_format(pair, sig.get('entry_price', 0)))
                        mc2.metric("Stop Loss", _price_format(pair, sig.get('stop_loss', 0)))
                        mc3.metric("Take Profit", _price_format(pair, sig.get('take_profit', 0)))
                        mc4.metric("R:R", f"1:{sig.get('risk_reward', 1):.1f}")

                        mp1, mp2, mp3 = st.columns(3)
                        mp1.metric("Win %", f"{sig.get('win_probability', 0) * 100:.1f}%")
                        mp2.metric("Historical WR", f"{sig.get('historical_win_rate', 0) * 100:.1f}%")
                        mp3.metric("ML Probability", f"{sig.get('model_probability', 0) * 100:.1f}%")

                        if live_price is not None:
                            entry_gap = float(live_price) - float(sig.get("entry_price", 0.0))
                            pg1, pg2 = st.columns(2)
                            pg1.metric("Live Price", _price_format(pair, float(live_price)))
                            pg2.metric("Entry Gap", _price_delta_label(pair, entry_gap))

                            running_status = precomputed_running_status or _running_signal_status(
                                current_price=float(live_price),
                                entry_price=float(sig.get("entry_price", 0.0)),
                                stop_loss=float(sig.get("stop_loss", 0.0)),
                                take_profit=float(sig.get("take_profit", 0.0)),
                                direction=direction,
                            )
                            status_label = running_status.get("label", "UNKNOWN")
                            if running_status.get("status") in {"stopped", "completed"}:
                                st.error(f"🔴 {status_label} | {_status_label_with_time('PASSED', signal_time_text)}")
                            else:
                                st.success(f"🟢 {status_label} | {_status_label_with_time('RUNNING', signal_time_text)}")
                        else:
                            st.info(f"🟣 {_status_label_with_time('UPCOMING', signal_time_text)}")

                        if age_bars > 0:
                            st.caption(f"Signal age: {age_bars} bar(s)")
                        st.caption(sig.get("signal_reason", ""))

        # === ALL TIMEFRAMES: RUNNING / UPCOMING / PASSED ===
        if overview and signal_view in ("All-in-One", "Timeframe Status"):
            st.markdown("---")
            st.markdown("**All Timeframes Status (Auto)**")
            summary_snapshot = api_get("/dashboard/summary", {"pair": pair})
            live_price = summary_snapshot.get("current_price") if summary_snapshot else None
            signals = signals or []

            for item in overview:
                tf_name = (item.get("timeframe", "?") or "?").upper()
                tf_signal = next((s for s in signals if str(s.get("timeframe", "")).upper() == tf_name), None)

                if tf_signal:
                    lifecycle, lifecycle_text = _signal_lifecycle_status_with_time(
                        tf_signal,
                        float(live_price) if live_price is not None else None,
                        sig_data.get("timestamp"),
                    )
                    display_status = "PASSED" if lifecycle == "EXPIRED" else lifecycle
                    if display_status == "PASSED":
                        status_line = lifecycle_text.replace("EXPIRED", "PASSED")
                        st.caption(f"{tf_name}: 🔴 {status_line}")
                    elif display_status == "RUNNING":
                        st.caption(f"{tf_name}: 🟢 {lifecycle_text}")
                    else:
                        st.caption(f"{tf_name}: 🟣 {lifecycle_text}")
                else:
                    diagnostics = item.get("signal_diagnostics", {}) or {}
                    buy = diagnostics.get("buy", {})
                    sell = diagnostics.get("sell", {})
                    buy_conf = float(buy.get("confidence", 0) or 0)
                    sell_conf = float(sell.get("confidence", 0) or 0)
                    buy_prob = float(buy.get("model_probability", 0) or 0)
                    sell_prob = float(sell.get("model_probability", 0) or 0)
                    upcoming_hint = "likely" if max(buy_conf, sell_conf) >= 50 or max(buy_prob, sell_prob) >= 0.50 else "watch"
                    up_time = _format_local_time(sig_data.get("timestamp"))
                    st.caption(f"{tf_name}: 🟣 UPCOMING | {up_time} {_local_timezone_label()} ({upcoming_hint})")

        # === TIMEFRAME OVERVIEW (kept) ===
        if overview and signal_view in ("All-in-One", "Timeframe Status"):
            st.markdown("---")
            st.markdown("**Timeframe Overview**")
            for item in overview:
                tf_name = item.get('timeframe', '?').upper()
                trend = item.get("trend_primary", "unknown").title()
                wr = f"{item.get('backtest_win_rate', 0) * 100:.1f}%"
                pf = f"{item.get('profit_factor', 0):.2f}"
                st.caption(f"{tf_name}: {trend} | WR {wr} | PF {pf}")

    if all_pairs_3type:
        st.markdown("---")
        st.markdown("## ALL Pairs + ALL Timeframes (3-Type Snapshot)")
        all_rows = []
        with st.spinner("Running full market snapshot..."):
            for p in SUPPORTED_PAIRS:
                pair_sig = api_get(
                    "/signals/multi-tf",
                    {"pair": p, "scan_bars": scan_bars, "signal_mode": signal_mode_api},
                    timeout=90,
                )
                pair_summary = api_get("/dashboard/summary", {"pair": p})
                pair_live = pair_summary.get("current_price") if pair_summary else None
                if not pair_sig:
                    all_rows.append(
                        {
                            "pair": p,
                            "running": 0,
                            "upcoming": 0,
                            "passed": 0,
                            "note": "scan failed",
                        }
                    )
                    continue

                pair_buckets = _build_three_type_buckets(
                    pair_name=p,
                    sig_data=pair_sig,
                    live_price=float(pair_live) if pair_live is not None else None,
                )
                all_rows.append(
                    {
                        "pair": p,
                        "running": len(pair_buckets.get("running", [])),
                        "upcoming": len(pair_buckets.get("upcoming", [])),
                        "passed": len(pair_buckets.get("passed", [])),
                        "note": "ok",
                    }
                )

                with st.expander(f"{p} | Running {len(pair_buckets.get('running', []))} | Upcoming {len(pair_buckets.get('upcoming', []))} | Passed {len(pair_buckets.get('passed', []))}"):
                    _render_three_type_sections(p, pair_buckets)

        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

    if auto_refresh:
        _time.sleep(60)
        st.rerun()

elif page == "Backtest":
    st.markdown(f"# 📉 Backtest — {pair}")

    col_config, col_run, col_run_all = st.columns([4, 1, 1])
    with col_config:
        subset_col, tf_col, rr_col, force_col = st.columns(4)
        with subset_col:
            strategy = st.text_input("Strategy", "DioMultiCondition_v1", label_visibility="collapsed")
        with tf_col:
            allowed_tf = _allowed_timeframes_for_pair(pair)
            backtest_timeframe = st.selectbox("Timeframe", allowed_tf, index=0)
        with rr_col:
            rr_ratio = st.selectbox("Risk:Reward", [1.0, 1.5, 2.0, 2.5, 3.0], index=0, label_visibility="collapsed")
        with force_col:
            force = st.checkbox("Refresh Data", label_visibility="collapsed")
    
    with col_run:
        run_backtest = st.button("▶️ Run Backtest", use_container_width=True, type="primary")

    with col_run_all:
        run_backtest_all = st.button("🚀 Run ALL", use_container_width=True)
    
    if run_backtest:
        with st.spinner("Running backtest..."):
            result = api_post(
                "/backtest/run",
                {
                    "pair": pair,
                    "timeframe": backtest_timeframe,
                    "strategy_name": strategy,
                    "rr": rr_ratio,
                    "force_refresh": force,
                    "recent_signals": 20,
                },
            )

        if result:
            st.success("✓ Backtest completed!")
            
            # === MAIN METRICS (compact 4-column layout) ===
            st.markdown("**Overall Performance**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Signals", result.get("total_signals", 0))
            m2.metric("Total Win Rate", f"{result.get('total_win_rate', 0) * 100:.1f}%")
            m3.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
            m4.metric("Max Drawdown", f"{result.get('max_drawdown_pct', 0):.1f}%")

            # === DIRECTION METRICS ===
            buy_count = int(result.get("buy_signals", 0) or 0)
            sell_count = int(result.get("sell_signals", 0) or 0)
            buy_wr = f"{result.get('buy_win_rate', 0) * 100:.1f}%" if buy_count > 0 else "N/A"
            sell_wr = f"{result.get('sell_win_rate', 0) * 100:.1f}%" if sell_count > 0 else "N/A"
            
            st.markdown("**Direction Breakdown (BUY/SELL)**")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Buy Win Rate", buy_wr)
            d2.metric("Sell Win Rate", sell_wr)
            d3.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.3f}")
            d4.metric("Total Return", f"{result.get('total_return_pct', 0):.1f}%")

            st.caption(
                f"BUY Signals: {buy_count} | SELL Signals: {sell_count}"
            )

            st.caption(f"Mode: {result.get('direction_bias', 'BOTH')} | BUY: {buy_count} | SELL: {sell_count}")
            
            if result.get("analysis_scope") == "recent_signals":
                st.caption(f"Scope: Last {result.get('recent_signals_used', 0)}/{result.get('recent_signals_requested', 20)} signals")
            if result.get("fallback_from_recent_scope"):
                st.info("Recent window-এ closed trade না থাকায় full-history metrics দেখানো হয়েছে (0.0 avoid করতে)।")

            # === EQUITY CURVE ===
            equity = result.get("equity_curve", [])
            if equity:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=equity, mode="lines", name="Equity", line=dict(color="#00d26a")))
                fig.update_layout(template="plotly_dark", height=250, margin=dict(l=30, r=30, t=30, b=30), title="Equity Curve")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📝 Details"):
                st.caption(result.get("disclaimer", ""))
                st.caption(result.get("volume_disclaimer", ""))

    if run_backtest_all:
        all_rows = []
        total_jobs = sum(len(_allowed_timeframes_for_pair(p)) for p in SUPPORTED_PAIRS)
        prog = st.progress(0.0)
        done = 0

        with st.spinner("Running all pair + timeframe backtests..."):
            for p in SUPPORTED_PAIRS:
                for tf in _allowed_timeframes_for_pair(p):
                    res = api_post(
                        "/backtest/run",
                        {
                            "pair": p,
                            "timeframe": tf,
                            "strategy_name": strategy,
                            "rr": rr_ratio,
                            "force_refresh": force,
                            "recent_signals": 20,
                        },
                    )
                    done += 1
                    prog.progress(done / total_jobs)

                    if res:
                        all_rows.append(
                            {
                                "pair": p,
                                "timeframe": tf,
                                "signals": res.get("total_signals", 0),
                                "win_rate_%": round(float(res.get("total_win_rate", 0.0) or 0.0) * 100, 1),
                                "buy_wr_%": round(float(res.get("buy_win_rate", 0.0) or 0.0) * 100, 1),
                                "sell_wr_%": round(float(res.get("sell_win_rate", 0.0) or 0.0) * 100, 1),
                                "profit_factor": round(float(res.get("profit_factor", 0.0) or 0.0), 3),
                                "max_dd_%": round(float(res.get("max_drawdown_pct", 0.0) or 0.0), 2),
                                "return_%": round(float(res.get("total_return_pct", 0.0) or 0.0), 2),
                                "scope": res.get("analysis_scope", "unknown"),
                            }
                        )
                    else:
                        all_rows.append(
                            {
                                "pair": p,
                                "timeframe": tf,
                                "signals": None,
                                "win_rate_%": None,
                                "buy_wr_%": None,
                                "sell_wr_%": None,
                                "profit_factor": None,
                                "max_dd_%": None,
                                "return_%": None,
                                "scope": "failed",
                            }
                        )

        st.session_state["all_backtest_results"] = all_rows
        st.success("All pair + timeframe backtest completed.")

    if st.session_state.get("all_backtest_results"):
        st.markdown("**All Backtest Results (Separate)**")
        all_bt_df = pd.DataFrame(st.session_state["all_backtest_results"])
        st.dataframe(all_bt_df, use_container_width=True, hide_index=True)

elif page == "Model & Prediction":
    st.markdown(f"# 🤖 ML Model — {pair}")
    st.caption("Single pair controls নিচে আছে, সাথে ALL pairs + ALL timeframes একসাথে train/predict করা যাবে।")

    model_allowed_tf = _allowed_timeframes_for_pair(pair)
    default_model_index = 2 if len(model_allowed_tf) > 2 else 0
    model_tf = st.selectbox("Model Timeframe", model_allowed_tf, index=default_model_index)

    # === ALL PAIRS + ALL TIMEFRAMES ===
    all_train_col, all_pred_col = st.columns(2)
    with all_train_col:
        if st.button("🚀 Train ALL Pairs + ALL TF", type="primary", use_container_width=True):
            rows = []
            total = sum(len(_allowed_timeframes_for_pair(p)) for p in SUPPORTED_PAIRS)
            progress = st.progress(0.0)
            done = 0
            for p in SUPPORTED_PAIRS:
                for tf in _allowed_timeframes_for_pair(p):
                    result = api_post("/model/train", {"pair": p, "timeframe": tf})
                    done += 1
                    progress.progress(done / total)
                    if result:
                        rows.append(
                            {
                                "pair": p,
                                "timeframe": tf,
                                "status": "trained",
                                "test_auc": round(float(result.get("test_auc", 0.0) or 0.0), 4),
                                "cv_auc": round(float(result.get("cv_mean_auc", 0.0) or 0.0), 4),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "pair": p,
                                "timeframe": tf,
                                "status": "failed",
                                "test_auc": None,
                                "cv_auc": None,
                            }
                        )
            st.session_state["all_train_results"] = rows
            st.success("All pair/timeframe training run completed.")

    with all_pred_col:
        if st.button("📊 Show Probabilities: ALL Pairs + ALL TF", use_container_width=True):
            rows = []
            total = sum(len(_allowed_timeframes_for_pair(p)) for p in SUPPORTED_PAIRS)
            progress = st.progress(0.0)
            done = 0
            for p in SUPPORTED_PAIRS:
                for tf in _allowed_timeframes_for_pair(p):
                    pred = api_get("/model/predict", {"pair": p, "timeframe": tf})
                    done += 1
                    progress.progress(done / total)
                    if pred:
                        rows.append(
                            {
                                "pair": p,
                                "timeframe": tf,
                                "bull_%": round(float(pred.get("bull_probability", 0.0) or 0.0) * 100, 1),
                                "bear_%": round(float(pred.get("bear_probability", 0.0) or 0.0) * 100, 1),
                                "lean": pred.get("direction_lean", "NEUTRAL"),
                                "confidence": pred.get("confidence_label", "low"),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "pair": p,
                                "timeframe": tf,
                                "bull_%": None,
                                "bear_%": None,
                                "lean": "N/A",
                                "confidence": "N/A",
                            }
                        )

            st.session_state["all_predict_results"] = rows
            st.success("All pair/timeframe probability scan completed.")

    if st.session_state.get("all_train_results"):
        st.markdown("**All Training Results**")
        st.dataframe(pd.DataFrame(st.session_state["all_train_results"]), use_container_width=True, hide_index=True)

    if st.session_state.get("all_predict_results"):
        st.markdown("**All Probabilities (Live)**")
        st.dataframe(pd.DataFrame(st.session_state["all_predict_results"]), use_container_width=True, hide_index=True)

        # Quick grouped view by pair
        pred_df = pd.DataFrame(st.session_state["all_predict_results"])
        if not pred_df.empty and "bull_%" in pred_df.columns:
            st.markdown("**Average Bull Probability by Pair**")
            pair_avg = (
                pred_df.dropna(subset=["bull_%"])
                .groupby("pair", as_index=False)["bull_%"]
                .mean()
                .rename(columns={"bull_%": "avg_bull_%"})
            )
            if not pair_avg.empty:
                st.dataframe(pair_avg, use_container_width=True, hide_index=True)
    
    # === TWO-COLUMN LAYOUT ===
    col_train, col_predict = st.columns(2)
    
    with col_train:
        st.subheader("🔧 Training")
        if st.button("Train Model (5yr)", type="primary", use_container_width=True):
            with st.spinner("Training (~2-5 min)..."):
                result = api_post("/model/train", {"pair": pair, "timeframe": model_tf})
            if result:
                st.success("✓ Model trained!")
                t1, t2 = st.columns(2)
                t1.metric("Test AUC", f"{result.get('test_auc', 0):.4f}", label_visibility="collapsed")
                t2.metric("CV AUC", f"{result.get('cv_mean_auc', 0):.4f}", label_visibility="collapsed")
                
                fi = result.get("feature_importance_top10", {})
                if fi:
                    df_fi = pd.DataFrame(fi.items(), columns=["Feature", "Importance"])
                    fig = px.bar(
                        df_fi.sort_values("Importance"),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        template="plotly_dark",
                        height=300,
                    )
                    fig.update_layout(margin=dict(l=100, r=30, t=30, b=30))
                    st.plotly_chart(fig, use_container_width=True)

    with col_predict:
        st.subheader("🎯 Live Prediction")
        if st.button("Get Current Probability", use_container_width=True):
            with st.spinner("Computing..."):
                pred = api_get("/model/predict", {"pair": pair, "timeframe": model_tf})
            if pred:
                bull_prob = pred.get('bull_probability', 0)
                bear_prob = pred.get('bear_probability', 0)
                
                p1, p2 = st.columns(2)
                p1.metric("Bull %", f"{bull_prob * 100:.1f}%", label_visibility="collapsed")
                p2.metric("Bear %", f"{bear_prob * 100:.1f}%", label_visibility="collapsed")
                
                direction_lean = pred.get('direction_lean', 'NEUTRAL')
                confidence_label = pred.get('confidence_label', 'low')
                
                lean_emoji = "📈" if direction_lean == "BULLISH" else "📉" if direction_lean == "BEARISH" else "➡️"
                st.markdown(f"**{lean_emoji} Lean:** {direction_lean} ({confidence_label})")
                
                st.progress(bull_prob)
                
                with st.expander("📝 Disclaimer"):
                    st.caption(pred.get("disclaimer", ""))
