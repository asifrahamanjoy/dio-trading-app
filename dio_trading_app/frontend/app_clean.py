"""
Dio Trading App — Streamlit Dashboard
=======================================
Clean, compact dashboard for multi-market analysis, live signals,
backtesting, and ML probability checks.

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


# ─── Constants ────────────────────────────────────────────────────────────────

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


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dio Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>.stMetric{padding:4px 0!important}</style>",
    unsafe_allow_html=True,
)


# ─── Utility Helpers ──────────────────────────────────────────────────────────

def _normalise_params(params: dict | None) -> tuple[tuple[str, str], ...]:
    if not params:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in params.items()))


def _tz_label() -> str:
    return datetime.now().astimezone().tzname() or "Local"


def _fmt_time(ts: str | None, fallback: str = "—") -> str:
    if not ts:
        return fallback
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return fallback


def _pfmt(pair: str, price: float) -> str:
    if pair == "XAU/USD":
        return f"{price:.2f}"
    if "JPY" in pair:
        return f"{price:.3f}"
    return f"{price:.5f}"


def _pip_label(pair: str, delta: float) -> str:
    if pair == "XAU/USD":
        return f"{delta:+.2f} USD"
    scale = 100 if "JPY" in pair else 10000
    return f"{delta * scale:+.1f} pips"


def _candidate_api_bases(base: str) -> list[str]:
    out = [base]
    for port in API_DISCOVERY_PORTS:
        c = f"http://localhost:{port}"
        if c not in out:
            out.append(c)
    return out


# ─── Signal Status Logic ─────────────────────────────────────────────────────

def _running_signal_status(
    current_price: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
) -> dict:
    risk = max(abs(entry_price - stop_loss), 1e-9)

    if direction == "BUY":
        if current_price <= stop_loss:
            return {"status": "stopped", "color": "red", "label": "🔴 STOPPED", "profit_pct": -abs(current_price - entry_price) / risk}
        pct = (current_price - entry_price) / risk
        if current_price >= take_profit:
            return {"status": "completed", "color": "blue", "label": "🔵 TP HIT", "profit_pct": 1.0}
        if current_price >= entry_price:
            return {"status": "active", "color": "green", "label": f"🟢 +{pct * 100:.1f}R", "profit_pct": pct}
        d = (entry_price - current_price) / risk
        if d > 0.7:
            return {"status": "danger_zone", "color": "orange", "label": f"🟠 DANGER -{d * 100:.1f}R", "profit_pct": -d}
        return {"status": "pending", "color": "yellow", "label": f"🟡 -{d * 100:.1f}R", "profit_pct": -d}

    # SELL
    if current_price >= stop_loss:
        return {"status": "stopped", "color": "red", "label": "🔴 STOPPED", "profit_pct": -abs(current_price - entry_price) / risk}
    pct = (entry_price - current_price) / risk
    if current_price <= take_profit:
        return {"status": "completed", "color": "blue", "label": "🔵 TP HIT", "profit_pct": 1.0}
    if current_price <= entry_price:
        return {"status": "active", "color": "green", "label": f"🟢 +{pct * 100:.1f}R", "profit_pct": pct}
    d = (current_price - entry_price) / risk
    if d > 0.7:
        return {"status": "danger_zone", "color": "orange", "label": f"🟠 DANGER -{d * 100:.1f}R", "profit_pct": -d}
    return {"status": "pending", "color": "yellow", "label": f"🟡 -{d * 100:.1f}R", "profit_pct": -d}


def _resolve_signal_time(sig: dict, scan_ts: str | None = None) -> str:
    for key in ("timestamp", "signal_time", "created_at", "generated_at", "bar_time", "analysis_time"):
        v = sig.get(key)
        if v:
            return _fmt_time(str(v))
    if scan_ts:
        return _fmt_time(scan_ts)
    return datetime.now().astimezone().strftime("%Y-%m-%d %I:%M %p")


def _signal_outcome(price: float, sig: dict) -> str:
    d = str(sig.get("direction", "")).upper()
    entry = float(sig.get("entry_price", 0) or 0)
    sl = float(sig.get("stop_loss", 0) or 0)
    tp = float(sig.get("take_profit", 0) or 0)
    if d not in ("BUY", "SELL") or entry == 0 or sl == 0 or tp == 0:
        return "unknown"
    r = _running_signal_status(current_price=price, entry_price=entry, stop_loss=sl, take_profit=tp, direction=d)
    if r["status"] == "completed":
        return "tp_hit"
    if r["status"] == "stopped":
        return "sl_hit"
    return "running"


# ─── 3-Type Signal System ────────────────────────────────────────────────────

def _signal_explanation(sig: dict, ov: dict | None = None) -> str:
    """One-line compact explanation with all key analysis metrics."""
    direction = str(sig.get("direction", "?")).upper()
    conf = float(sig.get("confidence_score", 0) or 0)
    ml = float(sig.get("model_probability", 0) or 0) * 100
    win = float(sig.get("win_probability", 0) or 0) * 100
    hist = float(sig.get("historical_win_rate", 0) or 0) * 100
    liq = str(ov.get("liquidity_zone", sig.get("liquidity_zone", "—"))) if ov else str(sig.get("liquidity_zone", "—"))
    vol = str(sig.get("market_condition", "—"))
    trend = str(ov.get("trend_primary", sig.get("trend_direction", "—"))) if ov else str(sig.get("trend_direction", "—"))
    factors = sig.get("contributing_factors") or []
    trigger = next(
        (str(f) for f in factors if any(k in str(f).lower() for k in ("macd", "sweep", "bollinger", "ema", "rsi", "sma"))),
        "—",
    )
    return (
        f"{direction} | Conf {conf:.0f} | ML {ml:.0f}% | Win {win:.0f}% | Hist {hist:.0f}% | "
        f"Liq {liq} | Vol {vol} | Trend {trend} | {trigger}"
    )


def _build_buckets(pair_name: str, sig_data: dict, live_price: float | None) -> dict:
    """Classify signals into running / upcoming / passed buckets."""
    allowed = _allowed_timeframes_for_pair(pair_name)
    raw_sigs = sig_data.get("signals") or []
    sigs = [s for s in raw_sigs if str(s.get("timeframe", "")).lower() in allowed]
    raw_ov = sig_data.get("timeframe_overview") or []
    ov_map = {str(o.get("timeframe", "")).lower(): o for o in raw_ov}
    buckets: dict = {"running": [], "upcoming": [], "passed": []}
    scan_ts = sig_data.get("timestamp")

    for tf in allowed:
        tf_up = tf.upper()
        ov = ov_map.get(tf, {"timeframe": tf})
        sig = next((s for s in sigs if str(s.get("timeframe", "")).lower() == tf), None)

        if sig:
            started = _resolve_signal_time(sig, scan_ts)
            if live_price is None:
                buckets["upcoming"].append({
                    "pair": pair_name, "tf": tf_up, "time": started,
                    "status": "UPCOMING", "explanation": _signal_explanation(sig, ov), "signal": sig,
                })
                continue
            outcome = _signal_outcome(float(live_price), sig)
            if outcome == "running":
                buckets["running"].append({
                    "pair": pair_name, "tf": tf_up, "time": started,
                    "status": "RUNNING", "explanation": _signal_explanation(sig, ov), "signal": sig,
                })
            else:
                tag = "TP HIT ✅" if outcome == "tp_hit" else "SL HIT ❌" if outcome == "sl_hit" else "CLOSED"
                buckets["passed"].append({
                    "pair": pair_name, "tf": tf_up, "time": started,
                    "status": tag, "explanation": _signal_explanation(sig, ov), "signal": sig,
                })
        else:
            diag = ov.get("signal_diagnostics") or {}
            buy = diag.get("buy") or {}
            sell = diag.get("sell") or {}
            b_score = float(buy.get("confidence", 0) or 0) + float(buy.get("model_probability", 0) or 0) * 100
            s_score = float(sell.get("confidence", 0) or 0) + float(sell.get("model_probability", 0) or 0) * 100
            side = "BUY" if b_score >= s_score else "SELL"
            side_conf = float(buy.get("confidence", 0) or 0) if side == "BUY" else float(sell.get("confidence", 0) or 0)
            side_ml = float(buy.get("model_probability", 0) or 0) if side == "BUY" else float(sell.get("model_probability", 0) or 0)
            hint = "likely" if side_conf >= 50 or side_ml >= 0.50 else "watch"
            buckets["upcoming"].append({
                "pair": pair_name, "tf": tf_up, "time": _fmt_time(scan_ts),
                "status": f"UPCOMING ({hint})",
                "explanation": f"{side} সম্ভাবনা | Conf {side_conf:.0f} | ML {side_ml * 100:.0f}% | Liq {ov.get('liquidity_zone', '—')}",
                "signal": None,
            })

    # Smart-rank running signals (best entry first)
    def _rank(item: dict) -> float:
        s = item.get("signal") or {}
        return (
            float(s.get("confidence_score", 0) or 0) * 0.30
            + float(s.get("model_probability", 0) or 0) * 25
            + float(s.get("win_probability", 0) or 0) * 25
            + float(s.get("historical_win_rate", 0) or 0) * 10
            - float(s.get("risk_score", 50) or 50) * 0.10
        )

    buckets["running"].sort(key=_rank, reverse=True)
    for i, item in enumerate(buckets["running"], 1):
        item["rank"] = i
        item["rank_score"] = round(_rank(item), 1)

    return buckets


def _render_signal_card(pair_name: str, item: dict, expanded: bool = False) -> None:
    """Render an expandable signal card with entry/SL/TP and analysis."""
    sig = item.get("signal") or {}
    direction = str(sig.get("direction", "")).upper()
    entry_p = float(sig.get("entry_price", 0) or 0)
    sl_p = float(sig.get("stop_loss", 0) or 0)
    tp_p = float(sig.get("take_profit", 0) or 0)

    rank = item.get("rank")
    if rank:
        badge = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        header = f"{badge} {direction} [{item['tf']}] @ {_pfmt(pair_name, entry_p)} | Score {item.get('rank_score', 0)} | {item.get('time', '—')}"
    else:
        header = f"{item.get('status', '')} | {direction} [{item['tf']}] @ {_pfmt(pair_name, entry_p)} | {item.get('time', '—')}"

    with st.expander(header, expanded=expanded):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", _pfmt(pair_name, entry_p))
        c2.metric("SL", _pfmt(pair_name, sl_p))
        c3.metric("TP", _pfmt(pair_name, tp_p))
        c4.metric("R:R", f"1:{float(sig.get('risk_reward', 2) or 2):.1f}")

        p1, p2, p3 = st.columns(3)
        p1.metric("Win %", f"{float(sig.get('win_probability', 0) or 0) * 100:.1f}%")
        p2.metric("Hist WR", f"{float(sig.get('historical_win_rate', 0) or 0) * 100:.1f}%")
        p3.metric("ML Prob", f"{float(sig.get('model_probability', 0) or 0) * 100:.1f}%")

        st.caption(item.get("explanation", ""))
        reason = sig.get("signal_reason", "")
        if reason:
            st.caption(f"💡 {reason[:250]}")


def _render_three_types(pair_name: str, buckets: dict) -> None:
    """Render Running → Upcoming → Passed sections."""

    # ── 1) Running ────────────────────────────────────────────
    st.markdown("### 🟢 Running Signals — Entry possible now")
    if not buckets.get("running"):
        st.caption("কোনো active signal নেই এখন।")
    for item in buckets.get("running", []):
        _render_signal_card(pair_name, item, expanded=(item.get("rank") == 1))

    # ── 2) Upcoming ───────────────────────────────────────────
    st.markdown("### 🟣 Upcoming — Get ready")
    if not buckets.get("upcoming"):
        st.caption("সব timeframe মনিটর হচ্ছে।")
    for item in buckets.get("upcoming", []):
        st.info(f"🟣 **{item['tf']}** | {item['status']} | {item.get('time', '—')}")
        st.caption(item.get("explanation", ""))

    # ── 3) Passed ─────────────────────────────────────────────
    st.markdown("### 🔴 Passed — TP/SL reached")
    if not buckets.get("passed"):
        st.caption("কোনো expired signal নেই।")
    for item in buckets.get("passed", []):
        is_tp = "TP HIT" in item.get("status", "")
        if is_tp:
            st.success(f"✅ **{item['tf']}** | {item['status']} | {item.get('time', '—')}")
        else:
            st.error(f"❌ **{item['tf']}** | {item['status']} | {item.get('time', '—')}")
        if item.get("signal"):
            _render_signal_card(pair_name, item, expanded=False)
        else:
            st.caption(item.get("explanation", ""))


# ─── API Layer ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=15, show_spinner=False)
def _discover_api_base(configured_base: str) -> str:
    for base_url in _candidate_api_bases(configured_base):
        try:
            r = requests.get(f"{base_url}{API_HEALTH_PATH}", timeout=(1.5, 1.5))
            r.raise_for_status()
            return base_url.rstrip("/")
        except requests.RequestException:
            continue
    raise RuntimeError("No healthy backend on localhost:8000-8005. Run start.py first.")


def get_api_base(force_refresh: bool = False) -> str:
    if force_refresh:
        _discover_api_base.clear()
    resolved = _discover_api_base(DEFAULT_API_BASE)
    st.session_state["resolved_api_base"] = resolved
    return resolved


@st.cache_data(ttl=20, show_spinner=False)
def _api_get_cached(api_base: str, endpoint: str, params_key: tuple[tuple[str, str], ...], timeout: int) -> dict:
    r = requests.get(f"{api_base}{endpoint}", params=dict(params_key), timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_get(endpoint: str, params: dict | None = None, timeout: int = 30) -> dict | None:
    api_base = get_api_base()
    try:
        return _api_get_cached(api_base, endpoint, _normalise_params(params), timeout)
    except requests.RequestException:
        try:
            api_base = get_api_base(force_refresh=True)
            return _api_get_cached(api_base, endpoint, _normalise_params(params), timeout)
        except Exception as exc:
            st.error(f"API error ({endpoint}): {exc}")
            return None
    except Exception as exc:
        st.error(f"API error ({endpoint}): {exc}")
        return None


def api_post(endpoint: str, json_body: dict | None = None) -> dict | None:
    api_base = get_api_base()
    try:
        r = requests.post(f"{api_base}{endpoint}", json=json_body or {}, timeout=180)
        r.raise_for_status()
        _api_get_cached.clear()
        return r.json()
    except requests.RequestException:
        try:
            api_base = get_api_base(force_refresh=True)
            r = requests.post(f"{api_base}{endpoint}", json=json_body or {}, timeout=180)
            r.raise_for_status()
            _api_get_cached.clear()
            return r.json()
        except Exception as exc:
            st.error(f"API error ({endpoint}): {exc}")
            return None
    except Exception as exc:
        st.error(f"API error ({endpoint}): {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    '<div style="padding:10px 12px;border:1px solid #21262d;border-radius:10px;background:#111827;">'
    '<div style="font-size:11px;letter-spacing:.18em;color:#9ca3af;margin-bottom:4px;">DIO</div>'
    '<div style="font-size:20px;font-weight:700;color:#f8fafc;line-height:1;">TRADING</div>'
    '<div style="font-size:11px;color:#94a3b8;margin-top:6px;">Signal Dashboard</div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
pair = st.sidebar.selectbox("📊 Pair", SUPPORTED_PAIRS, index=0)
st.sidebar.markdown("---")

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

nc1, nc2 = st.sidebar.columns(2)
if nc1.button("🏠 Home", use_container_width=True, key="btn_home"):
    st.session_state.nav_page = "Home"
    st.rerun()
if nc2.button("📊 Signals", use_container_width=True, key="btn_signals"):
    st.session_state.nav_page = "Live Signals"
    st.rerun()

nc3, nc4 = st.sidebar.columns(2)
if nc3.button("📉 Backtest", use_container_width=True, key="btn_backtest"):
    st.session_state.nav_page = "Backtest"
    st.rerun()
if nc4.button("🤖 Model", use_container_width=True, key="btn_model"):
    st.session_state.nav_page = "Model & Prediction"
    st.rerun()

page = st.session_state.nav_page

st.sidebar.markdown("---")
try:
    _api_url = get_api_base()
    st.sidebar.caption(f"API ✓ {_api_url.split(':')[-1]} • ⚠️ Not financial advice")
except Exception:
    st.sidebar.caption("API ⚠️ offline • ⚠️ Not financial advice")


# ═══════════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.markdown(f"# 💼 {pair} Dashboard")
    summary = api_get("/dashboard/summary", {"pair": pair})

    if summary:
        # Key metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", _pfmt(pair, summary.get("current_price", 0)),
                   delta=_pip_label(pair, summary.get("live_analysis_delta", 0.0)))
        c2.metric("RSI(14)", f"{summary.get('rsi', 0):.1f}")
        c3.metric("Volatility", summary.get("volatility_regime", "—").title())
        c4.metric("Session", summary.get("current_session", "—").replace("_", " ").title())

        # Recommended setup
        rec = summary.get("recommended_live_setup") or {}
        if rec:
            st.info(
                f"🎯 **Recommended**: {rec.get('primary_timeframe', '1d').upper()} | "
                f"{rec.get('signal_mode', 'high_accuracy').replace('_', ' ').title()} | "
                f"Scan {rec.get('scan_bars_min', 3)}-{rec.get('scan_bars_max', 5)} bars"
            )

        # Market + Timing side-by-side
        left, right = st.columns(2)
        with left:
            st.markdown("**Market Conditions**")
            st.caption(
                f"Trend: **{summary.get('trend_primary', '—').upper()}** · "
                f"Liquidity: **{summary.get('liquidity_zone', '—').upper()}** · "
                f"Volume: **{summary.get('volume_condition', '—')}**"
            )
            st.caption(
                f"Model: **{'✓ Loaded' if summary.get('model_loaded') else '✗ Not Loaded'}** · "
                f"Events: **{'⚠️ Active' if summary.get('event_window') else '✓ Clear'}**"
            )
        with right:
            st.markdown("**Timing**")
            st.caption(f"Quote: {_fmt_time(summary.get('live_price_time'))} {_tz_label()}")
            st.caption(f"Analysis: {_fmt_time(summary.get('analysis_time'))} {_tz_label()}")
            st.caption(f"Source: {summary.get('live_price_source', '—')} · Gap: {_pip_label(pair, summary.get('live_analysis_delta', 0.0))}")

        if summary.get("event_window"):
            st.warning("⚠️ High-impact event window ACTIVE — Signals may be suppressed")

        st.caption(summary.get("volume_disclaimer", ""))
    else:
        st.warning("Dashboard data unavailable. Is the backend running?")


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Live Signals":
    st.markdown(f"# 📊 Live Signals — {pair}")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        scan_bars = st.slider("Scan Bars", 1, 20, 3)
    with fc2:
        sig_mode_label = st.radio("Mode", ["High Accuracy", "Balanced"], horizontal=True)
    with fc3:
        auto_refresh = st.toggle("Auto-Refresh (60s)", value=True)

    sig_mode_api = "high_accuracy" if sig_mode_label == "High Accuracy" else "balanced"
    all_pairs_btn = st.button("🌐 ALL Pairs Snapshot", use_container_width=True)

    if auto_refresh:
        import time as _time

    # Scan signals
    with st.spinner("Scanning all timeframes..."):
        sig_data = api_get(
            "/signals/multi-tf",
            {"pair": pair, "scan_bars": scan_bars, "signal_mode": sig_mode_api},
            timeout=90,
        )

    if sig_data:
        dash = api_get("/dashboard/summary", {"pair": pair})
        live_price = float(dash["current_price"]) if dash and dash.get("current_price") else None
        buckets = _build_buckets(pair, sig_data, live_price)

        # Quick counters
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Signals", sig_data.get("signal_count", 0))
        q2.metric("🟢 Running", len(buckets["running"]))
        q3.metric("🟣 Upcoming", len(buckets["upcoming"]))
        q4.metric("🔴 Passed", len(buckets["passed"]))

        st.caption(f"Scan: {_fmt_time(sig_data.get('timestamp'))} {_tz_label()}")

        best_tf = sig_data.get("best_timeframe") or {}
        if best_tf:
            st.caption(
                f"🏆 Best: {best_tf.get('timeframe', '?').upper()} · "
                f"WR {best_tf.get('win_rate', 0) * 100:.1f}% · PF {best_tf.get('profit_factor', 0):.2f}"
            )

        st.markdown("---")

        # ── 3-Type Signal Sections (single source of truth) ──
        _render_three_types(pair, buckets)

        # ── Diagnostics (only when zero signals) ──
        if sig_data.get("signal_count", 0) == 0:
            st.markdown("---")
            st.markdown("### 🔍 Why No Signals?")
            for ov in (sig_data.get("timeframe_overview") or []):
                diag = ov.get("signal_diagnostics") or {}
                if not diag:
                    continue
                tf_lbl = diag.get("timeframe", "?").upper()
                buy = diag.get("buy") or {}
                sell = diag.get("sell") or {}
                with st.expander(f"{tf_lbl}: {diag.get('summary', 'No setup')}", expanded=False):
                    b1, s1 = st.columns(2)
                    with b1:
                        st.caption(f"**BUY** — Conf {float(buy.get('confidence', 0) or 0):.0f} | ML {float(buy.get('model_probability', 0) or 0) * 100:.0f}%")
                        for r in (buy.get("blockers") or [])[:3]:
                            st.caption(f"  • {r}")
                    with s1:
                        st.caption(f"**SELL** — Conf {float(sell.get('confidence', 0) or 0):.0f} | ML {float(sell.get('model_probability', 0) or 0) * 100:.0f}%")
                        for r in (sell.get("blockers") or [])[:3]:
                            st.caption(f"  • {r}")

    # All pairs snapshot
    if all_pairs_btn:
        st.markdown("---")
        st.markdown("## 🌐 All Pairs Snapshot")
        rows = []
        with st.spinner("Scanning all pairs..."):
            for p in SUPPORTED_PAIRS:
                p_sig = api_get("/signals/multi-tf", {"pair": p, "scan_bars": scan_bars, "signal_mode": sig_mode_api}, timeout=90)
                p_dash = api_get("/dashboard/summary", {"pair": p})
                p_live = float(p_dash["current_price"]) if p_dash and p_dash.get("current_price") else None
                if not p_sig:
                    rows.append({"pair": p, "🟢": 0, "🟣": 0, "🔴": 0})
                    continue
                pb = _build_buckets(p, p_sig, p_live)
                rows.append({"pair": p, "🟢": len(pb["running"]), "🟣": len(pb["upcoming"]), "🔴": len(pb["passed"])})
                with st.expander(f"{p} — 🟢{len(pb['running'])} 🟣{len(pb['upcoming'])} 🔴{len(pb['passed'])}"):
                    _render_three_types(p, pb)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if auto_refresh:
        _time.sleep(60)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Backtest":
    st.markdown(f"# 📉 Backtest — {pair}")

    cfg_col, run_col, all_col = st.columns([4, 1, 1])
    with cfg_col:
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            strategy = st.text_input("Strategy", "DioMultiCondition_v1", label_visibility="collapsed")
        with sc2:
            bt_tf = st.selectbox("Timeframe", _allowed_timeframes_for_pair(pair), index=0)
        with sc3:
            rr = st.selectbox("R:R", [1.0, 1.5, 2.0, 2.5, 3.0], index=0, label_visibility="collapsed")
        with sc4:
            force = st.checkbox("Refresh", label_visibility="collapsed")
    with run_col:
        run_bt = st.button("▶️ Run", use_container_width=True, type="primary")
    with all_col:
        run_all = st.button("🚀 ALL", use_container_width=True)

    if run_bt:
        with st.spinner("Running backtest..."):
            res = api_post("/backtest/run", {
                "pair": pair, "timeframe": bt_tf, "strategy_name": strategy,
                "rr": rr, "force_refresh": force, "recent_signals": 20,
            })
        if res:
            st.success("✓ Backtest complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signals", res.get("total_signals", 0))
            m2.metric("Win Rate", f"{res.get('total_win_rate', 0) * 100:.1f}%")
            m3.metric("Profit Factor", f"{res.get('profit_factor', 0):.2f}")
            m4.metric("Max DD", f"{res.get('max_drawdown_pct', 0):.1f}%")

            buy_n = int(res.get("buy_signals", 0) or 0)
            sell_n = int(res.get("sell_signals", 0) or 0)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Buy WR", f"{res.get('buy_win_rate', 0) * 100:.1f}%" if buy_n else "N/A")
            d2.metric("Sell WR", f"{res.get('sell_win_rate', 0) * 100:.1f}%" if sell_n else "N/A")
            d3.metric("Sharpe", f"{res.get('sharpe_ratio', 0):.3f}")
            d4.metric("Return", f"{res.get('total_return_pct', 0):.1f}%")

            st.caption(f"BUY: {buy_n} · SELL: {sell_n} · Bias: {res.get('direction_bias', 'BOTH')}")

            if res.get("fallback_from_recent_scope"):
                st.info("Recent scope empty → full-history metrics shown.")

            equity = res.get("equity_curve") or []
            if equity:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=equity, mode="lines", name="Equity", line=dict(color="#00d26a")))
                fig.update_layout(template="plotly_dark", height=220, margin=dict(l=30, r=30, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

    if run_all:
        rows = []
        total = sum(len(_allowed_timeframes_for_pair(p)) for p in SUPPORTED_PAIRS)
        prog = st.progress(0.0)
        done = 0
        with st.spinner("Running all backtests..."):
            for p in SUPPORTED_PAIRS:
                for tf in _allowed_timeframes_for_pair(p):
                    r = api_post("/backtest/run", {
                        "pair": p, "timeframe": tf, "strategy_name": strategy,
                        "rr": rr, "force_refresh": force, "recent_signals": 20,
                    })
                    done += 1
                    prog.progress(done / total)
                    if r:
                        rows.append({
                            "pair": p, "tf": tf,
                            "signals": r.get("total_signals", 0),
                            "wr_%": round(float(r.get("total_win_rate", 0) or 0) * 100, 1),
                            "pf": round(float(r.get("profit_factor", 0) or 0), 2),
                            "dd_%": round(float(r.get("max_drawdown_pct", 0) or 0), 1),
                            "ret_%": round(float(r.get("total_return_pct", 0) or 0), 1),
                        })
                    else:
                        rows.append({"pair": p, "tf": tf, "signals": None, "wr_%": None, "pf": None, "dd_%": None, "ret_%": None})
        st.session_state["all_bt"] = rows
        st.success("All backtests done.")

    if st.session_state.get("all_bt"):
        st.markdown("**All Backtest Results**")
        st.dataframe(pd.DataFrame(st.session_state["all_bt"]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL & PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Model & Prediction":
    st.markdown(f"# 🤖 ML Model — {pair}")

    allowed_tf = _allowed_timeframes_for_pair(pair)
    model_tf = st.selectbox("Timeframe", allowed_tf, index=min(2, len(allowed_tf) - 1))

    # Bulk actions
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button("🚀 Train ALL Pairs + TF", type="primary", use_container_width=True):
            rows = []
            total = sum(len(_allowed_timeframes_for_pair(p)) for p in SUPPORTED_PAIRS)
            prog = st.progress(0.0)
            done = 0
            for p in SUPPORTED_PAIRS:
                for tf in _allowed_timeframes_for_pair(p):
                    r = api_post("/model/train", {"pair": p, "timeframe": tf})
                    done += 1
                    prog.progress(done / total)
                    if r:
                        rows.append({"pair": p, "tf": tf, "status": "✓",
                                     "test_auc": round(float(r.get("test_auc", 0) or 0), 4),
                                     "cv_auc": round(float(r.get("cv_mean_auc", 0) or 0), 4)})
                    else:
                        rows.append({"pair": p, "tf": tf, "status": "✗", "test_auc": None, "cv_auc": None})
            st.session_state["all_train"] = rows
            st.success("All training done.")

    with ac2:
        if st.button("📊 Predict ALL Pairs + TF", use_container_width=True):
            rows = []
            total = sum(len(_allowed_timeframes_for_pair(p)) for p in SUPPORTED_PAIRS)
            prog = st.progress(0.0)
            done = 0
            for p in SUPPORTED_PAIRS:
                for tf in _allowed_timeframes_for_pair(p):
                    pred = api_get("/model/predict", {"pair": p, "timeframe": tf})
                    done += 1
                    prog.progress(done / total)
                    if pred:
                        rows.append({
                            "pair": p, "tf": tf,
                            "bull_%": round(float(pred.get("bull_probability", 0) or 0) * 100, 1),
                            "bear_%": round(float(pred.get("bear_probability", 0) or 0) * 100, 1),
                            "lean": pred.get("direction_lean", "—"),
                            "conf": pred.get("confidence_label", "—"),
                        })
                    else:
                        rows.append({"pair": p, "tf": tf, "bull_%": None, "bear_%": None, "lean": "N/A", "conf": "N/A"})
            st.session_state["all_pred"] = rows
            st.success("All predictions done.")

    if st.session_state.get("all_train"):
        st.markdown("**Training Results**")
        st.dataframe(pd.DataFrame(st.session_state["all_train"]), use_container_width=True, hide_index=True)

    if st.session_state.get("all_pred"):
        st.markdown("**Probabilities**")
        st.dataframe(pd.DataFrame(st.session_state["all_pred"]), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Single pair train/predict
    tc, pc = st.columns(2)
    with tc:
        st.markdown(f"**Train {pair} [{model_tf}]**")
        if st.button("Train Model", type="primary", use_container_width=True):
            with st.spinner("Training..."):
                result = api_post("/model/train", {"pair": pair, "timeframe": model_tf})
            if result:
                st.success("✓ Trained")
                t1, t2 = st.columns(2)
                t1.metric("Test AUC", f"{result.get('test_auc', 0):.4f}")
                t2.metric("CV AUC", f"{result.get('cv_mean_auc', 0):.4f}")
                fi = result.get("feature_importance_top10") or {}
                if fi:
                    fig = px.bar(
                        pd.DataFrame(fi.items(), columns=["Feature", "Importance"]).sort_values("Importance"),
                        x="Importance", y="Feature", orientation="h", template="plotly_dark", height=280,
                    )
                    fig.update_layout(margin=dict(l=100, r=20, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

    with pc:
        st.markdown(f"**Predict {pair} [{model_tf}]**")
        if st.button("Get Probability", use_container_width=True):
            with st.spinner("Computing..."):
                pred = api_get("/model/predict", {"pair": pair, "timeframe": model_tf})
            if pred:
                p1, p2 = st.columns(2)
                p1.metric("Bull", f"{pred.get('bull_probability', 0) * 100:.1f}%")
                p2.metric("Bear", f"{pred.get('bear_probability', 0) * 100:.1f}%")
                lean = pred.get("direction_lean", "NEUTRAL")
                emoji = "📈" if lean == "BULLISH" else "📉" if lean == "BEARISH" else "➡️"
                st.markdown(f"**{emoji} {lean}** ({pred.get('confidence_label', 'low')})")
                st.progress(pred.get("bull_probability", 0.5))
