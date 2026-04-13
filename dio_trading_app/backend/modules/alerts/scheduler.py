"""
Dio Trading App — Alert System
================================
Monitors conditions and dispatches alerts via:
  1. Log (always active)
  2. Email (optional, configure SMTP in .env)
  3. Webhook (optional, configure URL in .env)

Scheduler: APScheduler runs the signal check every N minutes.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from backend.core.config import settings, ALERT_CHECK_INTERVAL_SECONDS
from backend.modules.signals.engine import TradingSignal


# ─── Alert Dispatcher ─────────────────────────────────────────────────────────

def format_signal_alert(signal: TradingSignal) -> str:
    """Format a signal into a human-readable alert message."""
    direction_emoji = "📈 BUY" if signal.direction == "BUY" else "📉 SELL"
    return f"""
╔══════════════════════════════════════════════════════════╗
║  DIO TRADING APP — NEW SIGNAL                            ║
╠══════════════════════════════════════════════════════════╣
║  Pair:        {signal.pair:<20} Time: {signal.generated_at[:16]}  ║
║  Direction:   {direction_emoji:<45}  ║
║  Timeframe:   {signal.timeframe:<45}  ║
╠══════════════════════════════════════════════════════════╣
║  ENTRY:       {signal.entry_price:<45.5f}  ║
║  STOP LOSS:   {signal.stop_loss:<45.5f}  ║
║  TAKE PROFIT: {signal.take_profit:<45.5f}  ║
║  R:R Ratio:   1:{signal.risk_reward:<43.0f}  ║
╠══════════════════════════════════════════════════════════╣
║  Confidence:      {signal.confidence_score:<38.1f}  ║
║  Win Probability: {signal.win_probability * 100:<37.1f}%  ║
║  Historical WR:   {signal.historical_win_rate * 100:<37.1f}%  ║
║  ML Probability:  {signal.model_probability * 100:<37.1f}%  ║
╠══════════════════════════════════════════════════════════╣
║  Session:     {signal.session:<45}  ║
║  Liquidity:   {signal.liquidity_zone:<45}  ║
╠══════════════════════════════════════════════════════════╣
║  WHY:                                                    ║
{'║  ' + signal.signal_reason[:56] + '  ║'}
╠══════════════════════════════════════════════════════════╣
║  INVALIDATION:                                           ║
{'║  ' + signal.invalidation_reason[:56] + '  ║'}
╠══════════════════════════════════════════════════════════╣
║  ⚠ DISCLAIMER: NOT financial advice. Probabilistic      ║
║  signal only. Volume data is proxy/tick — not true       ║
║  spot volume. Past win rates ≠ future results.           ║
╚══════════════════════════════════════════════════════════╝
""".strip()


def send_log_alert(message: str, level: str = "info"):
    if level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)


def send_email_alert(subject: str, body: str):
    """Send email alert. Requires SMTP settings in .env."""
    if not settings.alert_email_to or not settings.smtp_user:
        logger.debug("Email alerting not configured — skipping.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = settings.smtp_user
        msg["To"] = settings.alert_email_to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, settings.alert_email_to, msg.as_string())

        logger.success(f"Email alert sent to {settings.alert_email_to}")
    except Exception as e:
        logger.error(f"Email alert failed: {e}")


def dispatch_signal_alert(signal: TradingSignal):
    """Dispatch a signal alert through all configured channels."""
    message = format_signal_alert(signal)
    subject = f"[Dio] {signal.direction} Signal — EUR/USD @ {signal.entry_price:.5f} (Conf: {signal.confidence_score:.0f})"

    # Always log
    send_log_alert(message)

    # Email if configured
    if settings.alert_email_to:
        send_email_alert(subject, message)


# ─── Scheduler ────────────────────────────────────────────────────────────────

class AlertScheduler:
    """
    Background scheduler that periodically runs the signal scan
    and dispatches alerts.
    """

    def __init__(self, signal_check_fn: Callable):
        """
        signal_check_fn: a callable that returns list[TradingSignal].
        Called on every check interval.
        """
        self.signal_check_fn = signal_check_fn
        self._scheduler = BackgroundScheduler()
        self._last_signal_ids: set = set()

    def _run_check(self):
        logger.debug(f"[AlertScheduler] Running signal check at {datetime.utcnow()}")
        try:
            signals = self.signal_check_fn()
            for sig in signals:
                sig_id = f"{sig.direction}_{sig.entry_price}_{sig.generated_at[:16]}"
                if sig_id not in self._last_signal_ids:
                    dispatch_signal_alert(sig)
                    self._last_signal_ids.add(sig_id)
                    # Keep set bounded
                    if len(self._last_signal_ids) > 100:
                        self._last_signal_ids = set(list(self._last_signal_ids)[-50:])
        except Exception as e:
            logger.error(f"[AlertScheduler] Check failed: {e}")

    def start(self, interval_seconds: int = ALERT_CHECK_INTERVAL_SECONDS):
        self._scheduler.add_job(
            self._run_check,
            "interval",
            seconds=interval_seconds,
            id="signal_check",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.success(f"Alert scheduler started — checking every {interval_seconds}s")

    def stop(self):
        self._scheduler.shutdown(wait=False)
        logger.info("Alert scheduler stopped.")
