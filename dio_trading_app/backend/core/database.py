"""
Dio Trading App - Database Models
SQLAlchemy ORM models for signals, backtest results, and alerts.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, Text, JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.core.config import SYNC_DATABASE_URL, DATABASE_URL


# ─── Base ─────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─── Models ───────────────────────────────────────────────────────────────────
class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    pair = Column(String(20), default="EUR/USD", nullable=False)
    timeframe = Column(String(10), nullable=False)

    # Direction
    direction = Column(String(10), nullable=False)  # "BUY" | "SELL"

    # Prices
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    risk_reward = Column(Float, default=2.0, nullable=False)

    # Confidence & Probability
    confidence_score = Column(Float, nullable=False)   # 0–100
    win_probability = Column(Float, nullable=False)    # 0–1 (historical win rate × model prob)
    historical_win_rate = Column(Float, nullable=True)
    model_probability = Column(Float, nullable=True)
    setup_frequency = Column(Integer, nullable=True)   # how often this setup appeared historically

    # Context
    session = Column(String(30), nullable=True)
    market_condition = Column(String(50), nullable=True)
    trend_direction = Column(String(20), nullable=True)

    # Explanation
    signal_reason = Column(Text, nullable=True)        # why the signal fired
    invalidation_reason = Column(Text, nullable=True)  # what would cancel the trade

    # Outcome (filled after trade closes)
    outcome = Column(String(10), nullable=True)        # "WIN" | "LOSS" | "BREAKEVEN" | None
    outcome_price = Column(Float, nullable=True)
    pnl_pips = Column(Float, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    # Flags
    is_active = Column(Boolean, default=True)
    volume_disclaimer = Column(Text, nullable=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_at = Column(DateTime, default=datetime.utcnow)
    strategy_name = Column(String(100), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)

    # Performance Metrics
    total_signals = Column(Integer)
    buy_signals = Column(Integer)
    sell_signals = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    buy_win_rate = Column(Float)
    sell_win_rate = Column(Float)
    total_win_rate = Column(Float)
    total_loss_rate = Column(Float)
    profit_pct = Column(Float)
    loss_pct = Column(Float)
    expected_value = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    initial_capital = Column(Float)
    final_capital = Column(Float)

    # Serialized trade log
    trade_log = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    signal_id = Column(Integer, nullable=True)
    alert_type = Column(String(30), nullable=False)  # "SIGNAL" | "RISK" | "NEWS"
    message = Column(Text, nullable=False)
    delivered = Column(Boolean, default=False)
    channel = Column(String(20), default="log")      # "log" | "email" | "webhook"


class EconomicEvent(Base):
    __tablename__ = "economic_events"

    id = Column(Integer, primary_key=True, index=True)
    event_date = Column(DateTime, nullable=False)
    name = Column(String(200), nullable=False)
    currency = Column(String(10), nullable=False)
    impact = Column(String(10), nullable=True)        # "HIGH" | "MEDIUM" | "LOW"
    actual = Column(String(50), nullable=True)
    forecast = Column(String(50), nullable=True)
    previous = Column(String(50), nullable=True)
    sentiment_score = Column(Float, nullable=True)    # -1 to +1


# ─── Engine & Sessions ────────────────────────────────────────────────────────
sync_engine = create_engine(SYNC_DATABASE_URL, connect_args={"check_same_thread": False})
SyncSession = sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)

async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


def init_db():
    """Create all tables synchronously (used at startup)."""
    Base.metadata.create_all(bind=sync_engine)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
