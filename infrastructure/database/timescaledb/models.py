import datetime
from typing import Optional
from sqlalchemy import Column, DateTime, Double, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class Tick(Base):
    """
    SQLAlchemy Model for the raw ticks hypertable.
    Uses time and pair as composite primary keys to support ORM requirements.
    """
    __tablename__ = "ticks"

    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    pair: Mapped[str] = mapped_column(String(10), primary_key=True)
    bid: Mapped[float] = mapped_column(Double)
    ask: Mapped[float] = mapped_column(Double)
    volume: Mapped[float] = mapped_column(Double)


class OHLCV(Base):
    """
    SQLAlchemy Model for the OHLCV hypertable.
    Uses time, pair, and tf as composite primary keys to support ORM requirements.
    """
    __tablename__ = "ohlcv"

    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    pair: Mapped[str] = mapped_column(String(10), primary_key=True)
    tf: Mapped[str] = mapped_column(String(5), primary_key=True)
    open: Mapped[float] = mapped_column(Double)
    high: Mapped[float] = mapped_column(Double)
    low: Mapped[float] = mapped_column(Double)
    close: Mapped[float] = mapped_column(Double)
    volume: Mapped[float] = mapped_column(Double)


class Feature(Base):
    """
    SQLAlchemy Model for the features hypertable.
    Uses time, pair, tf, and feature_name as composite primary keys.
    """
    __tablename__ = "features"

    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    pair: Mapped[str] = mapped_column(String(10), primary_key=True)
    tf: Mapped[str] = mapped_column(String(5), primary_key=True)
    feature_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[float] = mapped_column(Double)


class Order(Base):
    """
    SQLAlchemy Model for the orders audit table.
    Uses primary key id.
    """
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    time: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    pair: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(10))
    size: Mapped[float] = mapped_column(Double)
    price: Mapped[float] = mapped_column(Double)
    status: Mapped[str] = mapped_column(String(20))
    fill_price: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    slippage: Mapped[Optional[float]] = mapped_column(Double, nullable=True)


class Trade(Base):
    """
    SQLAlchemy Model for the trades audit table.
    Uses primary key id.
    """
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    open_time: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    close_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pair: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(10))
    open_price: Mapped[float] = mapped_column(Double)
    close_price: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    pnl_pct: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    hold_time_sec: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
