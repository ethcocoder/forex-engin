-- Enable TimescaleDB extension if not already present
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. Raw Tick Data Table
CREATE TABLE IF NOT EXISTS ticks (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    pair VARCHAR(10) NOT NULL,
    bid DOUBLE PRECISION NOT NULL,
    ask DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable partitioned on 'time'
SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);

-- Create composite index on pair and time for speedy range queries
CREATE INDEX IF NOT EXISTS idx_ticks_pair_time ON ticks (pair, time DESC);


-- 2. Resampled OHLCV Candles Table
CREATE TABLE IF NOT EXISTS ohlcv (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    pair VARCHAR(10) NOT NULL,
    tf VARCHAR(5) NOT NULL, -- e.g., '1m', '5m', '1h', '1d'
    "open" DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    "close" DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable partitioned on 'time'
SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

-- Index for scanning specific pair and timeframe bars
CREATE INDEX IF NOT EXISTS idx_ohlcv_pair_tf_time ON ohlcv (pair, tf, time DESC);


-- 3. Extracted Mathematical Feature Matrices Table
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    pair VARCHAR(10) NOT NULL,
    tf VARCHAR(5) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable partitioned on 'time'
SELECT create_hypertable('features', 'time', if_not_exists => TRUE);

-- Index for retrieving target feature matrix subsets
CREATE INDEX IF NOT EXISTS idx_features_pair_tf_name_time ON features (pair, tf, feature_name, time DESC);


-- 4. Order Execution History Audit Table
CREATE TABLE IF NOT EXISTS orders (
    id VARCHAR(100) PRIMARY KEY,
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    pair VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
    size DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'PENDING', 'FILLED', 'REJECTED', 'CANCELLED'
    fill_price DOUBLE PRECISION,
    slippage DOUBLE PRECISION
);

-- Index for reviewing past orders by submission date
CREATE INDEX IF NOT EXISTS idx_orders_time ON orders (time DESC);


-- 5. Portfolio Trade Performance Audit Table
CREATE TABLE IF NOT EXISTS trades (
    id VARCHAR(100) PRIMARY KEY,
    open_time TIMESTAMP WITH TIME ZONE NOT NULL,
    close_time TIMESTAMP WITH TIME ZONE,
    pair VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    open_price DOUBLE PRECISION NOT NULL,
    close_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION, -- absolute PnL in currency
    pnl_pct DOUBLE PRECISION, -- percentage return
    hold_time_sec DOUBLE PRECISION -- duration of the trade
);

-- Index for historical PnL tracking and calculation
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades (open_time DESC);
