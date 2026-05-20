import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from backtesting.data_handler import CSVDataHandler
from backtesting.portfolio import BacktestPortfolio
from backtesting.performance import PerformanceCalculator
from backtesting.engines.event_driven import EventDrivenBacktestEngine
from backtesting.engines.vectorized import VectorizedBacktestEngine
from backtesting.scenarios.walk_forward import WalkForwardValidator
from backtesting.scenarios.monte_carlo import MonteCarloSimulator
from models.ensemble.signal_generator import AlphaSignal


def create_mock_csv(tmp_path) -> str:
    """Generates a mock EURUSD CSV file for tests."""
    data = []
    start_dt = datetime(2026, 1, 1)
    
    # 100 minutes of data
    for i in range(100):
        dt = start_dt + timedelta(minutes=i)
        # Price goes up slowly
        data.append({
            "timestamp": dt,
            "open": 1.1000 + (i * 0.0001),
            "high": 1.1005 + (i * 0.0001),
            "low": 1.0995 + (i * 0.0001),
            "close": 1.1001 + (i * 0.0001),
            "volume": 1000 + i
        })
        
    df = pd.DataFrame(data)
    file_path = tmp_path / "EURUSD.csv"
    df.to_csv(file_path, index=False)
    return str(tmp_path)


def test_csv_data_handler(tmp_path):
    csv_dir = create_mock_csv(tmp_path)
    handler = CSVDataHandler(csv_dir=csv_dir, pairs=["EURUSD"])
    handler.load_data()
    
    # Ensure loaded successfully
    assert "EURUSD" in handler.data
    assert len(handler.data["EURUSD"]) == 100
    
    # Stream first bar
    bars = list(handler.stream_bars())
    assert len(bars) == 100
    assert bars[0]["pair"] == "EURUSD"
    assert bars[0]["close"] == 1.1001


def test_backtest_portfolio():
    portfolio = BacktestPortfolio(initial_capital=100000.0, commission_per_lot=2.0)
    
    # Long Order Fill
    fill = {
        "pair": "EURUSD",
        "direction": 1,
        "size": 100000.0, # 1 lot
        "fill_price": 1.1000,
        "timestamp": datetime.utcnow()
    }
    portfolio.apply_fill(fill)
    
    # Verify Commission applied ($2.0)
    assert portfolio.cash == 99998.0
    assert portfolio.positions["EURUSD"] == 100000.0
    
    # Mark to market (price moves up 100 pips to 1.1100)
    # Unrealized = 100000 * (1.1100 - 1.1000) = 1000 USD
    latest_prices = {"EURUSD": 1.1100}
    equity = portfolio.update_equity(datetime.utcnow(), latest_prices)
    assert equity == 100998.0


def test_performance_calculator():
    equity = [100000.0, 101000.0, 102000.0, 99000.0]
    metrics = PerformanceCalculator.calculate_metrics(equity)
    
    assert metrics["total_return_pct"] == pytest.approx(-1.0) # 99k / 100k - 1
    assert metrics["max_drawdown_pct"] == pytest.approx(2.941, 0.001) # (102k - 99k) / 102k = 2.94%


def test_vectorized_engine(tmp_path):
    csv_dir = create_mock_csv(tmp_path)
    handler = CSVDataHandler(csv_dir=csv_dir, pairs=["EURUSD"])
    portfolio = BacktestPortfolio(initial_capital=100000.0)
    
    engine = VectorizedBacktestEngine(
        data_handler=handler,
        portfolio=portfolio,
        config={"primary_pair": "EURUSD"}
    )
    
    results = engine.run()
    assert "performance" in results
    assert results["final_equity"] > 0.0


def test_walk_forward_validator():
    # Generate 100 days of data
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=i) for i in range(110)]
    df = pd.DataFrame(index=dates)
    df.index.name = "timestamp"
    
    validator = WalkForwardValidator(train_days=30, test_days=10)
    splits = validator.generate_splits(df)
    
    # 100 days total. 
    # Split 1: Train 0-30, Test 30-40.
    # Split 2: Train 10-40, Test 40-50.
    # Split 3: Train 20-50, Test 50-60.
    # Split 4: Train 30-60, Test 60-70.
    # Split 5: Train 40-70, Test 70-80.
    # Split 6: Train 50-80, Test 80-90.
    # Split 7: Train 60-90, Test 90-100.
    assert len(splits) == 7
    assert len(splits[0][0]) == 31 # inclusive boundary
    assert len(splits[0][1]) == 11


def test_monte_carlo_simulator():
    simulator = MonteCarloSimulator(iterations=100)
    # Strategy returns: alternating +1% and -0.5%
    returns = [0.01, -0.005] * 50
    
    res = simulator.run_simulations(returns, initial_capital=100000.0)
    assert res["probability_of_ruin_pct"] == 0.0 # Standard drift is positive, ruin unlikely
    assert res["median_final_equity"] > 100000.0
