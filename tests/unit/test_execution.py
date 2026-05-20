import time
import pytest
import unittest
from unittest.mock import MagicMock

from risk.risk_engine import OrderRequest
from execution.brokers.paper_broker import PaperBroker
from execution.simulation.slippage_model import SlippageModel
from execution.simulation.market_impact import MarketImpactModel
from execution.routing.twap import TWAPRouter
from execution.routing.smart_router import SmartRouter
from execution.execution_engine import ExecutionEngine


def create_order(size=10000.0, direction=1):
    return OrderRequest(
        pair="EURUSD",
        direction=direction,
        size=size
    )

def test_slippage_model():
    model = SlippageModel(volatility_scalar=0.5, noise_std=0.0) # Zero noise for deterministic test
    # base spread = 2.0, impact = 1.0, vol = 1.0
    # spread_penalty = 1.0
    # vol_penalty = 1.0 * (1.0 + 0.5) = 1.5
    # total = 2.5 pips
    slippage = model.calculate_slippage(base_spread_pips=2.0, market_impact_pips=1.0, volatility=1.0)
    assert slippage == 2.5

def test_market_impact_model():
    model = MarketImpactModel(impact_scalar=0.1)
    # size = 10k, adv = 1M -> participation = 0.01 -> sqrt = 0.1
    # impact = 0.1 * 1.0 * 0.1 = 0.01 pips
    impact = model.calculate_impact(order_size=10000.0, average_daily_volume=1000000.0)
    assert pytest.approx(impact, 0.0001) == 0.01

def test_paper_broker_fill_and_pnl():
    broker = PaperBroker(config={"initial_capital": 100000.0})
    
    # Mock zero slippage for simple test
    broker.slippage_model.calculate_slippage = MagicMock(return_value=0.0)
    broker.impact_model.calculate_impact = MagicMock(return_value=0.0)
    
    market_data = {
        "EURUSD": {
            "mid_price": 1.1000,
            "pip_value": 0.0001
        }
    }
    broker.update_market_state(market_data)
    
    # Buy 100k
    order1 = create_order(size=100000.0, direction=1)
    result1 = broker.place_order(order1)
    assert result1["status"] == "FILLED"
    assert result1["fill_price"] == 1.1000
    assert broker.get_positions()["EURUSD"] == 100000.0
    
    # Market moves up 100 pips to 1.1100
    market_data["EURUSD"]["mid_price"] = 1.1100
    broker.update_market_state(market_data)
    
    # Sell 100k to close
    order2 = create_order(size=100000.0, direction=-1)
    result2 = broker.place_order(order2)
    assert result2["status"] == "FILLED"
    assert result2["fill_price"] == 1.1100
    
    # Position should be closed
    assert "EURUSD" not in broker.get_positions()
    
    # PnL = 100k * (1.1100 - 1.1000) = 100k * 0.01 = 1000 USD
    assert pytest.approx(broker.get_account_balance(), 0.001) == 101000.0

def test_twap_router():
    router = TWAPRouter(slices=5, duration_seconds=1, randomize=False)
    order = create_order(size=50000.0)
    
    mock_broker = MagicMock()
    mock_broker.place_order.return_value = {"status": "FILLED"}
    
    # Route should return True immediately
    assert router.route(order, mock_broker) is True
    
    # Wait for background thread to finish (approx 1 second)
    time.sleep(1.2)
    
    # Should be called 5 times with size 10000
    assert mock_broker.place_order.call_count == 5
    first_call_order = mock_broker.place_order.call_args_list[0][0][0]
    assert first_call_order.size == 10000.0

def test_execution_engine_retry():
    mock_broker = MagicMock()
    # Fail first two times with network errors, succeed on third
    mock_broker.place_order.side_effect = [
        Exception("Network Error 1"),
        Exception("Network Error 2"),
        {"status": "FILLED"}
    ]
    
    engine = ExecutionEngine(broker=mock_broker)
    order = create_order()
    
    # Should eventually succeed because max_retries = 3
    # Wait time will be 2^1 + 2^2 = 6 seconds total
    # Let's mock time.sleep so the test runs instantly
    with unittest.mock.patch('time.sleep', return_value=None):
        result = engine._execute_direct(order, max_retries=3)
    
    assert result is True
    assert mock_broker.place_order.call_count == 3
