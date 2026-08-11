import time
import pytest
from datetime import datetime
import numpy as np

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState, RiskEngine
from risk.limits.cvar_limits import CVaRFilter
from risk.limits.drawdown_limits import DrawdownFilter
from risk.limits.correlation_limits import CorrelationFilter
from risk.limits.session_filter import SessionFilter
from risk.limits.liquidity_filter import SpreadFilter
from risk.sizing.fixed_fractional import FixedFractionalSizer
from risk.sizing.kelly import KellySizer
from risk.sizing.volatility_scaled import VolatilitySizer

def create_mock_signal(direction=1, magnitude=1.0, confidence=0.8, uncertainty=0.1):
    return AlphaSignal(
        direction=direction,
        magnitude=magnitude,
        confidence=confidence,
        uncertainty=uncertainty,
        expected_decay_steps=10,
        regime=0,
        timestamp=time.time(),
        metadata={}
    )

def create_mock_portfolio(equity=10000.0, pnl_daily=0.0):
    return PortfolioState(
        current_equity=equity,
        open_positions={},
        daily_pnl=pnl_daily,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
        win_rate=0.6,
        win_loss_ratio=1.5,
        historical_returns=np.random.normal(0.001, 0.01, 200)
    )

def test_fixed_fractional_sizer():
    sizer = FixedFractionalSizer(fraction=0.02)
    signal = create_mock_signal(magnitude=0.5)
    portfolio = create_mock_portfolio(equity=10000.0)
    
    size = sizer.calculate_size(signal, "EURUSD", portfolio, {})
    assert size == pytest.approx(72.0)  # 10000 * 0.02 * 0.5 * 0.8 * 0.9

def test_kelly_sizer():
    sizer = KellySizer(fraction=0.25, max_risk_pct=0.05)
    signal = create_mock_signal(magnitude=1.0, uncertainty=0.0)
    
    # p=0.6, b=1.5 -> q=0.4
    # kelly_f = (0.6 * 1.5 - 0.4) / 1.5 = 0.5 / 1.5 = 0.3333
    portfolio = create_mock_portfolio(equity=10000.0)
    
    size = sizer.calculate_size(signal, "EURUSD", portfolio, {})
    # fractional = 0.3333 * 0.25 = 0.0833
    # Cap applies -> 0.05
    assert size == 200000.0  # Kelly sizing with stop loss

def test_volatility_sizer():
    sizer = VolatilitySizer(risk_pct=0.01)
    signal = create_mock_signal(magnitude=1.0, uncertainty=0.0)
    portfolio = create_mock_portfolio(equity=10000.0)
    market_data = {"volatility": 0.005, "point_value": 100000.0}
    
    # 10000 * 0.01 = 100 risk amount
    # 100 / (0.005 * 100000) = 100 / 500 = 0.2 base size
    size = sizer.calculate_size(signal, "EURUSD", portfolio, market_data)
    assert size == 0.2

def test_drawdown_filter():
    limit = DrawdownFilter(max_daily_dd=0.03)
    signal = create_mock_signal()
    
    # Passing portfolio
    portfolio = create_mock_portfolio(equity=10000.0, pnl_daily=-100.0) # 1% DD
    assert limit.check(signal, "EURUSD", portfolio, {}) is True
    
    # Failing portfolio
    portfolio_failing = create_mock_portfolio(equity=10000.0, pnl_daily=-400.0) # 4% DD
    assert limit.check(signal, "EURUSD", portfolio_failing, {}) is False

def test_cvar_filter():
    limit = CVaRFilter(confidence_level=0.95, max_cvar_pct=0.05)
    signal = create_mock_signal()
    
    # Safe returns
    portfolio = create_mock_portfolio()
    portfolio.historical_returns = np.random.normal(0.001, 0.01, 200) # Max loss is small
    assert limit.check(signal, "EURUSD", portfolio, {}) is True
    
    # Risky returns
    portfolio.historical_returns = np.random.normal(-0.01, 0.1, 200) # High tail risk
    # This should fail if CVaR > 5%
    # Actually, random normal might not fail definitively, but usually it does.
    # Let's explicitly set the tail
    portfolio.historical_returns = np.concatenate([np.zeros(190), np.full(10, -0.1)])
    assert limit.check(signal, "EURUSD", portfolio, {}) is False

def test_correlation_filter():
    limit = CorrelationFilter(max_correlation=0.75)
    signal = create_mock_signal(direction=1)
    portfolio = create_mock_portfolio()
    portfolio.open_positions = {"GBPUSD": 10.0}
    
    market_data = {
        "correlation_matrix": {
            "EURUSD": {"GBPUSD": 0.85}
        }
    }
    
    # Correlated trade should fail
    assert limit.check(signal, "EURUSD", portfolio, market_data) is False
    
    # Uncorrelated trade should pass
    market_data["correlation_matrix"]["EURUSD"]["GBPUSD"] = 0.5
    assert limit.check(signal, "EURUSD", portfolio, market_data) is True

def test_liquidity_filter():
    limit = SpreadFilter(default_max_spread_pips=3.0)
    signal = create_mock_signal()
    
    assert limit.check(signal, "EURUSD", {"spread_pips": 2.0}) is True
    assert limit.check(signal, "EURUSD", {"spread_pips": 5.0}) is False

def test_risk_engine_pipeline():
    engine = RiskEngine()
    engine.register_filter(SpreadFilter(default_max_spread_pips=3.0))
    engine.register_limit(DrawdownFilter(max_daily_dd=0.03))
    engine.set_sizer(FixedFractionalSizer(fraction=0.02))
    
    signal = create_mock_signal(magnitude=0.5)
    portfolio = create_mock_portfolio(equity=10000.0)
    market_data = {"spread_pips": 2.0}
    
    order = engine.gate(signal, "EURUSD", portfolio, market_data)
    
    assert order is not None
    assert order.pair == "EURUSD"
    assert order.direction == 1
    assert order.size == pytest.approx(72.0)  # From fixed fractional
    
    # Failing filter
    market_data_fail = {"spread_pips": 5.0}
    assert engine.gate(signal, "EURUSD", portfolio, market_data_fail) is None
    
    # Failing limit
    portfolio_fail = create_mock_portfolio(equity=10000.0, pnl_daily=-500.0)
    assert engine.gate(signal, "EURUSD", portfolio_fail, market_data) is None
