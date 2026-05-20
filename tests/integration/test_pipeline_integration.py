import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import RiskEngine
from risk.sizing.fixed_fractional import FixedFractionalSizer
from execution.brokers.paper_broker import PaperBroker
from execution.execution_engine import ExecutionEngine
from infrastructure.trading_pipeline import TradingPipeline

class DeterministicMockEnsemble:
    def __init__(self):
        self.call_count = 0
        
    def predict(self, features, return_signal=True):
        self.call_count += 1
        
        # Issue a Long signal on the very first tick
        if self.call_count == 1:
            return AlphaSignal(
                direction=1,
                magnitude=1.0,
                confidence=0.9,
                uncertainty=0.1,
                expected_decay_steps=10,
                regime=0,
                timestamp=time.time(),
                metadata={}
            )
        
        # Issue a Short (Close) signal on the 5th tick
        if self.call_count == 5:
            return AlphaSignal(
                direction=-1,
                magnitude=1.0,
                confidence=0.9,
                uncertainty=0.1,
                expected_decay_steps=10,
                regime=0,
                timestamp=time.time(),
                metadata={}
            )
            
        # Flat otherwise
        return AlphaSignal(0, 0.0, 0.0, 0.0, 0, 0, time.time(), {})


def test_end_to_end_pipeline():
    """
    Tests a full cycle:
    Tick -> Ensemble -> Signal -> Risk Engine -> Order -> Execution Engine -> PaperBroker Fill -> Portfolio State Sync
    """
    # 1. Setup
    ensemble = DeterministicMockEnsemble()
    
    risk_engine = RiskEngine()
    risk_engine.set_sizer(FixedFractionalSizer(fraction=0.01)) # 1% of 100k = 1000 size
    
    broker = PaperBroker(config={"initial_capital": 100000.0})
    # Mock slippage to 0 for deterministic math
    broker.slippage_model.calculate_slippage = MagicMock(return_value=0.0)
    broker.impact_model.calculate_impact = MagicMock(return_value=0.0)
    
    execution_engine = ExecutionEngine(broker=broker)
    
    pipeline = TradingPipeline(
        ensemble=ensemble,
        risk_engine=risk_engine,
        execution_engine=execution_engine,
        initial_capital=100000.0
    )
    
    # Base Market Data
    market_data = {
        "mid_price": 1.1000,
        "spread_pips": 2.0,
        "adv": 1000000.0,
        "pip_value": 0.0001,
        "volatility": 0.001
    }
    
    # Tick 1: Expect Long Order to open
    broker.update_market_state({"EURUSD": market_data})
    pipeline.process_tick("EURUSD", np.zeros(10), market_data)
    
    # Verify open position
    positions = broker.get_positions()
    assert "EURUSD" in positions
    assert positions["EURUSD"] == 1000.0
    
    # Verify Portfolio State synced (will be done on next tick)
    
    # Tick 2, 3, 4: Flat signals, just price moves up
    market_data["mid_price"] = 1.1100
    broker.update_market_state({"EURUSD": market_data})
    
    for _ in range(3):
        pipeline.process_tick("EURUSD", np.zeros(10), market_data)
        
    # Portfolio state should now reflect the open position from Tick 1
    assert pipeline.portfolio_state.open_positions.get("EURUSD") == 1000.0
        
    # Tick 5: Expect Short Order to close
    pipeline.process_tick("EURUSD", np.zeros(10), market_data)
    
    # Verify closed position
    positions = broker.get_positions()
    assert "EURUSD" not in positions
    
    # Verify PnL realization
    # Entry at 1.1000, Exit at 1.1100. Diff = 0.01
    # Size = 1000. PnL = 1000 * 0.01 = 10.0
    assert pytest.approx(broker.get_account_balance(), 0.001) == 100010.0

if __name__ == "__main__":
    test_end_to_end_pipeline()
    print("Integration test passed successfully!")
