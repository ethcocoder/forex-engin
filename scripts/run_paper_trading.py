import time
import numpy as np
import structlog
from datetime import datetime

from risk.risk_engine import RiskEngine
from risk.sizing.fixed_fractional import FixedFractionalSizer
from risk.limits.liquidity_filter import SpreadFilter
from execution.brokers.paper_broker import PaperBroker
from execution.execution_engine import ExecutionEngine
from infrastructure.trading_pipeline import TradingPipeline
from monitoring.performance_tracker import PerformanceTracker

# Mock Ensemble Aggregator for simulation
class MockEnsemble:
    def predict(self, features, return_signal=True):
        from models.ensemble.signal_generator import AlphaSignal
        
        # Randomly generate a signal 10% of the time
        if np.random.rand() > 0.9:
            direction = 1 if np.random.rand() > 0.5 else -1
            return AlphaSignal(
                direction=direction,
                magnitude=0.5 + np.random.rand() * 0.5,
                confidence=0.8,
                uncertainty=0.1,
                expected_decay_steps=10,
                regime=0,
                timestamp=time.time(),
                metadata={}
            )
        else:
            return AlphaSignal(0, 0.0, 0.0, 0.0, 0, 0, time.time(), {})


logger = structlog.get_logger()

def run_simulation(days: int = 30, pair: str = "EURUSD", initial_capital: float = 100000.0):
    logger.info("Starting Paper Trading Fast-Forward Simulation", days=days, pair=pair)
    
    # Setup Tracker
    tracker = PerformanceTracker(initial_capital=initial_capital)
    
    # Setup Models (Mocked for speed in this script, real system loads checkpoints)
    ensemble = MockEnsemble()
    
    # Setup Risk
    risk_engine = RiskEngine()
    risk_engine.set_sizer(FixedFractionalSizer(fraction=0.01)) # 1% risk per trade
    risk_engine.register_filter(SpreadFilter(default_max_spread_pips=5.0))
    
    # Setup Execution
    broker = PaperBroker(config={"initial_capital": initial_capital})
    execution_engine = ExecutionEngine(broker=broker)
    
    # Setup Pipeline
    pipeline = TradingPipeline(
        ensemble=ensemble,
        risk_engine=risk_engine,
        execution_engine=execution_engine,
        initial_capital=initial_capital
    )
    
    # Simulation Loop
    # 30 days = 720 hours = 43200 minutes
    # We will simulate 1 tick per minute to fast-forward
    total_ticks = days * 24 * 60
    
    current_price = 1.1000
    
    start_time = time.time()
    
    for i in range(total_ticks):
        # 1. Simulate Market Data
        # Random walk price
        price_change = np.random.normal(0, 0.0001)
        current_price += price_change
        
        market_data = {
            "mid_price": current_price,
            "spread_pips": 1.5 + np.random.rand(), # 1.5 to 2.5 pips
            "adv": 1000000.0,
            "pip_value": 10.0, # Approximate for EURUSD 1 standard lot
            "volatility": 0.001
        }
        
        # Update broker simulator with true market state
        broker.update_market_state({pair: market_data})
        
        # 2. Process Tick
        features = np.random.randn(10) # Mock features
        pipeline.process_tick(pair, features, market_data)
        
        # 3. Simulate Exits (Close position if it exists after some time)
        # For this simulation, we randomly close open positions to realize PnL
        positions = broker.get_positions()
        if pair in positions and np.random.rand() > 0.95:
            # Generate close order
            current_size = positions[pair]
            direction = -1 if current_size > 0 else 1
            
            from risk.risk_engine import OrderRequest
            close_order = OrderRequest(pair=pair, direction=direction, size=abs(current_size))
            result = broker.place_order(close_order)
            
            # Record Performance
            if result.get("status") == "FILLED":
                # Get the PnL (in a real system, the broker adapter would return this or we calculate it)
                # Let's read the broker's cash difference
                pnl = broker.cash - pipeline.portfolio_state.current_equity
                return_pct = pnl / pipeline.portfolio_state.current_equity
                
                pipeline.update_pnl(pnl, return_pct)
                tracker.log_trade(
                    pair=pair,
                    direction=direction,
                    size=abs(current_size),
                    pnl=pnl,
                    slippage_pips=result.get("slippage_pips", 0.0)
                )
                tracker.update_equity(broker.cash, time.time())
                
        # Progress logging
        if i > 0 and i % 10000 == 0:
            logger.info("Simulation progress", ticks=i, total=total_ticks, current_equity=broker.cash)
            
    exec_time = time.time() - start_time
    logger.info("Simulation complete", time_seconds=exec_time)
    
    # Generate Report
    print("\n" + "="*50)
    print(tracker.generate_tear_sheet())
    print("="*50 + "\n")

if __name__ == "__main__":
    run_simulation(days=30, pair="EURUSD")
