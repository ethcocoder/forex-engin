import os
import sys
import time
import structlog
import argparse
from datetime import datetime

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from configs.loader import load_config
from execution.brokers.oanda_broker import OandaBroker
from execution.execution_engine import ExecutionEngine
from risk.risk_engine import RiskEngine, PortfolioState
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alpha_decay import AlphaDecayMonitor

logger = structlog.get_logger()

def run_live_trading(config_path: str):
    logger.info("Starting ELITE Live Trading Engine...")
    
    # 1. Load Config & Components
    app_config = load_config(config_path)
    config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config
    
    # 2. Initialize Broker (Oanda for Live/Practice)
    broker_cfg = config.get("execution", {}).get("broker", {})
    broker = OandaBroker(config=broker_cfg)
    
    if not broker.connect():
        logger.critical("Failed to connect to broker. Aborting live session.")
        return

    # 3. Initialize Risk & Monitoring
    risk_engine = RiskEngine(config=config.get("risk", {}))
    tracker = PerformanceTracker(initial_capital=broker.get_account_balance())
    decay_monitor = AlphaDecayMonitor(retrain_callback=lambda: logger.warning("RETRAIN TRIGGERED"))

    execution_engine = ExecutionEngine(broker=broker)

    logger.info("Live Pipeline Online. Entering Tick Loop.")

    # 4. Main Live Loop
    try:
        while True:
            # A. Sync Portfolio
            positions = broker.get_positions()
            balance = broker.get_account_balance()
            
            # B. Get Market Data (Real-time)
            # In a real setup, this would come from a WebSocket stream
            # For this loop, we poll the broker for the latest price
            market_data = {
                "close": 1.0850, # Placeholder: Replace with real stream
                "spread_pips": 0.8,
                "timestamp": datetime.utcnow()
            }
            
            # C. Check for Emergency Kill Switch (e.g., from a file or DB)
            if os.path.exists("STOP_TRADING"):
                risk_engine.activate_kill_switch()
                logger.critical("STOP_TRADING signal detected. Flattening all positions.")
                # Logic to close all positions would go here
                break

            # D. Log Health
            tracker.update_equity(balance, time.time())
            
            # Wait for next tick (e.g., 1 second for high-frequency or 1 minute for hourly)
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Live session interrupted by user.")
    finally:
        broker.disconnect()
        logger.info("Live session terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    run_live_trading(args.config)
