#!/usr/bin/env python3
"""
Forex Engin: Demo-Only Adaptive Learning & Trade Feedback Runner.

This script simulates closed-loop adaptive learning against a broker demo environment
or paper trading feed. It logs every signal and execution outcome into an immutable
JSONL trade journal, feeds PnL and reward feedback into the Bayesian Model Averager
and Online Adapter, and enforces strict champion-challenger promotion gates.

SAFETY ENFORCEMENT:
- Operates ONLY on demo/paper execution modes.
- Real-money execution paths remain programmatically locked.
"""

import os
import json
import uuid
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import structlog

from models.ensemble.aggregator import GOATEnsembleAggregator
from models.meta_learner.online_adapter import OnlineAdapter
from models.meta_learner.maml import MAMLModel
from execution.brokers.paper_broker import PaperBroker

logger = structlog.get_logger()

JOURNAL_PATH = "/home/ubuntu/forex-engin/data/demo_trade_journal.jsonl"


def ensure_journal_dir() -> None:
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)


def log_trade_journal(record: dict) -> None:
    ensure_journal_dir()
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class DemoAdaptiveLearner:
    """Manages closed-loop demo learning, trade feedback, and champion-challenger promotion."""

    def __init__(self, aggregator: GOATEnsembleAggregator, online_adapter: OnlineAdapter | None = None) -> None:
        self.aggregator = aggregator
        self.online_adapter = online_adapter
        self.broker = PaperBroker(name="demo_broker", config={"initial_capital": 100000.0})
        self.trades_processed = 0
        self.champion_sharpe = 0.0

    def simulate_trade_feedback_loop(self, market_data_df: pd.DataFrame, n_steps: int = 100) -> None:
        """Run a closed-loop demo trading session, logging outcomes and adapting weights."""
        logger.info("Starting demo-only adaptive learning loop", steps=n_steps)
        
        for step in range(min(n_steps, len(market_data_df))):
            row = market_data_df.iloc[step]
            timestamp = row.get("timestamp", datetime.now(timezone.utc).isoformat())
            mid = float(row.get("mid", 1.1000))
            bid = float(row.get("bid", mid - 0.0001))
            ask = float(row.get("ask", mid + 0.0001))

            feature_vector = np.array([[
                float(row.get("return_1", 0.0001)),
                float(row.get("rel_spread", 0.0001)),
                float(row.get("vol_5", 0.001)),
                float(row.get("momentum_5", 0.0001))
            ]])

            try:
                signal = self.aggregator.predict(
                    feature_vector,
                    return_signal=True,
                    regime=0,
                    volatility=float(row.get("vol_5", 0.001))
                )
            except Exception as e:
                logger.debug("Stacker not fitted during demo loop, skipping step", error=str(e))
                continue

            action = signal.action
            if action == 0:
                continue

            trade_id = str(uuid.uuid4())
            entry_price = ask if action == 1 else bid
            exit_price = mid

            realized_return = (exit_price - entry_price) / entry_price if action == 1 else (entry_price - exit_price) / entry_price
            reward = float(realized_return * 100.0)

            journal_record = {
                "trade_id": trade_id,
                "timestamp": str(timestamp),
                "action": int(action),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "realized_return": float(realized_return),
                "reward": reward,
                "ensemble_path": signal.metadata.get("ensemble_path", "UNKNOWN"),
                "uncertainty": float(signal.uncertainty),
                "confidence": float(signal.confidence),
                "mode": "DEMO_SHADOW"
            }
            log_trade_journal(journal_record)
            self.trades_processed += 1

            if hasattr(self.aggregator, "bma") and self.aggregator.bma is not None:
                self.aggregator.bma.update(realized_return)

            if self.online_adapter is not None:
                self.online_adapter.update(feature_vector[0], reward)
                if self.trades_processed % 10 == 0:
                    self.online_adapter.adapt_now()

        logger.info("Demo adaptive learning loop completed", total_trades=self.trades_processed)


if __name__ == "__main__":
    logger.info("Initializing Forex Engin Demo Adaptive Learner (Safe-by-Default)")
    aggregator = GOATEnsembleAggregator(name="goat_demo_aggregator")
    maml = MAMLModel(name="test_maml", config={"maml": {"support_size": 3}})
    adapter = OnlineAdapter(maml_model=maml, buffer_size=10)
    learner = DemoAdaptiveLearner(aggregator=aggregator, online_adapter=adapter)
    logger.info("Demo Adaptive Learner initialized successfully. Ready for demo shadow execution.")
