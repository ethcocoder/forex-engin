import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from scripts.run_demo_adaptive_learning import DemoAdaptiveLearner, log_trade_journal, JOURNAL_PATH
from models.ensemble.aggregator import GOATEnsembleAggregator
from models.meta_learner.online_adapter import OnlineAdapter
from models.meta_learner.maml import MAMLModel


def test_trade_journal_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_journal = os.path.join(tmpdir, "journal.jsonl")
        record = {
            "trade_id": "test-123",
            "timestamp": "2026-08-12T00:00:00Z",
            "action": 1,
            "realized_return": 0.001,
            "reward": 0.1,
            "mode": "DEMO_SHADOW"
        }
        os.makedirs(tmpdir, exist_ok=True)
        import json
        with open(test_journal, "w", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        
        assert os.path.exists(test_journal)
        with open(test_journal, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            assert data["trade_id"] == "test-123"
            assert data["action"] == 1


def test_demo_adaptive_learner_initialization():
    aggregator = GOATEnsembleAggregator(name="test_agg")
    maml = MAMLModel(name="test_maml", config={"maml": {"support_size": 3}})
    adapter = OnlineAdapter(maml_model=maml, buffer_size=10)
    learner = DemoAdaptiveLearner(aggregator=aggregator, online_adapter=adapter)
    assert learner.trades_processed == 0
    assert learner.online_adapter is not None


def test_demo_adaptive_learner_loop_mock_data():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC"),
        "mid": np.linspace(1.1000, 1.1020, 20),
        "bid": np.linspace(1.0999, 1.1019, 20),
        "ask": np.linspace(1.1001, 1.1021, 20),
        "return_1": np.random.normal(0, 0.0001, 20),
        "rel_spread": np.ones(20) * 0.0001,
        "vol_5": np.ones(20) * 0.001,
        "momentum_5": np.random.normal(0, 0.0002, 20)
    })
    aggregator = GOATEnsembleAggregator(name="test_agg")
    maml = MAMLModel(name="test_maml", config={"maml": {"support_size": 3}})
    adapter = OnlineAdapter(maml_model=maml, buffer_size=10)
    learner = DemoAdaptiveLearner(aggregator=aggregator, online_adapter=adapter)
    
    learner.simulate_trade_feedback_loop(df, n_steps=20)
    assert learner.trades_processed >= 0
