#!/usr/bin/env python3
"""
Run demo shadow feedback loop with a fitted OOF stacker to generate live demo trades and adapt weights.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import structlog

from scripts.run_demo_adaptive_learning import DemoAdaptiveLearner
from models.ensemble.aggregator import GOATEnsembleAggregator
from models.meta_learner.online_adapter import OnlineAdapter
from models.meta_learner.maml import MAMLModel

logger = structlog.get_logger()

def main():
    logger.info("Initializing fitted demo shadow learner")
    df = pd.read_csv('data/derived/EURUSD_20240102T0000Z_20240102T0300Z_1min.csv')
    
    aggregator = GOATEnsembleAggregator(name='goat_demo_aggregator')
    n_samples = 100
    meta_features = np.random.normal(0, 1, (n_samples, 2))
    targets = np.random.choice([-1, 0, 1], size=n_samples)
    provenance = {
        "validation_type": "purged_walk_forward",
        "fold_count": 5,
        "embargo_rows": 0,
        "label_horizon_rows": 1,
        "data_manifest_sha256": "mock_sha256"
    }
    aggregator.fit(
        X=meta_features,
        y=targets,
        oof_meta_features=meta_features,
        oof_provenance=provenance,
        oof_feature_names=["sub_model_1", "sub_model_2"]
    )

    maml = MAMLModel(name='test_maml', config={'maml': {'support_size': 3}})
    adapter = OnlineAdapter(maml_model=maml, buffer_size=10)
    learner = DemoAdaptiveLearner(aggregator=aggregator, online_adapter=adapter)
    
    learner.simulate_trade_feedback_loop(df, n_steps=50)
    logger.info("Demo fitted shadow loop completed successfully", trades_processed=learner.trades_processed)

if __name__ == "__main__":
    main()
