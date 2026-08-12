#!/usr/bin/env python3
"""
Run purged walk-forward evaluation on the downloaded 2-year hourly EUR/USD dataset.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import structlog

from models.feature_pipeline import LeakageSafeFeaturePipeline
from models.train_harness import ModelTrainingHarness

logger = structlog.get_logger()

def main():
    logger.info("Loading 2-year hourly EUR/USD history for walk-forward evaluation")
    df = pd.read_csv("data/raw/EURUSD_H1_2y.csv")
    
    # Rename Datetime/Close to match feature pipeline requirements ("timestamp", "bid", "ask", "mid")
    # Yahoo data provides Close, Open, High, Low. We estimate bid/ask with a 1-pip spread.
    df = df.rename(columns={"Datetime": "timestamp", "Close": "mid"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df["bid"] = df["mid"] - 0.00005
    df["ask"] = df["mid"] + 0.00005
    
    pipeline = LeakageSafeFeaturePipeline(window_sizes=(5, 15, 60))
    features = pipeline.compute_features(df)
    labelled = pipeline.attach_executable_labels(features, horizon_bars=1)
    
    harness = ModelTrainingHarness(n_estimators=100)
    result = harness.evaluate_purged_walk_forward(labelled, pipeline, n_splits=5)
    
    print("=== 2-YEAR HOURLY WALK-FORWARD EVALUATION RESULTS ===")
    for k, v in result.items():
        if k != "fold_results":
            print(f"{k}: {v}")
    print("Folds evaluated:", len(result.get("fold_results", [])))

if __name__ == "__main__":
    main()
