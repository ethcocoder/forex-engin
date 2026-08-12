#!/usr/bin/env python3
"""
Cost-Aware Research Simulation for Forex Engin.
Evaluates model features and baseline rules under realistic spread/commission assumptions on EUR/USD hourly history.
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
    logger.info("Starting Cost-Aware Research Simulation")
    h1_path = "data/raw/EURUSD_H1_2y.csv"
    if not os.path.exists(h1_path):
        logger.error("Hourly dataset not found", path=h1_path)
        return

    df_h1 = pd.read_csv(h1_path)
    df_h1 = df_h1.rename(columns={"Datetime": "timestamp", "Close": "mid"})
    df_h1["timestamp"] = pd.to_datetime(df_h1["timestamp"], utc=True, errors="raise")
    # Apply realistic 1.5 pip spread (0.00015)
    spread = 0.00015
    df_h1["bid"] = df_h1["mid"] - spread / 2.0
    df_h1["ask"] = df_h1["mid"] + spread / 2.0

    pipeline = LeakageSafeFeaturePipeline(window_sizes=(5, 15, 60))
    features = pipeline.compute_features(df_h1)
    labelled = pipeline.attach_executable_labels(features, horizon_bars=1, min_return=0.0001)

    harness = ModelTrainingHarness(n_estimators=100)
    result = harness.evaluate_purged_walk_forward(labelled, pipeline, n_splits=5)
    
    logger.info("Cost-aware simulation completed", status=result.get("status"), aggregate_return=result.get("aggregate_cumulative_return"))
    print("Simulation Result Summary:")
    print(f"Status: {result.get('status')}")
    print(f"Mean Balanced Accuracy: {result.get('mean_balanced_accuracy', 0.0):.4f}")
    print(f"Aggregate Cumulative Return: {result.get('aggregate_cumulative_return', 0.0):.4f}")

if __name__ == "__main__":
    main()
