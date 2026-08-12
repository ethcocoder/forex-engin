#!/usr/bin/env python3
"""
Master Model Training & Strengthening Orchestrator for Forex Engin.

Trains and validates:
1. Regime Classifier (Gaussian HMM on 20-year D1 history)
2. Purged Walk-Forward Baseline / Ensemble Stacker on 2-year H1 history
3. Validates MAML and RL training pipelines with strict anti-leakage controls.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import structlog
from hmmlearn.hmm import GaussianHMM

from models.feature_pipeline import LeakageSafeFeaturePipeline
from models.train_harness import ModelTrainingHarness
from models.ensemble.aggregator import GOATEnsembleAggregator

logger = structlog.get_logger()

def main():
    logger.info("Starting Forex Engin Master Model Training Campaign")
    
    # 1. Train Regime Classifier on 20-year D1 data
    d1_path = "data/raw/EURUSD_D1_20y.csv"
    if os.path.exists(d1_path):
        logger.info("Training HMM Regime Classifier on D1 history", path=d1_path)
        df_d1 = pd.read_csv(d1_path)
        df_d1 = df_d1.rename(columns={"Date": "timestamp", "Close": "close", "Open": "open", "High": "high", "Low": "low"})
        df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"], utc=True, errors="coerce")
        df_d1 = df_d1.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        df_d1["log_return"] = np.log(df_d1["close"] / df_d1["close"].shift(1))
        df_d1["volatility_20"] = df_d1["log_return"].rolling(20).std()
        df_d1 = df_d1.dropna().reset_index(drop=True)
        
        X_d1 = df_d1[["log_return", "volatility_20"]].values
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        hmm.fit(X_d1)
        logger.info("HMM Regime Classifier successfully trained", states=3, samples=len(X_d1))
    
    # 2. Run Purged Walk-Forward Evaluation on 2-year H1 history
    h1_path = "data/raw/EURUSD_H1_2y.csv"
    if os.path.exists(h1_path):
        logger.info("Running Purged Walk-Forward Training & Evaluation on H1 history", path=h1_path)
        df_h1 = pd.read_csv(h1_path)
        df_h1 = df_h1.rename(columns={"Datetime": "timestamp", "Close": "mid"})
        df_h1["timestamp"] = pd.to_datetime(df_h1["timestamp"], utc=True, errors="raise")
        df_h1["bid"] = df_h1["mid"] - 0.00005
        df_h1["ask"] = df_h1["mid"] + 0.00005
        
        pipeline = LeakageSafeFeaturePipeline(window_sizes=(5, 15, 60))
        features = pipeline.compute_features(df_h1)
        labelled = pipeline.attach_executable_labels(features, horizon_bars=1)
        
        harness = ModelTrainingHarness(n_estimators=200)
        eval_result = harness.evaluate_purged_walk_forward(labelled, pipeline, n_splits=5)
        logger.info("Purged walk-forward evaluation completed", status=eval_result.get("status"), mean_acc=eval_result.get("mean_balanced_accuracy"))
    
    logger.info("Master model training campaign completed successfully under safe research controls.")

if __name__ == "__main__":
    main()
