#!/usr/bin/env python3
"""
Train and evaluate HMM regime classification on the 20-year daily EUR/USD dataset.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import structlog
from hmmlearn.hmm import GaussianHMM

logger = structlog.get_logger()

def main():
    logger.info("Loading 20-year daily EUR/USD history for HMM regime training")
    df = pd.read_csv("data/raw/EURUSD_D1_20y.csv")
    
    df = df.rename(columns={"Date": "timestamp", "Close": "close", "Open": "open", "High": "high", "Low": "low"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
    
    # Compute log returns and volatility features for HMM
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_20"] = df["log_return"].rolling(20).std()
    df = df.dropna().reset_index(drop=True)
    
    X = df[["log_return", "volatility_20"]].values
    
    logger.info("Fitting Gaussian HMM with 3 hidden states (Bull, Bear, High Volatility)", samples=len(X))
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    
    df["regime"] = hidden_states
    regime_counts = df["regime"].value_counts().to_dict()
    
    print("=== HMM REGIME CLASSIFICATION RESULTS (20-YEAR D1) ===")
    print("Total daily bars processed:", len(df))
    print("State distribution:", regime_counts)
    print("Transition matrix:\n", model.transmat_)
    logger.info("HMM regime training completed successfully", regime_counts=regime_counts)

if __name__ == "__main__":
    main()
