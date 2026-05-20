import os
import sys
import argparse
import numpy as np
import pandas as pd
import structlog
import pickle

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.regime.combined import RegimeEnsembleEstimator
from models.regime.trainer import RegimeTrainer
from configs.loader import load_config

logger = structlog.get_logger()

def create_dataset(features_df: pd.DataFrame, hmm_features: list, seq_len: int):
    logger.info("Extracting HMM features and creating sequence windows...", features=hmm_features)
    
    # Check that all features exist
    missing_cols = [col for col in hmm_features if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features in DataFrame: {missing_cols}")
        
    features_arr = features_df[hmm_features].values
    
    # 1. Standardize features
    mean = np.nanmean(features_arr, axis=0)
    std = np.nanstd(features_arr, axis=0)
    std[std == 0] = 1e-8
    features_arr = (features_arr - mean) / std
    features_arr = np.nan_to_num(features_arr, 0.0)
    
    # 2. Sliding windows
    from numpy.lib.stride_tricks import sliding_window_view
    
    windows = sliding_window_view(features_arr, window_shape=(seq_len, features_arr.shape[1]))
    X = windows.squeeze(1)  # shape: (N - seq_len + 1, seq_len, d_feat)
    
    # Copy array to make it writable for PyTorch
    X = np.copy(X)
    
    # Save standardizer params for live inference
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/regime_feature_scaler.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
        
    logger.info(f"Dataset created! X: {X.shape}")
    return X

def main():
    parser = argparse.ArgumentParser(description="Train the Market Regime Classification Ensemble")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length (lookback window)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of LSTM training epochs")
    parser.add_argument("--cv", action="store_true", help="Run walk-forward cross-validation before final training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features):
        logger.error(f"Features file {args.features} not found. Did you run generate_features.py?")
        sys.exit(1)
        
    logger.info("Loading DataFrames...")
    features_df = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
        app_config = load_config(config_path)
        config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    except Exception as e:
        logger.warning(f"Could not load config file, using empty dict: {e}")
        config = {}
        
    # Get HMM features from configuration
    regime_cfg = config.get("models", {}).get("regime", {})
    hmm_features = regime_cfg.get("hmm_features", ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"])
    
    X = create_dataset(features_df, hmm_features, args.seq_len)
    
    # Map configuration parameters to sub-model expectations
    sub_config = {
        "hmm": {
            "n_components": regime_cfg.get("n_states", 4)
        },
        "lstm_regime": {
            "num_classes": regime_cfg.get("n_states", 4),
            "epochs": args.epochs,
            "batch_size": 64
        },
        "ensemble": config.get("models", {}).get("ensemble", {})
    }
    
    # Instantiate the estimator
    estimator = RegimeEnsembleEstimator(name="regime_ensemble", config=sub_config)
    trainer = RegimeTrainer(n_splits=4)
    
    if args.cv:
        logger.info("Starting Walk-Forward Cross-Validation for Regimes...")
        metrics = trainer.evaluate_cv(estimator, X)
        logger.info(f"CV Mean LSTM-HMM Alignment: {metrics['mean_alignment']:.4f}")
        logger.info(f"CV Mean State Entropy: {metrics['mean_entropy']:.4f}")
    else:
        logger.info("Training regime ensemble on full dataset...")
        estimator.fit(X)
        
    # Save the trained model
    save_path = "saved_models/regime_ensemble.pkl"
    estimator.save(save_path)
    logger.info(f"Regime ensemble saved successfully to {save_path}")
    
    # Analyze and print regimes
    logger.info("Analyzing predicted market regimes...")
    analysis = trainer.analyze_regimes(estimator, X)
    for state, pct in analysis["distribution"].items():
        logger.info(f"Regime State {state}: {pct * 100:.2f}% of historical data")

if __name__ == "__main__":
    main()
