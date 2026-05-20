import os
import sys
import argparse
import numpy as np
import pandas as pd
import structlog
import pickle

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.temporal.combined import TemporalFusionModel
from models.temporal.trainer import TimeSeriesPurgedTrainer
from configs.loader import load_config

logger = structlog.get_logger()

def create_dataset(features_df: pd.DataFrame, raw_df: pd.DataFrame, seq_len: int, horizon: int):
    logger.info("Aligning features and computing forward return targets...")
    
    features_arr = features_df.values
    close_prices = raw_df['close'].values
    
    if len(features_arr) != len(close_prices):
        raise ValueError("Features and Raw DataFrames must have the exact same number of rows!")
    
    # 1. Standardize features
    mean = np.nanmean(features_arr, axis=0)
    std = np.nanstd(features_arr, axis=0)
    std[std == 0] = 1e-8
    features_arr = (features_arr - mean) / std
    
    # Fill remaining NaNs from standardizing empty columns
    features_arr = np.nan_to_num(features_arr, 0.0)
    
    # 2. Sliding windows
    from numpy.lib.stride_tricks import sliding_window_view
    
    windows = sliding_window_view(features_arr, window_shape=(seq_len, features_arr.shape[1]))
    X = windows.squeeze(1)  # shape: (N - seq_len + 1, seq_len, d_feat)
    
    n_samples = len(X) - horizon
    if n_samples <= 0:
        raise ValueError(f"Not enough data to create windows of length {seq_len} and horizon {horizon}.")
        
    X_valid = np.copy(X[:n_samples])
    
    # Current time index is i + seq_len - 1
    current_idx = np.arange(seq_len - 1, seq_len - 1 + n_samples)
    future_idx = current_idx + horizon
    
    y_valid = np.log(close_prices[future_idx] / close_prices[current_idx])
    
    # Save standardizer params for live inference
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/feature_scaler.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)
        
    logger.info(f"Dataset created! X: {X_valid.shape}, y: {y_valid.shape}")
    return X_valid, y_valid

def main():
    parser = argparse.ArgumentParser(description="Train the Temporal Neural Ensemble")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw CSV (for close prices)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length (lookback window)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (forward steps)")
    parser.add_argument("--cv", action="store_true", help="Run purged cross-validation before final training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features):
        logger.error(f"Features file {args.features} not found. Did you run generate_features.py?")
        sys.exit(1)
        
    if not os.path.exists(args.raw):
        logger.error(f"Raw file {args.raw} not found.")
        sys.exit(1)
        
    logger.info("Loading DataFrames...")
    features_df = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv(args.raw, index_col="timestamp", parse_dates=True)
    
    # Align indexes
    common_idx = features_df.index.intersection(raw_df.index)
    features_df = features_df.loc[common_idx]
    raw_df = raw_df.loc[common_idx]
    
    X, y = create_dataset(features_df, raw_df, args.seq_len, args.horizon)
    
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
        app_config = load_config(config_path)
        config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    except Exception as e:
        logger.warning(f"Could not load config file, using empty dict: {e}")
        config = {}
        
    # Override config with argparse
    if "temporal_fusion" not in config:
        config["temporal_fusion"] = {}
    config["temporal_fusion"]["epochs"] = args.epochs
    config["temporal_fusion"]["batch_size"] = args.batch_size
    
    model = TemporalFusionModel(name="temporal_fusion", config=config)
    
    if args.cv:
        logger.info("Starting Purged Walk-Forward Cross-Validation...")
        trainer = TimeSeriesPurgedTrainer(n_splits=4, label_horizon=args.horizon, embargo_pct=0.01)
        metrics = trainer.evaluate_cv(model, X, y)
        logger.info(f"CV Mean RMSE: {metrics['mean_rmse']:.6f}")
        
    logger.info("Training final model on full dataset...")
    model.fit(X, y)
    
    save_path = "saved_models/temporal_model.pt"
    model.save(save_path)
    logger.info(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    main()
