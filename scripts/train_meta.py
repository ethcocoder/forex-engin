import os
import sys
import argparse
import numpy as np
import pandas as pd
import structlog
import pickle
import torch

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.meta_learner.maml import MAMLModel
from models.meta_learner.trainer import MAMLTrainer
from configs.loader import load_config

logger = structlog.get_logger()

def create_dataset(features_df: pd.DataFrame, raw_df: pd.DataFrame, scaler_path: str, seq_len: int, horizon: int):
    logger.info("Aligning features and computing forward return targets...")
    
    features_arr = features_df.values
    close_prices = raw_df['close'].values
    
    if len(features_arr) != len(close_prices):
        raise ValueError("Features and Raw DataFrames must have the exact same number of rows!")
    
    # 1. Standardize features using the saved scaler from temporal training if available
    if os.path.exists(scaler_path):
        logger.info(f"Loading feature scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            mean = scaler["mean"]
            std = scaler["std"]
        features_arr = (features_arr - mean) / std
        features_arr = np.nan_to_num(features_arr, 0.0)
    else:
        logger.warning(f"Feature scaler not found at {scaler_path}. Fitting a new scaler.")
        mean = np.nanmean(features_arr, axis=0)
        std = np.nanstd(features_arr, axis=0)
        std[std == 0] = 1e-8
        features_arr = (features_arr - mean) / std
        features_arr = np.nan_to_num(features_arr, 0.0)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump({"mean": mean, "std": std}, f)
            
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
    
    logger.info(f"Dataset created! X: {X_valid.shape}, y: {y_valid.shape}")
    return X_valid, y_valid

def main():
    parser = argparse.ArgumentParser(description="Train the Model-Agnostic Meta-Learner (MAML)")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw CSV (for close prices)")
    parser.add_argument("--scaler", type=str, default="saved_models/feature_scaler.pkl", help="Path to feature scaler pickle file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of meta-training epochs")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length (lookback window)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (forward steps)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--cv", action="store_true", help="Run walk-forward meta-validation before final training")
    parser.add_argument("--output", type=str, default="saved_models/maml_model.pt", help="Path to output saved MAML model")
    
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
    
    # Create dataset (X and y returns)
    X, y = create_dataset(features_df, raw_df, args.scaler, args.seq_len, args.horizon)
    
    # Flatten sequence features if required by MAML MLP structure (first step dimensions: Linear(d_feat, 128))
    # Note: MAMLNetwork's forward takes x[:, -1, :] if x is 3D, meaning it automatically takes the latest timestep of the window.
    # So we don't need to manually flatten it.
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
        app_config = load_config(config_path)
        config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    except Exception as e:
        logger.warning(f"Could not load config file, using empty dict: {e}")
        config = {}
        
    # Get modular meta learner config
    meta_cfg = config.get("models", {}).get("meta_learner", {})
    maml_config = {
        "device": args.device,
        "maml": {
            "inner_lr": meta_cfg.get("inner_lr", 0.01),
            "outer_lr": meta_cfg.get("outer_lr", 0.0001),
            "num_inner_steps": meta_cfg.get("adaptation_steps", 5),
            "support_size": meta_cfg.get("support_set_size", 50),
            "query_size": 20,
            "meta_batch_size": 8,
            "meta_epochs": args.epochs
        }
    }
    
    model = MAMLModel(name="maml", config=maml_config)
    
    if args.cv:
        logger.info("Evaluating meta-adaptation quality using walk-forward meta-validation...")
        trainer = MAMLTrainer(n_splits=3, holdout_pct=0.2)
        results = trainer.evaluate_meta_learning(model, X, y)
        logger.info(f"Walk-forward Meta-Validation results: {results}")
        
    logger.info("Starting MAML meta-training on full dataset...")
    model.fit(X, y)
    
    # Save final model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    logger.info(f"MAML Meta-Learner model saved successfully to {args.output}")

if __name__ == "__main__":
    main()
