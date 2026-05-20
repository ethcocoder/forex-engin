import os
import sys
import argparse
import numpy as np
import pandas as pd
import structlog
import pickle

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rl_agent.ppo_agent import PPOModel
from models.rl_agent.trainer import RLTrainer
from models.regime.combined import RegimeEnsembleEstimator
from configs.loader import load_config

logger = structlog.get_logger()

def main():
    parser = argparse.ArgumentParser(description="Train the Forex Reinforcement Learning Agent")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw ticks CSV")
    parser.add_argument("--regime_model", type=str, default="saved_models/regime_ensemble.pkl", help="Path to saved regime model")
    parser.add_argument("--regime_scaler", type=str, default="saved_models/regime_feature_scaler.pkl", help="Path to saved regime feature scaler")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total timesteps to train per curriculum stage")
    parser.add_argument("--output", type=str, default="saved_models/rl_agent_ppo.zip", help="Path to output saved RL model zip")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on ('cpu' or 'cuda')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features):
        logger.error(f"Features file {args.features} not found. Did you run generate_features.py?")
        sys.exit(1)
    if not os.path.exists(args.raw):
        logger.error(f"Raw ticks file {args.raw} not found. Did you run download_data.py?")
        sys.exit(1)
    if not os.path.exists(args.regime_model) or not os.path.exists(args.regime_scaler):
        logger.error("Regime model or scaler files not found. Did you train the regime model first?")
        sys.exit(1)
        
    logger.info("Loading DataFrames...")
    features_df = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv(args.raw, index_col="timestamp", parse_dates=True)
    
    # Align indexes
    common_idx = raw_df.index.intersection(features_df.index)
    raw_df = raw_df.loc[common_idx]
    features_df = features_df.loc[common_idx]
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
        app_config = load_config(config_path)
        config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    except Exception as e:
        logger.warning(f"Could not load config file, using empty dict: {e}")
        config = {}
        
    logger.info("Predicting market regimes...")
    # Load regime model and scaler to generate regime probabilities
    regime_model = RegimeEnsembleEstimator()
    regime_model.load(args.regime_model)
    
    with open(args.regime_scaler, "rb") as f:
        scaler = pickle.load(f)
        regime_mean = scaler["mean"]
        regime_std = scaler["std"]
        
    # Get HMM features from configuration
    regime_cfg = config.get("models", {}).get("regime", {})
    hmm_features = regime_cfg.get("hmm_features", ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"])
    
    regime_features_df = features_df[hmm_features]
    regime_features_scaled = (regime_features_df.values - regime_mean) / regime_std
    regime_features_scaled = np.nan_to_num(regime_features_scaled, 0.0)
    
    # Generate sequence windows for regime prediction (defaults to seq_len=60)
    seq_len = 60
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(regime_features_scaled, window_shape=(seq_len, regime_features_scaled.shape[1]))
    X_regime = windows.squeeze(1)
    
    probs = regime_model.predict(X_regime, return_proba=True)
    
    # Pad the beginning of the series to align with features_df length
    padding = np.tile(probs[0], (seq_len - 1, 1))
    aligned_probs = np.vstack([padding, probs])
    
    # Add regime columns to features_df
    regime_cols = [f"regime_{i}" for i in range(probs.shape[1])]
    for i, col in enumerate(regime_cols):
        features_df[col] = aligned_probs[:, i]
        
    # Merge raw features and features_df
    df_for_env = raw_df.copy()
    for col in features_df.columns:
        if col not in df_for_env.columns:
            df_for_env[col] = features_df[col]
            
    # Define features and regime columns for the RL Agent
    exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"] + regime_cols
    features_cols = [col for col in features_df.columns if col not in exclude]
    
    # Setup trainer configuration
    trainer_config = {
        "initial_balance": 10000.0,
        "leverage": 30.0,
        "multiplier": 100000.0,
        "slippage": config.get("execution", {}).get("slippage", {}).get("base_spread_pct", 0.0001),
        "kyle_lambda_multiplier": 1.0,
        "reward_config": None,
        "force_python_fallback": False
    }
    
    logger.info("Initializing PPO RL Agent and Trainer...")
    rl_cfg = config.get("models", {}).get("rl_agent", {})
    rl_cfg["features_cols"] = features_cols
    rl_cfg["regime_cols"] = regime_cols
    rl_cfg["device"] = args.device
    
    agent = PPOModel(name="ppo_agent", config=rl_cfg)
    trainer = RLTrainer(config=trainer_config, checkpoint_dir="saved_models/checkpoints")
    
    # Train the RL Agent using curriculum stages
    logger.info("Starting RL training via volatility curriculum learning...")
    trainer.train(
        agent=agent,
        df=df_for_env,
        features_cols=features_cols,
        regime_cols=regime_cols,
        total_timesteps_per_stage=args.timesteps
    )
    
    # Save final model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    agent.save(args.output)
    logger.info(f"Final RL Agent saved successfully to {args.output}")

if __name__ == "__main__":
    main()
