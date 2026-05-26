import pickle
import numpy as np
import pandas as pd
import os
import sys
import torch

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ensemble.aggregator import EnsembleAggregator
from models.temporal.combined import TemporalFusionModel
from models.meta_learner.maml import MAMLModel
from models.rl_agent.ppo_agent import PPOModel
from scripts.run_backtest import TemporalEnsembleWrapper, MAMLEnsembleWrapper, RegimeEnsembleWrapper, RLEnsembleWrapper
from models.regime.combined import RegimeEnsembleEstimator

def main():
    print("Loading data...")
    features_df = pd.read_csv("data/EUR_USD_features.csv", index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv("data/EUR_USD_ticks.csv", index_col="timestamp", parse_dates=True)
    
    # Align indices
    common_idx = raw_df.index.intersection(features_df.index)
    raw_df = raw_df.loc[common_idx]
    features_df = features_df.loc[common_idx]

    # Load Regime model
    print("Loading Regime model...")
    regime_model = RegimeEnsembleEstimator()
    regime_model.load("saved_models/regime_ensemble.pkl")
    
    with open("saved_models/regime_feature_scaler.pkl", "rb") as f:
        regime_scaler = pickle.load(f)
        regime_mean = regime_scaler["mean"]
        regime_std = regime_scaler["std"]
        
    hmm_features = ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"]
    regime_features_df = features_df[hmm_features]
    regime_features_scaled = (regime_features_df.values - regime_mean) / regime_std
    regime_features_scaled = np.nan_to_num(regime_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(regime_features_scaled, window_shape=(60, regime_features_scaled.shape[1]))
    X_regime = windows.squeeze(1)
    probs = regime_model.predict(X_regime, return_proba=True)
    
    padding = np.tile(probs[0], (59, 1))
    aligned_probs = np.vstack([padding, probs])
    
    regime_cols = [f"regime_{i}" for i in range(probs.shape[1])]
    for i, col in enumerate(regime_cols):
        features_df[col] = aligned_probs[:, i]

    exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"] + regime_cols
    features_cols = [col for col in features_df.columns if col not in exclude]
    
    with open("saved_models/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        scaler_mean = scaler["mean"]
        scaler_std = scaler["std"]
        
    raw_feature_indices = [features_df.columns.get_loc(col) for col in features_cols]
    
    features_arr = features_df.copy().values
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    windows = sliding_window_view(features_arr, window_shape=(60, features_arr.shape[1]))
    X_master = np.copy(windows.squeeze(1))
    
    n_samples = len(X_master) - 1
    X_valid = np.copy(X_master[:n_samples])
    
    # Load EnsembleAggregator
    print("Loading EnsembleAggregator...")
    agg = EnsembleAggregator(config={})
    agg.load("saved_models/ensemble_aggregator")
    
    temporal_model = TemporalFusionModel(name="temporal_fusion", config={})
    temporal_model.load("saved_models/temporal_model.pt")
    
    maml_model = MAMLModel(name="maml", config={"device": "cpu", "maml": {"inner_lr": 0.01}})
    maml_model.load("saved_models/maml_model.pt")
    
    rl_model = PPOModel(name="ppo_agent", config={"features_cols": features_cols, "regime_cols": regime_cols, "device": "cpu"})
    rl_model.load("saved_models/rl_agent_ppo.zip")
    
    temporal_wrapper = TemporalEnsembleWrapper(temporal_model, scaler_mean, scaler_std, raw_feature_indices, "cpu")
    maml_wrapper = MAMLEnsembleWrapper(maml_model, scaler_mean, scaler_std, raw_feature_indices, "cpu")
    regime_wrapper = RegimeEnsembleWrapper(regime_model, regime_mean, regime_std, hmm_features, features_df, 60)
    rl_wrapper = RLEnsembleWrapper(rl_model, features_cols, regime_cols, features_df, scaler_mean, scaler_std, 60)
    
    agg.register_model("temporal", temporal_wrapper, is_torch=True)
    agg.register_model("maml", maml_wrapper, is_torch=True)
    agg.register_model("regime", regime_wrapper, is_torch=False)
    agg.register_model("rl", rl_wrapper, is_torch=False)
    
    print("Running batch predictions...")
    preds = agg.predict_batch(X_valid)
    print("Predictions shape:", preds.shape)
    print("Predictions min/max/mean/std:")
    print(f"Min: {preds.min():.8f}")
    print(f"Max: {preds.max():.8f}")
    print(f"Mean: {preds.mean():.8f}")
    print(f"Std: {preds.std():.8f}")
    
    # Let's count how many exceed some thresholds
    for t in [0.0, 0.0001, 0.0002, 0.0005, 0.001]:
        exceed = np.sum(np.abs(preds) > t)
        print(f"Count of predictions > {t}: {exceed} ({exceed / len(preds) * 100:.2f}%)")

if __name__ == "__main__":
    main()
