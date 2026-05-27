import pickle
import numpy as np
import pandas as pd
import os
import sys
import torch

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.temporal.combined import TemporalFusionModel
from models.meta_learner.maml import MAMLModel
from models.regime.combined import RegimeEnsembleEstimator
from models.rl_agent.ppo_agent import PPOModel
from models.ensemble.aggregator import EnsembleAggregator

class TorchEnsembleModelWrapper(torch.nn.Module):
    def __init__(self, inner_torch_model, scaler_mean, scaler_std, raw_feature_indices, device):
        super().__init__()
        self.inner_model = inner_torch_model
        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32).to(device)
        self.scaler_std = torch.tensor(scaler_std, dtype=torch.float32).to(device)
        self.raw_feature_indices = raw_feature_indices
        
    def forward(self, x):
        x_raw = x[:, :, self.raw_feature_indices]
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_std
        x_scaled = torch.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.inner_model(x_scaled)


class TemporalEnsembleWrapper:
    def __init__(self, temporal_model, scaler_mean, scaler_std, raw_feature_indices, device):
        self.temporal_model = temporal_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.raw_feature_indices = raw_feature_indices
        self.model = TorchEnsembleModelWrapper(
            temporal_model.model, scaler_mean, scaler_std, raw_feature_indices, device
        )
        
    def predict(self, X, **kwargs):
        X_raw = X[:, :, self.raw_feature_indices]
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.temporal_model.predict(X_scaled)


class MAMLEnsembleWrapper:
    def __init__(self, maml_model, scaler_mean, scaler_std, raw_feature_indices, device):
        self.maml_model = maml_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.raw_feature_indices = raw_feature_indices
        self.model = TorchEnsembleModelWrapper(
            maml_model.model, scaler_mean, scaler_std, raw_feature_indices, device
        )
        
    def predict(self, X, **kwargs):
        X_raw = X[:, :, self.raw_feature_indices]
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.maml_model.predict(X_scaled)


class RegimeEnsembleWrapper:
    def __init__(self, regime_model, regime_mean, regime_std, hmm_features, features_df, seq_len=60):
        self.regime_model = regime_model
        self.regime_mean = regime_mean
        self.regime_std = regime_std
        self.hmm_features = hmm_features
        self.features_df = features_df
        self.seq_len = seq_len
        
    def predict(self, X, **kwargs):
        return X[:, -1, -4:]


class RLEnsembleWrapper:
    def __init__(self, ppo_model, features_cols, regime_cols, features_df, scaler_mean, scaler_std, seq_len=60):
        self.ppo_model = ppo_model
        self.features_cols = features_cols
        self.regime_cols = regime_cols
        self.features_df = features_df
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.seq_len = seq_len
        
    def predict(self, X, **kwargs):
        n_samples = X.shape[0]
        feats_raw = X[:, -1, :len(self.features_cols)]
        feats = (feats_raw - self.scaler_mean) / self.scaler_std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        pos = np.full((n_samples, 1), kwargs.get("current_position", 0.0), dtype=np.float32)
        unrealized = np.full((n_samples, 1), kwargs.get("unrealized_pnl", 0.0) / 10000.0, dtype=np.float32)
        time_ind = np.full((n_samples, 1), kwargs.get("time_indicator", 0.0), dtype=np.float32)
        
        regimes = X[:, -1, -len(self.regime_cols):]
        
        obs = np.hstack([feats, pos, unrealized, time_ind, regimes])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        actions, _ = self.ppo_model.model.predict(obs, deterministic=True)
        
        action_mapping = {0: 0.0, 1: 0.5, 2: 1.0, 3: -0.5, 4: -1.0}
        pred = np.array([action_mapping[int(a)] for a in actions], dtype=np.float64)
        return pred


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
