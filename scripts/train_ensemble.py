import os
import sys

# 1. Parse threads count before imports to configure thread environments
num_threads = None
for i, arg in enumerate(sys.argv):
    if arg.startswith("--threads="):
        num_threads = arg.split("=")[1]
    elif arg == "--threads" and i + 1 < len(sys.argv):
        num_threads = sys.argv[i + 1]

if num_threads is not None:
    try:
        threads_to_use = int(num_threads)
    except ValueError:
        threads_to_use = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
else:
    # Use max cores - 1 by default to boost speed while keeping the OS responsive
    threads_to_use = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

os.environ["OMP_NUM_THREADS"] = str(threads_to_use)
os.environ["MKL_NUM_THREADS"] = str(threads_to_use)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads_to_use)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads_to_use)
os.environ["NUMEXPR_NUM_THREADS"] = str(threads_to_use)

import argparse
import numpy as np
import pandas as pd
import structlog
import pickle
import torch
import gc
import joblib

torch.set_num_threads(threads_to_use)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

# Ensure the root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.temporal.combined import TemporalFusionModel
from models.meta_learner.maml import MAMLModel
from models.regime.combined import RegimeEnsembleEstimator
from models.rl_agent.ppo_agent import PPOModel
from models.ensemble.aggregator import EnsembleAggregator
from configs.loader import load_config

logger = structlog.get_logger()

# Helper classes for PyTorch forward pass wrapping (enables MC Dropout scaling)
class TorchEnsembleModelWrapper(torch.nn.Module):
    def __init__(self, inner_torch_model, scaler_mean, scaler_std, raw_feature_indices, device):
        super().__init__()
        self.inner_model = inner_torch_model
        self.scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32).to(device)
        self.scaler_std = torch.tensor(scaler_std, dtype=torch.float32).to(device)
        self.raw_feature_indices = raw_feature_indices
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_feat_total]
        # Extract raw features that the model was trained on
        x_raw = x[:, :, self.raw_feature_indices]
        # Scale using the temporal feature scaler
        x_scaled = (x_raw - self.scaler_mean) / self.scaler_std
        x_scaled = torch.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.inner_model(x_scaled)


# Wrappers to adapt models to EnsembleAggregator inference expectations
class TemporalEnsembleWrapper:
    def __init__(self, temporal_model, scaler_mean, scaler_std, raw_feature_indices, device):
        self.temporal_model = temporal_model
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.raw_feature_indices = raw_feature_indices
        # Expose inner model wrapper for MC Dropout
        self.model = TorchEnsembleModelWrapper(
            temporal_model.model, scaler_mean, scaler_std, raw_feature_indices, device
        )
        
    def fit(self, X, y=None, **kwargs):
        pass
        
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
        # Expose inner model wrapper for MC Dropout
        self.model = TorchEnsembleModelWrapper(
            maml_model.model, scaler_mean, scaler_std, raw_feature_indices, device
        )
        
    def fit(self, X, y=None, **kwargs):
        pass
        
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
        
    def fit(self, X, y=None, **kwargs):
        pass
        
    def predict(self, X, **kwargs):
        # Extract regime probabilities from the last columns of X (which has been appended with regime columns)
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
        
    def fit(self, X, y=None, **kwargs):
        pass
        
    def predict(self, X, **kwargs):
        # X shape: [n_samples, seq_len, d_feat_total]
        # Construct RL observation: [feats_raw] + [pos] + [unrealized] + [time_indicator] + [regime_probs]
        n_samples = X.shape[0]
        
        # 1. Raw unscaled features (RL agent is trained on raw unscaled features)
        feats_raw = X[:, -1, :len(self.features_cols)]
        feats = np.nan_to_num(feats_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Position (flat during offline prediction/alignment)
        pos = np.zeros((n_samples, 1), dtype=np.float32)
        
        # 3. Unrealized PnL (normalized, flat)
        unrealized = np.zeros((n_samples, 1), dtype=np.float32)
        
        # 4. Normalized time indicators (hour of day / 23.0)
        if isinstance(self.features_df.index, pd.DatetimeIndex):
            hours = self.features_df.index.hour.values.astype(np.float32) / 23.0
        else:
            hours = (np.arange(len(self.features_df)) % 24).astype(np.float32) / 23.0
            
        time_ind = np.array([hours[i + self.seq_len - 1] for i in range(n_samples)], dtype=np.float32).reshape(-1, 1)
        
        # 5. Regime probabilities (last regime_cols)
        regimes = X[:, -1, -len(self.regime_cols):]
        
        obs = np.hstack([feats, pos, unrealized, time_ind, regimes])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        import torch
        obs_tensor = torch.as_tensor(obs, device=self.ppo_model.model.device)
        with torch.no_grad():
            distribution = self.ppo_model.model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs.cpu().numpy()
            
        action_values = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float64)
        pred = np.sum(action_probs * action_values, axis=1)
        return pred


def main():
    parser = argparse.ArgumentParser(description="Train the Ensemble Aggregator stacking layer")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw ticks CSV")
    parser.add_argument("--seq_len", type=int, default=60, help="Lookback sequence length")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output", type=str, default="saved_models/ensemble_aggregator", help="Base path for saved aggregator")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads to use (default: max physical cores - 1)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features) or not os.path.exists(args.raw):
        logger.error("Features or raw data files not found. Ensure previous steps are run.")
        sys.exit(1)
        
    logger.info("Initializing run context", threads_configured=threads_to_use)
    logger.info("Loading DataFrames...")
    features_df = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv(args.raw, index_col="timestamp", parse_dates=True)
    
    # Align indices
    common_idx = raw_df.index.intersection(features_df.index)
    raw_df = raw_df.loc[common_idx]
    features_df = features_df.loc[common_idx]
    
    logger.info("Loading configuration...")
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml")
        app_config = load_config(config_path)
        config = app_config.model_dump() if hasattr(app_config, "model_dump") else app_config.dict()
    except Exception as e:
        logger.warning(f"Could not load config file, using empty dict: {e}")
        config = {}
        
    # 1. Load Regime Ensemble to generate regime probabilities
    logger.info("Loading Regime Ensemble...")
    regime_model = RegimeEnsembleEstimator()
    regime_model.load("saved_models/regime_ensemble.pkl")
    
    with open("saved_models/regime_feature_scaler.pkl", "rb") as f:
        regime_scaler = pickle.load(f)
        regime_mean = regime_scaler["mean"]
        regime_std = regime_scaler["std"]
        
    regime_cfg = config.get("models", {}).get("regime", {})
    hmm_features = regime_cfg.get("hmm_features", ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"])
    
    regime_features_df = features_df[hmm_features]
    regime_features_scaled = (regime_features_df.values - regime_mean) / regime_std
    regime_features_scaled = np.nan_to_num(regime_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(regime_features_scaled, window_shape=(args.seq_len, regime_features_scaled.shape[1]))
    X_regime = windows.squeeze(1)
    
    probs = regime_model.predict(X_regime, return_proba=True)
    
    # Pad probs to match features_df length
    padding = np.tile(probs[0], (args.seq_len - 1, 1))
    aligned_probs = np.vstack([padding, probs])
    
    regime_cols = [f"regime_{i}" for i in range(probs.shape[1])]
    for i, col in enumerate(regime_cols):
        features_df[col] = aligned_probs[:, i]
        
    logger.info("Regime probabilities computed and appended to feature set.")
    
    # 2. Extract column groups
    exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"] + regime_cols
    features_cols = [col for col in features_df.columns if col not in exclude]
    
    # Load raw feature scaler used by Temporal & MAML models
    with open("saved_models/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        scaler_mean = scaler["mean"]
        scaler_std = scaler["std"]
        
    # Get raw indices in features_df
    raw_feature_indices = [features_df.columns.get_loc(col) for col in features_cols]
    
    # 3. Create full master dataset for Stacker training
    # We keep the master features UN-SCALED because individual wrappers handle their own scaling
    features_arr = features_df.copy().values
    features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate rolling windows of shape [n_samples, seq_len, d_feat_total]
    windows = sliding_window_view(features_arr, window_shape=(args.seq_len, features_arr.shape[1]))
    X_master = np.copy(windows.squeeze(1))
    
    n_samples = len(X_master) - args.horizon
    if n_samples <= 0:
        raise ValueError("Not enough data to construct master windows.")
        
    X_valid = np.copy(X_master[:n_samples])
    
    # Targets log returns ratio
    close_prices = raw_df["close"].values
    current_idx = np.arange(args.seq_len - 1, args.seq_len - 1 + n_samples)
    future_idx = current_idx + args.horizon
    y_valid = np.log(close_prices[future_idx] / close_prices[current_idx])
    
    logger.info("Dataset created successfully", X_shape=X_valid.shape, y_shape=y_valid.shape)
    
    # 4. Load remaining submodels and instantiate wrappers
    logger.info("Loading Temporal Fusion Model...")
    temporal_model = TemporalFusionModel(name="temporal_fusion", config=config)
    temporal_model.load("saved_models/temporal_model.pt")
    
    logger.info("Loading MAML Meta-Learner...")
    maml_cfg = config.get("models", {}).get("meta_learner", {})
    maml_model = MAMLModel(name="maml", config={"device": args.device, "maml": {"inner_lr": maml_cfg.get("inner_lr", 0.01)}})
    maml_model.load("saved_models/maml_model.pt")
    
    logger.info("Loading RL PPO Agent...")
    rl_cfg = config.get("models", {}).get("rl_agent", {})
    rl_cfg["features_cols"] = features_cols
    rl_cfg["regime_cols"] = regime_cols
    rl_cfg["device"] = args.device
    rl_model = PPOModel(name="ppo_agent", config=rl_cfg)
    rl_model.load("saved_models/rl_agent_ppo.zip")
    
    # Instantiating wrappers
    temporal_wrapper = TemporalEnsembleWrapper(temporal_model, scaler_mean, scaler_std, raw_feature_indices, args.device)
    maml_wrapper = MAMLEnsembleWrapper(maml_model, scaler_mean, scaler_std, raw_feature_indices, args.device)
    regime_wrapper = RegimeEnsembleWrapper(regime_model, regime_mean, regime_std, hmm_features, features_df, args.seq_len)
    rl_wrapper = RLEnsembleWrapper(rl_model, features_cols, regime_cols, features_df, scaler_mean, scaler_std, args.seq_len)
    
    # 5. Instantiate Master Ensemble Aggregator
    agg = EnsembleAggregator(config=config)
    
    # Register wrapped models
    agg.register_model("temporal", temporal_wrapper, is_torch=True)
    agg.register_model("maml", maml_wrapper, is_torch=True)
    agg.register_model("regime", regime_wrapper, is_torch=False)
    agg.register_model("rl", rl_wrapper, is_torch=False)

    raise RuntimeError(
        "Legacy ensemble training is disabled: it attempted to fit the stacker on a validation split "
        "rather than purged out-of-fold base-model predictions. Generate a provenance-verified OOF "
        "meta-feature package through the research harness before enabling ensemble training."
    )


if __name__ == "__main__":
    main()
