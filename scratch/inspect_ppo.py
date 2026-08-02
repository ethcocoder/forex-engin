import os
import sys

# Limit CPU threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rl_agent.ppo_agent import PPOModel

# Load PPO Model
rl_model = PPOModel(name="ppo_agent", config={"device": "cpu"})
rl_model.load("saved_models/rl_agent_ppo.zip")

# Load features and scalers
features_df = pd.read_csv("data/EUR_USD_features.csv", index_col="timestamp", parse_dates=True)
with open("saved_models/feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    scaler_mean = scaler["mean"]
    scaler_std = scaler["std"]

# Select feature columns (excluding timestamp, close, etc.)
exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask"]
features_cols = [col for col in features_df.columns if col not in exclude and not col.startswith("regime_")]

features_df_clean = features_df.dropna(subset=features_cols)
raw_features = features_df_clean[features_cols].values[100:120]  # shape: [20, d_feat]
scaled_features = (raw_features - scaler_mean) / scaler_std

# Check for nan/inf in scaled features
print("Are there NaNs in raw_features?", np.isnan(raw_features).any())
print("Are there NaNs in scaled_features?", np.isnan(scaled_features).any())
print("Are there Infs in scaled_features?", np.isinf(scaled_features).any())

dummy_regime = np.zeros((20, 4))
dummy_regime[:, 3] = 1.0  # Regime 3

pos = np.zeros((20, 1))
unrealized = np.zeros((20, 1))
time_ind = np.zeros((20, 1))

obs_raw = np.hstack([raw_features, pos, unrealized, time_ind, dummy_regime]).astype(np.float32)

print("\nTesting get_distribution...")
obs_tensor = torch.as_tensor(obs_raw, device=rl_model.model.device)
distribution = rl_model.model.policy.get_distribution(obs_tensor)
probs = distribution.distribution.probs.cpu().numpy()
print("Probs shape:", probs.shape)
print("First prob vector:", probs[0])
