import os
import sys
import pickle
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Ensure root dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load dataset
features_df = pd.read_csv("data/EUR_USD_features.csv", index_col="timestamp", parse_dates=True)

# Load scaler and model
with open("saved_models/regime_feature_scaler.pkl", "rb") as f:
    regime_scaler = pickle.load(f)
    regime_mean = regime_scaler["mean"]
    regime_std = regime_scaler["std"]

with open("saved_models/regime_ensemble.pkl", "rb") as f:
    state = pickle.load(f)

from models.regime.combined import RegimeEnsembleEstimator
regime_model = RegimeEnsembleEstimator()
regime_model.load("saved_models/regime_ensemble.pkl")

# Prepare features
hmm_features = ["volatility_cc", "mean_reversion_hurst", "trend_strength", "vpin"]
regime_features_df = features_df[hmm_features]
regime_features_scaled = (regime_features_df.values - regime_mean) / regime_std
regime_features_scaled = np.nan_to_num(regime_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

seq_len = 60
windows = sliding_window_view(regime_features_scaled, window_shape=(seq_len, regime_features_scaled.shape[1]))
X_regime = windows.squeeze(1)

probs = regime_model.predict(X_regime, return_proba=True)
preds = np.argmax(probs, axis=-1)

# Analyze last 500 points
last_500_preds = preds[-500:]
unique, counts = np.unique(last_500_preds, return_counts=True)
dist = dict(zip(unique, counts / len(last_500_preds)))

print("Total predictions length:", len(preds))
print("Overall distribution:", pd.Series(preds).value_counts(normalize=True).to_dict())
print("Last 500 distribution:", dist)
print("Last 50 predictions:", preds[-50:].tolist())
