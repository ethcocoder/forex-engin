import os
import sys
import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from models.onnx_inference import ONNXTemporalWrapper, ONNXMAMLWrapper

# Load feature scaler
with open("saved_models/feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    scaler_mean = scaler["mean"]
    scaler_std = scaler["std"]

# Load features data
features_df = pd.read_csv("data/EUR_USD_features.csv", index_col="timestamp", parse_dates=True)
exclude = ["timestamp", "close", "open", "high", "low", "volume", "bid", "ask", "regime_0", "regime_1", "regime_2", "regime_3"]
features_cols = [col for col in features_df.columns if col not in exclude]
raw_feature_indices = [features_df.columns.get_loc(col) for col in features_cols]

# Create wrappers
orig_temp = ONNXTemporalWrapper(model_path="saved_models/temporal_model.onnx")
orig_temp.set_feature_indices(raw_feature_indices)

quant_temp = ONNXTemporalWrapper(model_path="saved_models/quantized/temporal_model_int8.onnx")
quant_temp.set_feature_indices(raw_feature_indices)

orig_maml = ONNXMAMLWrapper(model_path="saved_models/maml_model.onnx")
orig_maml.set_feature_indices(raw_feature_indices)

quant_maml = ONNXMAMLWrapper(model_path="saved_models/quantized/maml_model_int8.onnx")
quant_maml.set_feature_indices(raw_feature_indices)

# Check predictions on a few indices
features_arr = features_df.copy().values
features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

print("Comparing 5 samples:")
for idx in [100, 500, 1000, 5000, 10000]:
    X_input = features_arr[idx - 60 + 1 : idx + 1]
    X_input = np.expand_dims(X_input, axis=0)
    
    p_orig_temp = orig_temp.predict(X_input)
    p_quant_temp = quant_temp.predict(X_input)
    
    p_orig_maml = orig_maml.predict(X_input)
    p_quant_maml = quant_maml.predict(X_input)
    
    print(f"Sample {idx}:")
    print(f"  Temporal - Orig: {p_orig_temp[0]:.6f}, Quant: {p_quant_temp[0]:.6f}, AbsDiff: {abs(p_orig_temp[0]-p_quant_temp[0]):.6f}")
    print(f"  MAML     - Orig: {p_orig_maml[0]:.6f}, Quant: {p_quant_maml[0]:.6f}, AbsDiff: {abs(p_orig_maml[0]-p_quant_maml[0]):.6f}")
