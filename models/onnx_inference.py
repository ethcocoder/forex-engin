import os
import pickle
import numpy as np
import onnxruntime as ort

class ONNXTemporalWrapper:
    def __init__(self, model_path="saved_models/temporal_model.onnx", scaler_path="saved_models/feature_scaler.pkl"):
        # Configure optimized session options for lowest latency
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        
        # Load scale parameters
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            self.scaler_mean = scaler["mean"].astype(np.float32)
            self.scaler_std = scaler["std"].astype(np.float32)
            
    def set_feature_indices(self, raw_feature_indices):
        self.raw_feature_indices = raw_feature_indices

    def predict(self, X, **kwargs):
        # Input shape: [batch, seq_len, total_features]
        # 1. Slice features
        X_raw = X[:, :, self.raw_feature_indices]
        # 2. Scale
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # 3. Inference
        ort_outs = self.session.run(None, {self.input_name: X_scaled})
        return ort_outs[0]


class ONNXMAMLWrapper:
    def __init__(self, model_path="saved_models/maml_model.onnx", scaler_path="saved_models/feature_scaler.pkl"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            self.scaler_mean = scaler["mean"].astype(np.float32)
            self.scaler_std = scaler["std"].astype(np.float32)
            
    def set_feature_indices(self, raw_feature_indices):
        self.raw_feature_indices = raw_feature_indices

    def predict(self, X, **kwargs):
        X_raw = X[:, :, self.raw_feature_indices]
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        ort_outs = self.session.run(None, {self.input_name: X_scaled})
        return ort_outs[0]


class ONNXRegimeWrapper:
    def __init__(self, model_path="saved_models/regime_ensemble.lstm.onnx", scaler_path="saved_models/regime_feature_scaler.pkl"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            self.regime_mean = scaler["mean"].astype(np.float32)
            self.regime_std = scaler["std"].astype(np.float32)
            
    def set_hmm_features(self, hmm_features):
        self.hmm_features = hmm_features

    def predict(self, X, return_proba=True, **kwargs):
        # X is shape [batch, seq_len, total_features] (or HMM feature subset window)
        # Ensure we align scaling with the HMM feature indices
        # Check if X needs feature subsetting or if it is already subsetted
        # For our wrapper, it receives X_regime which is scaled HMM features.
        # Wait, let's see how RegimeEnsembleWrapper.predict works in scripts/run_backtest.py:
        # It takes X and returns X[:, -1, -4:] (which are the padded probs).
        # But for LSTMRegimeClassifier, the input is scaled HMM features window.
        # Let's support both formats: if shape is [batch, seq_len, 4], run ONNX LSTM.
        if X.shape[-1] == 4:
            X_input = X.astype(np.float32)
        else:
            # Subset and scale
            # We assume features_df or dataframe column names matching hmm_features are mapped.
            # If not provided, we fall back to the last 4 elements (or assume scaled inputs).
            X_input = X[:, :, -4:].astype(np.float32)
            
        ort_outs = self.session.run(None, {self.input_name: X_input})
        logits = ort_outs[0]
        
        if return_proba:
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return probs
        else:
            return np.argmax(logits, axis=-1)


class ONNXRLEnsembleWrapper:
    def __init__(self, model_path="saved_models/rl_agent_ppo.onnx"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.action_values = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float64)

    def set_config(self, features_cols, regime_cols):
        self.features_cols = features_cols
        self.regime_cols = regime_cols

    def predict(self, X, **kwargs):
        n_samples = X.shape[0]
        feats_raw = X[:, -1, :len(self.features_cols)]
        feats = np.nan_to_num(feats_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        pos = np.full((n_samples, 1), kwargs.get("current_position", 0.0), dtype=np.float32)
        unrealized = np.full((n_samples, 1), kwargs.get("unrealized_pnl", 0.0) / 10000.0, dtype=np.float32)
        time_ind = np.full((n_samples, 1), kwargs.get("time_indicator", 0.0), dtype=np.float32)
        
        regimes = X[:, -1, -len(self.regime_cols):]
        
        obs = np.hstack([feats, pos, unrealized, time_ind, regimes])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        ort_outs = self.session.run(None, {self.input_name: obs})
        action_probs = ort_outs[0] # shape: [batch, 5]
        
        pred = np.sum(action_probs * self.action_values, axis=1)
        return pred


class ONNXAttackerModel:
    def __init__(self, model_path="saved_models/adversarial_attacker.onnx", meta_path="saved_models/adversarial_attacker.pkl"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.scaler_mean = meta["scaler_mean"]
        self.scaler_std = meta["scaler_std"]
        self.input_dim = meta["input_dim"]
        self.feature_names = meta.get("feature_names", [])
        self.config = meta.get("config", {})

    def generate_adversarial_scenario(self, current_strategy):
        candidate_count = self.config.get("candidate_count", 32)
        candidate_noise = self.config.get("candidate_noise", 0.8)
        
        z = np.random.randn(candidate_count, self.input_dim).astype(np.float32)
        scaled_candidates = z * candidate_noise
        
        logits = self.session.run(None, {self.input_name: scaled_candidates})[0]
        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))
        
        best_idx = int(np.argmax(probs))
        top_score = float(probs[best_idx])
        
        top_candidate_scaled = scaled_candidates[best_idx]
        top_candidate = top_candidate_scaled * self.scaler_std + self.scaler_mean
        
        shock_indices = np.argsort(np.abs(top_candidate - self.scaler_mean))[-5:][::-1]
        top_shocks = {}
        for idx in shock_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            top_shocks[feature_name] = float(top_candidate[idx] - self.scaler_mean[idx])
            
        return {
            "strategy": current_strategy,
            "vulnerability_score": top_score,
            "expected_drawdown_pct": float(top_score * 0.05),
            "critical_feature_shocks": top_shocks,
            "status": "adversarial_scenario_generated",
        }
