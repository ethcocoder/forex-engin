import os
import pickle
import numpy as np
from typing import Any, Optional, Dict

from models.base_model import BaseModel
from models.regime.hmm import GaussianHMMRegimeEstimator
from models.regime.lstm_classifier import LSTMRegimeClassifier


class RegimeEnsembleEstimator(BaseModel):
    """
    Ensemble estimator fusing Gaussian HMM and LSTM regime classifier.
    Fits HMM on point-in-time features, generates pseudo-labels,
    trains the LSTM sequence classifier, and combines their posterior probabilities.
    """
    def __init__(self, name: str = "regime_ensemble", config: Any = None) -> None:
        super().__init__(name, config)
        
        cfg = config.get("ensemble", {}) if config else {}
        self.w_hmm = cfg.get("w_hmm", 0.5)
        self.w_lstm = cfg.get("w_lstm", 0.5)
        
        # Ensure weights sum to 1.0
        total_w = self.w_hmm + self.w_lstm
        if total_w > 0:
            self.w_hmm /= total_w
            self.w_lstm /= total_w
            
        self.hmm = GaussianHMMRegimeEstimator(name=f"{name}_hmm", config=config)
        self.lstm = LSTMRegimeClassifier(name=f"{name}_lstm", config=config)

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "RegimeEnsembleEstimator":
        """
        Fits the ensemble on sequence inputs X.
        X shape: [n_samples, seq_len, d_feat]
        y is ignored as HMM is unsupervised.
        """
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        n_samples, seq_len, d_feat = X_arr.shape
        
        # 1. Fit HMM on the latest step of the sequence (point-in-time features)
        # X_hmm shape: [n_samples, d_feat]
        X_hmm = X_arr[:, -1, :]
        self.hmm.fit(X_hmm)
        
        # 2. Decode current regime pseudo-labels using Viterbi algorithm
        y_pseudo = self.hmm.predict(X_hmm)
        
        # 3. Fit LSTM classifier on full sequence X and HMM pseudo-labels
        self.lstm.fit(X_arr, y_pseudo)
        
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """
        Predicts combined class label or probability distribution.
        X shape: [n_samples, seq_len, d_feat] or [seq_len, d_feat]
        """
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        
        # Support single sample inputs
        if len(X_arr.shape) == 2:
            X_arr = np.expand_dims(X_arr, axis=0)
            
        n_samples, seq_len, d_feat = X_arr.shape
        
        # Extract last step for HMM
        X_hmm = X_arr[:, -1, :]
        
        # Compute probabilities from HMM and LSTM
        p_hmm = self.hmm.predict(X_hmm, return_proba=True)  # [n_samples, num_classes]
        p_lstm = self.lstm.predict(X_arr, return_proba=True)  # [n_samples, num_classes]
        
        # Combine using weights
        p_combined = self.w_hmm * p_hmm + self.w_lstm * p_lstm
        
        # Normalize to ensure proper probabilities
        eps = 1e-15
        p_combined = np.clip(p_combined, eps, 1.0 - eps)
        p_combined = p_combined / np.sum(p_combined, axis=-1, keepdims=True)
        
        return_proba = kwargs.get("return_proba", False)
        if return_proba:
            return p_combined
        else:
            return np.argmax(p_combined, axis=-1)

    def save(self, path: str, **kwargs: Any) -> None:
        """Saves HMM and LSTM sub-models and ensemble metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hmm_path = path + ".hmm"
        lstm_path = path + ".lstm"
        
        self.hmm.save(hmm_path)
        self.lstm.save(lstm_path)
        
        state = {
            "name": self.name,
            "config": self.config,
            "w_hmm": self.w_hmm,
            "w_lstm": self.w_lstm,
            "hmm_path": hmm_path,
            "lstm_path": lstm_path
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str, **kwargs: Any) -> None:
        """Loads HMM and LSTM sub-models and ensemble metadata."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"State file not found at: {path}")
            
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        self.name = state["name"]
        self.config = state["config"]
        self.w_hmm = state["w_hmm"]
        self.w_lstm = state["w_lstm"]
        
        self.hmm.load(state["hmm_path"])
        self.lstm.load(state["lstm_path"])
