import os
import pickle
from typing import Any, Optional, Dict
import numpy as np
from hmmlearn.hmm import GaussianHMM

from models.base_model import BaseModel


class GaussianHMMRegimeEstimator(BaseModel):
    """
    Gaussian Hidden Markov Model (HMM) for dynamic market regime classification.
    Fits unsupervised states (e.g. 4 states) based on volatility and momentum features,
    and decodes latent states using the Viterbi algorithm.
    """
    def __init__(self, name: str = "hmm_regime", config: Any = None) -> None:
        super().__init__(name, config)
        
        cfg = config.get("hmm", {}) if config else {}
        self.n_components = cfg.get("n_components", 4)
        self.covariance_type = cfg.get("covariance_type", "full")
        self.n_iter = cfg.get("n_iter", 100)
        self.tol = cfg.get("tol", 1e-2)
        self.random_state = cfg.get("random_state", 42)
        
        self.model: Optional[GaussianHMM] = None
        
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "GaussianHMMRegimeEstimator":
        """
        Fits the unsupervised Gaussian HMM on features X.
        X shape: [n_samples, n_features]
        y is ignored (unsupervised).
        """
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        
        # Instantiate GaussianHMM
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state
        )
        
        self.model.fit(X_arr)
        
        # Sort states by volatility to ensure consistency
        self._align_states()
        
        return self

    def _align_states(self) -> None:
        """
        Sort HMM states by the trace or mean of their covariances (volatility),
        ensuring state 0 is low volatility and state (n_components-1) is high volatility.
        """
        if self.model is None:
            return
            
        covs = self.model.covars_
        vol_metric = []
        for i in range(self.n_components):
            c = covs[i]
            if self.covariance_type == "full":
                vol = np.trace(c)
            elif self.covariance_type in ("diag", "spherical"):
                vol = np.sum(c)
            else:
                vol = np.sum(np.abs(c))
            vol_metric.append(vol)
            
        # Get sorted order of states
        sort_order = np.argsort(vol_metric)
        
        # Permute HMM model parameters according to sorted order
        self.model.startprob_ = self.model.startprob_[sort_order]
        self.model.transmat_ = self.model.transmat_[sort_order][:, sort_order]
        self.model.means_ = self.model.means_[sort_order]
        self.model.covars_ = self.model.covars_[sort_order]

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        """
        Decodes the current regime state sequence or returns state probabilities.
        By default, decodes states using the Viterbi algorithm.
        If return_proba=True, returns posterior probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
            
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        
        return_proba = kwargs.get("return_proba", False)
        if return_proba:
            return self.model.predict_proba(X_arr)
        else:
            return self.model.predict(X_arr)

    def get_transition_matrix(self) -> np.ndarray:
        """
        Returns the fitted state transition matrix.
        Shape: [n_components, n_components]
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        return self.model.transmat_

    def save(self, path: str, **kwargs: Any) -> None:
        """Serializes HMM state and weights using pickle."""
        state = {
            "name": self.name,
            "config": self.config,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "model": self.model
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str, **kwargs: Any) -> None:
        """Loads HMM state and weights from a pickled state file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickled HMM file not found at: {path}")
            
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        self.name = state["name"]
        self.config = state["config"]
        self.n_components = state["n_components"]
        self.covariance_type = state["covariance_type"]
        self.n_iter = state["n_iter"]
        self.tol = state["tol"]
        self.random_state = state["random_state"]
        self.model = state["model"]
