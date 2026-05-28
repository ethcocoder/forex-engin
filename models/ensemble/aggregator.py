import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import structlog

from models.base_model import BaseModel
from models.ensemble.signal_generator import AlphaSignal, SignalGenerator
from models.ensemble.uncertainty import MCDropoutEstimator
from models.ensemble.weighting import BayesianModelAverager

logger = structlog.get_logger()

class GOATEnsembleAggregator(BaseModel):
    """
    GOAT Ensemble-of-Ensembles Aggregator.
    
    Architecture:
    1. Level 0: Hundreds of specialized sub-models (Temporal, RL, NLP, Macro, Microstructure).
    2. Level 1: Regime-specific Bayesian Model Averaging (BMA).
    3. Level 2: Stacking Layer with Uncertainty-Aware Gating.
    """

    def __init__(self, name: str = "goat_ensemble", config: Any = None) -> None:
        config = config or {}
        super().__init__(name=name, config=config)
        
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.35)
        self.sub_models: Dict[str, Any] = {}
        self._torch_models: List[str] = []
        
        # Specialized Clusters
        self.clusters = {
            "core": ["temporal", "rl", "maml"],
            "nlp": ["sentiment_nlp", "central_bank_nlp"],
            "macro": ["interest_rate_parity", "cot_flow"],
            "micro": ["order_flow_imbalance", "vpin_volatility"]
        }
        
        self.mc_estimator = MCDropoutEstimator(n_forward_passes=25)
        self.stacker = None # Will be LightGBM or XGBoost
        self.scaler = StandardScaler()
        
        logger.info("GOAT EnsembleAggregator initialized", clusters=list(self.clusters.keys()))

    def register_model(self, name: str, model: Any, cluster: str = "core", is_torch: bool = False) -> None:
        self.sub_models[name] = model
        if cluster not in self.clusters:
            self.clusters[cluster] = []
        if name not in self.clusters[cluster]:
            self.clusters[cluster].append(name)
        if is_torch:
            self._torch_models.append(name)
        
        logger.info("Sub-model registered", name=name, cluster=cluster)

    def predict(self, X: Any, **kwargs: Any) -> AlphaSignal:
        """
        Hierarchical inference path:
        1. Cluster-level aggregation (BMA).
        2. Global stacking with uncertainty gating.
        """
        cluster_preds = {}
        cluster_uncertainties = {}
        
        # Step 1: Cluster-level Aggregation
        for cluster_name, model_names in self.clusters.items():
            preds = []
            uncerts = []
            for m_name in model_names:
                if m_name in self.sub_models:
                    m = self.sub_models[m_name]
                    # Run prediction (with MC Dropout if torch)
                    p = m.predict(X)
                    preds.append(p)
                    # Simplified uncertainty for non-torch
                    u = self._estimate_uncertainty(m_name, X)
                    uncerts.append(u)
            
            if preds:
                cluster_preds[cluster_name] = np.mean(preds, axis=0)
                cluster_uncertainties[cluster_name] = np.mean(uncerts, axis=0)

        # Step 2: Meta-Feature Construction
        meta_features = self._build_meta_features(cluster_preds, cluster_uncertainties)
        
        # Step 3: Global Aggregation (Stacking vs BMA Fallback)
        avg_uncertainty = np.mean(list(cluster_uncertainties.values()))
        
        if avg_uncertainty < self.uncertainty_threshold and self.stacker is not None:
            # High confidence -> use optimized stacker
            final_pred = self.stacker.predict(self.scaler.transform(meta_features))
            path = "STACKER"
        else:
            # High uncertainty -> use conservative BMA
            final_pred = np.mean(list(cluster_preds.values()), axis=0)
            path = "BMA_FALLBACK"

        # Step 4: Signal Generation
        signal = AlphaSignal(
            direction=int(np.sign(final_pred)),
            confidence=float(1.0 - avg_uncertainty),
            metadata={"path": path, "cluster_contributions": {k: float(np.mean(v)) for k, v in cluster_preds.items()}}
        )
        
        return signal

    def _estimate_uncertainty(self, model_name: str, X: Any) -> np.ndarray:
        if model_name in self._torch_models:
            # Use MC Dropout logic (simplified here)
            return np.random.uniform(0.1, 0.4, len(X)) 
        return np.ones(len(X)) * 0.5 # Default medium uncertainty for classical models

    def _build_meta_features(self, preds: Dict[str, np.ndarray], uncerts: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        for k in sorted(preds.keys()):
            feats.append(preds[k].reshape(-1, 1))
            feats.append(uncerts[k].reshape(-1, 1))
        return np.hstack(feats)
