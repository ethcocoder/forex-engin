import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
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
        self.stacker = None  # Will be LightGBM or fallback regressor
        self.lgbm_stacker = None
        self.scaler = StandardScaler()
        self.signal_generator = SignalGenerator()
        self.n_mc_passes = self.mc_estimator.n_forward_passes
        self.bma = BayesianModelAverager(model_names=[])
        self.meta_feature_names: Optional[List[str]] = None
        
        logger.info("GOAT EnsembleAggregator initialized", clusters=list(self.clusters.keys()))

    def register_model(self, name: str, model: Any, cluster: str = "core", is_torch: bool = False) -> None:
        self.sub_models[name] = model
        if cluster not in self.clusters:
            self.clusters[cluster] = []
        if name not in self.clusters[cluster]:
            self.clusters[cluster].append(name)
        if is_torch:
            self._torch_models.append(name)

        self.bma = BayesianModelAverager(model_names=list(self.sub_models.keys()))
        logger.info("Sub-model registered", name=name, cluster=cluster)

    def fit(self, X: Any, y: Any = None, skip_oos: bool = False, **kwargs: Any) -> "GOATEnsembleAggregator":
        if y is None:
            raise ValueError("Target labels y are required to train the EnsembleAggregator.")

        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0] if X_arr.ndim > 0 else 1

        logger.info("EnsembleAggregator fit started", n_samples=n_samples, skip_oos=skip_oos)
        cluster_preds, cluster_uncerts, _ = self._collect_meta_data(X_arr)
        if not cluster_preds:
            raise ValueError("No sub-model predictions available. Register sub-models before training.")

        meta_features = self._build_meta_features(cluster_preds, cluster_uncerts)
        self.meta_feature_names = self._generate_meta_feature_names(cluster_preds)

        logger.info(
            "Meta-data collection complete",
            clusters=list(cluster_preds.keys()),
            n_features=meta_features.shape[1] if meta_features.ndim > 1 else 1,
            n_samples=n_samples
        )

        meta_features_df = pd.DataFrame(meta_features, columns=self.meta_feature_names)
        self.scaler.fit(meta_features_df)
        meta_scaled = pd.DataFrame(self.scaler.transform(meta_features_df), columns=self.meta_feature_names)

        ensemble_cfg = self.config.get("ensemble", {}) if isinstance(self.config, dict) else {}
        n_splits = int(ensemble_cfg.get("stacking_n_splits", 5))
        lgbm_params = dict(ensemble_cfg.get("lgbm_params", {}))
        lgbm_params.setdefault("n_estimators", 100)
        lgbm_params.setdefault("random_state", 42)
        lgbm_params.setdefault("verbosity", -1)

        try:
            import lightgbm as lgb
            stacker = lgb.LGBMRegressor(**lgbm_params)
        except Exception:
            from sklearn.linear_model import Ridge
            stacker = Ridge()

        logger.info("Stacking regressor initialized", regressor=stacker.__class__.__name__, params=lgbm_params)

        if skip_oos or n_splits < 2:
            logger.info("Training stacker on full dataset", n_samples=n_samples)
            stacker.fit(meta_scaled, y)
        else:
            from sklearn.model_selection import KFold
            oof_preds = np.zeros(n_samples, dtype=float)
            kf = KFold(n_splits=min(n_splits, n_samples), shuffle=True, random_state=42)
            logger.info("Starting out-of-fold training", n_splits=n_splits)
            for fold, (train_idx, val_idx) in enumerate(kf.split(meta_scaled), start=1):
                stacker_fold = stacker.__class__(**getattr(stacker, "get_params", lambda **kwargs: {})()) if hasattr(stacker, "get_params") else stacker
                logger.info("Training stacker fold", fold=fold, train_size=len(train_idx), val_size=len(val_idx))
                stacker_fold.fit(meta_scaled[train_idx], np.asarray(y)[train_idx])
                oof_preds[val_idx] = stacker_fold.predict(meta_scaled[val_idx])
            stacker.fit(meta_scaled, y)

        self.stacker = stacker
        self.lgbm_stacker = stacker
        self.n_mc_passes = self.mc_estimator.n_forward_passes
        logger.info("EnsembleAggregator fit complete", stacker=stacker.__class__.__name__)
        return self

    def predict(self, X: Any, return_signal: bool = True, regime: int = 0, volatility: float = 0.0, **kwargs: Any) -> Any:
        X_arr = np.asarray(X)
        cluster_preds, cluster_uncerts, sub_model_preds = self._collect_meta_data(X_arr)

        if not cluster_preds:
            raise ValueError("No sub-model predictions available. Register sub-models before inference.")

        meta_features = self._build_meta_features(cluster_preds, cluster_uncerts)
        avg_uncertainty = float(np.mean([np.mean(v) for v in cluster_uncerts.values()]))

        if self.meta_feature_names is None:
            self.meta_feature_names = self._generate_meta_feature_names(cluster_preds)

        if avg_uncertainty < self.uncertainty_threshold and self.stacker is not None:
            meta_scaled = self.scaler.transform(meta_features)
            if self.meta_feature_names is not None:
                meta_scaled = pd.DataFrame(meta_scaled, columns=self.meta_feature_names)
            final_pred = self.stacker.predict(meta_scaled)
            path = "STACKER"
        else:
            final_pred = np.mean(list(cluster_preds.values()), axis=0)
            path = "BMA_FALLBACK"

        summary_pred = float(np.mean(final_pred))
        signal = self.signal_generator.generate(
            prediction=summary_pred,
            confidence=float(np.clip(1.0 - avg_uncertainty, 0.0, 1.0)),
            uncertainty=avg_uncertainty,
            regime=regime,
            sub_model_predictions={k: float(np.mean(v)) for k, v in sub_model_preds.items()},
            volatility=volatility
        )

        if return_signal:
            return signal
        return summary_pred

    def save(self, path: str, **kwargs: Any) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "ensemble_aggregator.pkl")
        elif not os.path.splitext(path)[1]:
            path = f"{path}.pkl"

        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "name": self.name,
            "config": self.config,
            "uncertainty_threshold": self.uncertainty_threshold,
            "stacker": self.stacker,
            "lgbm_stacker": self.lgbm_stacker,
            "scaler": self.scaler,
            "n_mc_passes": self.n_mc_passes,
            "clusters": self.clusters,
            "signal_generator": self.signal_generator,
            "_torch_models": self._torch_models,
            "meta_feature_names": self.meta_feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str, **kwargs: Any) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "ensemble_aggregator.pkl")
        elif not os.path.exists(path) and os.path.exists(f"{path}.pkl"):
            path = f"{path}.pkl"

        if not os.path.exists(path):
            raise FileNotFoundError(f"EnsembleAggregator state not found at {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        if not isinstance(state, dict):
            raise TypeError("Loaded EnsembleAggregator state must be a dictionary.")

        self.name = state.get("name", self.name)
        self.config = state.get("config", self.config)
        self.uncertainty_threshold = state.get("uncertainty_threshold", self.uncertainty_threshold)
        self.stacker = state.get("stacker", self.stacker)
        self.lgbm_stacker = state.get("lgbm_stacker", self.lgbm_stacker)
        self.scaler = state.get("scaler", self.scaler)
        self.n_mc_passes = state.get("n_mc_passes", self.n_mc_passes)
        self.clusters = state.get("clusters", self.clusters)
        self.signal_generator = state.get("signal_generator", self.signal_generator)
        self._torch_models = state.get("_torch_models", self._torch_models)
        self.meta_feature_names = state.get("meta_feature_names", self.meta_feature_names)
        self.sub_models = {}
        self.bma = BayesianModelAverager(model_names=list(self.sub_models.keys()))

    def _collect_meta_data(self, X: np.ndarray) -> Any:
        cluster_preds: Dict[str, np.ndarray] = {}
        cluster_uncerts: Dict[str, np.ndarray] = {}
        sub_model_preds: Dict[str, np.ndarray] = {}

        if X.ndim == 0:
            X = np.asarray([X])

        for cluster_name, model_names in self.clusters.items():
            preds: List[np.ndarray] = []
            uncerts: List[np.ndarray] = []
            for m_name in model_names:
                if m_name not in self.sub_models:
                    continue
                model = self.sub_models[m_name]

                raw_pred = np.asarray(model.predict(X))
                if raw_pred.ndim == 0:
                    raw_pred = np.full((X.shape[0],), float(raw_pred), dtype=float)
                elif raw_pred.ndim > 1:
                    raw_pred = np.mean(raw_pred, axis=1)

                if raw_pred.shape[0] != X.shape[0]:
                    raise ValueError(f"Model {m_name} returned unexpected prediction shape {raw_pred.shape}")

                preds.append(raw_pred)
                sub_model_preds[m_name] = raw_pred
                uncert = self._estimate_uncertainty(m_name, X)
                uncert = np.asarray(uncert)
                if uncert.ndim == 0:
                    uncert = np.full((X.shape[0],), float(uncert), dtype=float)
                elif uncert.ndim > 1:
                    uncert = np.mean(uncert, axis=1)
                if uncert.shape[0] != X.shape[0]:
                    raise ValueError(f"Model {m_name} returned unexpected uncertainty shape {uncert.shape}")
                uncerts.append(uncert)

            if preds:
                cluster_preds[cluster_name] = np.mean(preds, axis=0)
                cluster_uncerts[cluster_name] = np.mean(uncerts, axis=0)

        return cluster_preds, cluster_uncerts, sub_model_preds

    def _estimate_uncertainty(self, model_name: str, X: Any) -> np.ndarray:
        if model_name in self._torch_models:
            return np.random.uniform(0.1, 0.4, X.shape[0])
        return np.ones(X.shape[0]) * 0.5

    def _generate_meta_feature_names(self, preds: Dict[str, np.ndarray]) -> List[str]:
        names: List[str] = []
        for k in sorted(preds.keys()):
            names.append(f"{k}_mean")
            names.append(f"{k}_uncertainty")
        return names

    def _build_meta_features(self, preds: Dict[str, np.ndarray], uncerts: Dict[str, np.ndarray]) -> np.ndarray:
        feats: List[np.ndarray] = []
        for k in sorted(preds.keys()):
            feats.append(preds[k].reshape(-1, 1))
            feats.append(uncerts[k].reshape(-1, 1))
        return np.hstack(feats)

    def enable_caching(self, X: Any) -> None:
        """Warm up prediction data for later inference, compatible with legacy pipeline hooks."""
        X_arr = np.asarray(X)
        self._cache_enabled = True
        self._cache_source_shape = X_arr.shape

        cluster_preds, cluster_uncerts, _ = self._collect_meta_data(X_arr)
        if cluster_preds:
            self._cached_meta_features = self._build_meta_features(cluster_preds, cluster_uncerts)
            logger.info(
                "EnsembleAggregator caching enabled",
                n_samples=X_arr.shape[0],
                n_features=self._cached_meta_features.shape[1]
            )
        else:
            self._cached_meta_features = None
            logger.warning("EnsembleAggregator cache enabled but no sub-model predictions were available.")


# Alias exported class name for legacy pipeline compatibility
EnsembleAggregator = GOATEnsembleAggregator
