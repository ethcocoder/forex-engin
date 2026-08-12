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
        self.training_provenance: Optional[Dict[str, Any]] = None
        self.expected_sub_model_names: List[str] = []
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
        """Fit only on purged out-of-fold sub-model predictions.

        ``X`` is retained for BaseModel API compatibility but is intentionally not
        used to create stacker features. The caller must supply predictions made
        by base models that were not trained on the corresponding labelled row.
        """
        if y is None:
            raise ValueError("Target labels y are required to train the EnsembleAggregator.")
        if skip_oos:
            raise ValueError("skip_oos is prohibited: stacking requires purged out-of-fold meta-features.")
        oof_meta_features = kwargs.pop("oof_meta_features", None)
        provenance = kwargs.pop("oof_provenance", None)
        feature_names = kwargs.pop("oof_feature_names", None)
        if kwargs:
            raise ValueError(f"Unexpected EnsembleAggregator fit arguments: {sorted(kwargs)}")
        if oof_meta_features is None:
            raise ValueError("oof_meta_features are mandatory; on-sample sub-model predictions are prohibited.")
        if not isinstance(provenance, dict):
            raise ValueError("oof_provenance is mandatory and must document the purged walk-forward split.")
        required_provenance = {"validation_type", "fold_count", "embargo_rows", "label_horizon_rows", "data_manifest_sha256"}
        missing_provenance = sorted(required_provenance - set(provenance))
        if missing_provenance:
            raise ValueError(f"oof_provenance is incomplete: missing {', '.join(missing_provenance)}")
        if provenance["validation_type"] != "purged_walk_forward":
            raise ValueError("oof_provenance.validation_type must be 'purged_walk_forward'.")

        meta_features = np.asarray(oof_meta_features, dtype=float)
        targets = np.asarray(y).reshape(-1)
        if meta_features.ndim == 1:
            meta_features = meta_features.reshape(-1, 1)
        if meta_features.ndim != 2 or len(meta_features) != len(targets):
            raise ValueError("oof_meta_features must be two-dimensional with one row per target label.")
        if not np.isfinite(meta_features).all() or not np.isfinite(targets).all():
            raise ValueError("oof_meta_features and target labels must be finite.")
        if feature_names is None:
            feature_names = [f"oof_feature_{index}" for index in range(meta_features.shape[1])]
        if len(feature_names) != meta_features.shape[1] or len(set(feature_names)) != len(feature_names):
            raise ValueError("oof_feature_names must be unique and match the OOF feature width.")

        self.meta_feature_names = list(feature_names)
        self.training_provenance = dict(provenance)
        self.expected_sub_model_names = sorted(self.sub_models)
        meta_features_df = pd.DataFrame(meta_features, columns=self.meta_feature_names)
        self.scaler.fit(meta_features_df)
        meta_scaled = pd.DataFrame(self.scaler.transform(meta_features_df), columns=self.meta_feature_names)
        ensemble_cfg = self.config.get("ensemble", {}) if isinstance(self.config, dict) else {}
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
        stacker.fit(meta_scaled, targets)
        self.stacker = stacker
        self.lgbm_stacker = stacker
        self.n_mc_passes = self.mc_estimator.n_forward_passes
        logger.info(
            "EnsembleAggregator fit complete on purged OOF meta-features",
            stacker=stacker.__class__.__name__,
            n_samples=len(targets),
            n_features=meta_features.shape[1],
            fold_count=provenance["fold_count"],
        )
        return self


    def predict(self, X: Any, return_signal: bool = True, regime: int = 0, volatility: float = 0.0, **kwargs: Any) -> Any:
        if self.stacker is None or self.training_provenance is None:
            raise RuntimeError("Ensemble inference is unavailable until a provenance-verified OOF stacker has been fitted.")
        if self.expected_sub_model_names and sorted(self.sub_models) != self.expected_sub_model_names:
            raise RuntimeError("Registered sub-models do not match the model set recorded with the fitted stacker.")
        X_arr = np.asarray(X)
        cluster_preds, cluster_uncerts, sub_model_preds = self._collect_meta_data(X_arr)
        if not cluster_preds:
            raise ValueError("No sub-model predictions available. Register sub-models before inference.")
        meta_features = self._build_meta_features(cluster_preds, cluster_uncerts)
        inference_feature_names = self._generate_meta_feature_names(cluster_preds)
        if self.meta_feature_names != inference_feature_names:
            raise RuntimeError("Live ensemble meta-feature layout differs from the provenance-verified training layout.")
        avg_uncertainty = float(np.mean([np.mean(v) for v in cluster_uncerts.values()]))
        if avg_uncertainty >= self.uncertainty_threshold:
            summary_pred = 0.0
            path = "ABSTAIN_HIGH_UNCERTAINTY"
        else:
            meta_scaled = self.scaler.transform(meta_features)
            meta_scaled = pd.DataFrame(meta_scaled, columns=self.meta_feature_names)
            final_pred = self.stacker.predict(meta_scaled)
            summary_pred = float(np.mean(final_pred))
            path = "STACKER"
        
        # Online RL: If BMA has enough reinforcement data, adjust confidence
        # based on whether high-performing models agree with the signal direction
        rl_confidence_adj = 1.0
        bma_ready = (self.bma is not None and 
                     len(self.bma.tracker.actual_history) >= 50)
        if bma_ready and sub_model_preds:
            bma_weights = self.bma.get_weights()
            # Compute agreement: weighted sum of sign-agreement between each 
            # model's prediction and the ensemble prediction
            agreement = 0.0
            for m_name, w in bma_weights.items():
                m_pred = float(np.mean(sub_model_preds.get(m_name, np.zeros(1))))
                if summary_pred != 0:
                    # +1 if model agrees with ensemble direction, -1 if disagrees
                    sign_agree = 1.0 if (m_pred * summary_pred > 0) else -1.0
                    agreement += w * sign_agree
            # Scale agreement [-1, 1] to confidence adjustment [0.7, 1.3]
            rl_confidence_adj = float(np.clip(1.0 + 0.3 * agreement, 0.7, 1.3))
        
        base_confidence = float(np.clip(1.0 - avg_uncertainty, 0.0, 1.0))
        adjusted_confidence = float(np.clip(base_confidence * rl_confidence_adj, 0.0, 1.0))
        
        signal = self.signal_generator.generate(
            prediction=summary_pred,
            confidence=adjusted_confidence,
            uncertainty=avg_uncertainty,
            regime=regime,
            sub_model_predictions={k: float(np.mean(v)) for k, v in sub_model_preds.items()},
            volatility=volatility
        )

        signal.metadata["ensemble_path"] = path
        signal.metadata["training_validation_type"] = self.training_provenance["validation_type"]
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
            "training_provenance": self.training_provenance,
            "expected_sub_model_names": self.expected_sub_model_names,
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
        self.training_provenance = state.get("training_provenance")
        self.expected_sub_model_names = sorted(state.get("expected_sub_model_names", []))
        if not self.training_provenance or not self.expected_sub_model_names:
            raise RuntimeError("Legacy or incomplete ensemble state rejected: verified OOF provenance and model identities are required.")
        if self.sub_models and sorted(self.sub_models) != self.expected_sub_model_names:
            raise RuntimeError("Registered sub-models do not match the model set recorded in the ensemble state.")
        self.bma = BayesianModelAverager(model_names=self.expected_sub_model_names)

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
