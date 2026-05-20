import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import structlog

from models.base_model import BaseModel
from models.ensemble.signal_generator import AlphaSignal, SignalGenerator
from models.ensemble.uncertainty import MCDropoutEstimator
from models.ensemble.weighting import BayesianModelAverager

logger = structlog.get_logger()


class EnsembleAggregator(BaseModel):
    """
    Master ensemble aggregator that unifies predictions from all four sub-models
    (Temporal, Regime, RL Agent, Meta-Learner) into a single AlphaSignal.

    Dual-mode inference path:
        1. Low uncertainty → LightGBM stacking prediction (higher alpha capture).
        2. High uncertainty → Bayesian Model Averaging fallback (safer, conservative).

    The stacking layer is trained on purged out-of-sample predictions from each
    sub-model to prevent information leakage during ensemble training.
    """

    def __init__(self, name: str = "ensemble", config: Any = None) -> None:
        config = config or {}
        super().__init__(name=name, config=config)

        ensemble_cfg = config.get("ensemble", {})

        # Hyperparameters
        self.uncertainty_threshold = ensemble_cfg.get("uncertainty_threshold", 0.3)
        self.direction_threshold = ensemble_cfg.get("direction_threshold", 0.002)
        self.n_mc_passes = ensemble_cfg.get("n_mc_passes", 30)
        self.stacking_n_splits = ensemble_cfg.get("stacking_n_splits", 4)
        self.lgbm_params = ensemble_cfg.get("lgbm_params", {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1
        })

        # Sub-model registry
        self.sub_models: Dict[str, Any] = {}

        # Components (initialized lazily or during fit)
        self.lgbm_stacker = None
        self.mc_estimator = MCDropoutEstimator(n_forward_passes=self.n_mc_passes)
        self.bma: Optional[BayesianModelAverager] = None
        self.signal_generator = SignalGenerator(
            direction_threshold=self.direction_threshold
        )

        # Track which models are PyTorch-based for MC Dropout
        self._torch_models: List[str] = []

        # Feature column names for the stacking layer
        self._meta_feature_names: List[str] = []

        logger.info(
            "EnsembleAggregator initialized",
            uncertainty_threshold=self.uncertainty_threshold,
            n_mc_passes=self.n_mc_passes,
            stacking_n_splits=self.stacking_n_splits
        )

    def register_model(self, name: str, model: Any, is_torch: bool = False) -> None:
        """
        Register a sub-model with the ensemble.

        Args:
            name: Unique identifier for this sub-model (e.g., 'temporal', 'regime', 'rl', 'maml').
            model: The sub-model instance. Must implement predict(X).
            is_torch: Whether this model wraps a PyTorch nn.Module (enables MC Dropout).
        """
        self.sub_models[name] = model
        if is_torch:
            self._torch_models.append(name)

        logger.info(
            "Sub-model registered with ensemble",
            model_name=name,
            is_torch=is_torch,
            total_models=len(self.sub_models)
        )

    def _collect_predictions(
        self,
        X: np.ndarray,
        with_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Collect predictions from all registered sub-models.

        Args:
            X: Input features [n_samples, seq_len, d_feat].
            with_uncertainty: Whether to run MC Dropout on torch models.

        Returns:
            Dict with keys 'predictions' (model_name -> np.ndarray),
            'uncertainties' (model_name -> np.ndarray), and
            'mean_uncertainty' (float).
        """
        predictions = {}
        uncertainties = {}

        for name, model in self.sub_models.items():
            try:
                if with_uncertainty and name in self._torch_models:
                    # MC Dropout for uncertainty estimation
                    inner_model = getattr(model, "model", None)
                    if inner_model is not None and isinstance(inner_model, torch.nn.Module):
                        device = next(inner_model.parameters()).device
                        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                        mean_pred, uncertainty = self.mc_estimator.estimate(inner_model, X_tensor)
                        predictions[name] = mean_pred
                        uncertainties[name] = uncertainty
                    else:
                        pred = model.predict(X)
                        predictions[name] = np.atleast_1d(np.asarray(pred, dtype=np.float64))
                        uncertainties[name] = np.zeros_like(predictions[name])
                else:
                    pred = model.predict(X)
                    predictions[name] = np.atleast_1d(np.asarray(pred, dtype=np.float64))
                    uncertainties[name] = np.zeros_like(predictions[name])

            except Exception as e:
                logger.warning(
                    "Sub-model prediction failed, using zero fallback",
                    model_name=name,
                    error=str(e)
                )
                n = X.shape[0] if hasattr(X, "shape") else 1
                predictions[name] = np.zeros(n)
                uncertainties[name] = np.ones(n)

        # Mean uncertainty across all torch models
        torch_uncerts = [uncertainties[n] for n in self._torch_models if n in uncertainties]
        if torch_uncerts:
            mean_uncertainty = float(np.mean([np.mean(u) for u in torch_uncerts]))
        else:
            mean_uncertainty = 0.0

        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "mean_uncertainty": mean_uncertainty
        }

    def _build_meta_features(
        self,
        predictions: Dict[str, np.ndarray],
        uncertainties: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Build the meta-feature matrix for the stacking layer.

        Columns: [model1_pred, model2_pred, ..., model1_uncert, model2_uncert, ...]

        Returns:
            np.ndarray of shape [n_samples, n_meta_features].
        """
        feature_arrays = []
        feature_names = []

        # Predictions from each model
        for name in sorted(self.sub_models.keys()):
            if name in predictions:
                pred = predictions[name]
                # Handle multi-dimensional predictions (e.g., regime probabilities)
                if pred.ndim == 1:
                    feature_arrays.append(pred.reshape(-1, 1))
                    feature_names.append(f"{name}_pred")
                else:
                    feature_arrays.append(pred)
                    for col_idx in range(pred.shape[1]):
                        feature_names.append(f"{name}_pred_{col_idx}")

        # Uncertainties from torch models
        for name in sorted(self._torch_models):
            if name in uncertainties:
                unc = uncertainties[name]
                if unc.ndim == 1:
                    feature_arrays.append(unc.reshape(-1, 1))
                    feature_names.append(f"{name}_uncertainty")

        if not feature_arrays:
            return np.zeros((1, 1))

        self._meta_feature_names = feature_names
        return np.hstack(feature_arrays)

    def _generate_oos_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int
    ) -> tuple:
        """
        Generate purged out-of-sample predictions using walk-forward splits.

        For each split:
            1. Train sub-models on training portion.
            2. Predict on validation portion.
            3. Collect meta-features and aligned targets.

        Args:
            X: Input features [n_samples, seq_len, d_feat].
            y: Target returns [n_samples].
            n_splits: Number of walk-forward splits.

        Returns:
            Tuple of (meta_features: np.ndarray, aligned_y: np.ndarray).
        """
        n_samples = X.shape[0]
        fold_size = n_samples // (n_splits + 1)
        purge_gap = max(1, fold_size // 20)  # 5% purge

        all_meta_features = []
        all_targets = []

        for split_idx in range(n_splits):
            train_end = fold_size * (split_idx + 1)
            val_start = train_end + purge_gap  # Purge gap
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples or val_end <= val_start:
                continue

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            # Train all sub-models on training data
            for name, model in self.sub_models.items():
                try:
                    model.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(
                        "Sub-model training failed in OOS generation",
                        model_name=name,
                        split=split_idx,
                        error=str(e)
                    )

            # Collect validation predictions
            result = self._collect_predictions(X_val, with_uncertainty=True)
            meta_features = self._build_meta_features(
                result["predictions"],
                result["uncertainties"]
            )

            all_meta_features.append(meta_features)
            all_targets.append(y_val)

            logger.info(
                "OOS fold completed",
                split=split_idx + 1,
                train_size=len(y_train),
                val_size=len(y_val),
                n_features=meta_features.shape[1]
            )

        if not all_meta_features:
            raise ValueError("No valid OOS folds were generated.")

        return np.vstack(all_meta_features), np.concatenate(all_targets)

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> "EnsembleAggregator":
        """
        Train the ensemble stacking layer on purged out-of-sample predictions.

        Args:
            X: Input features [n_samples, seq_len, d_feat].
            y: Target forward returns [n_samples].
            **kwargs: Additional arguments (e.g., skip_oos=True to train on direct predictions).

        Returns:
            self
        """
        import lightgbm as lgb

        if y is None:
            raise ValueError("EnsembleAggregator.fit() requires target values y.")

        if len(self.sub_models) == 0:
            raise ValueError("No sub-models registered. Call register_model() first.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)

        logger.info(
            "Beginning ensemble stacking training",
            n_samples=X.shape[0],
            n_sub_models=len(self.sub_models),
            n_splits=self.stacking_n_splits
        )

        skip_oos = kwargs.get("skip_oos", False)

        if skip_oos:
            # Direct prediction mode (for testing / small datasets)
            for name, model in self.sub_models.items():
                try:
                    model.fit(X, y)
                except Exception as e:
                    logger.warning("Sub-model fit failed", model=name, error=str(e))

            result = self._collect_predictions(X, with_uncertainty=True)
            meta_features = self._build_meta_features(
                result["predictions"],
                result["uncertainties"]
            )
            aligned_y = y
        else:
            # Purged OOS prediction generation
            meta_features, aligned_y = self._generate_oos_predictions(
                X, y, self.stacking_n_splits
            )

        # Train LightGBM stacking meta-model
        self.lgbm_stacker = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgbm_stacker.fit(meta_features, aligned_y)

        logger.info(
            "LightGBM stacking meta-model trained",
            n_meta_features=meta_features.shape[1],
            n_training_samples=meta_features.shape[0],
            feature_names=self._meta_feature_names
        )

        # Initialize BMA with sub-model names
        self.bma = BayesianModelAverager(
            model_names=list(self.sub_models.keys())
        )

        # Retrain all sub-models on full dataset for inference
        if not skip_oos:
            for name, model in self.sub_models.items():
                try:
                    model.fit(X, y)
                except Exception as e:
                    logger.warning("Sub-model full retraining failed", model=name, error=str(e))

        logger.info("Ensemble stacking training completed successfully")
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Run ensemble inference with automatic uncertainty gating.

        Pipeline:
            1. Collect predictions from all sub-models.
            2. Run MC Dropout on PyTorch-based models for uncertainty.
            3. If uncertainty < threshold: use LightGBM stacking.
            4. If uncertainty >= threshold: fall back to BMA weighted average.
            5. Generate AlphaSignal.

        Args:
            X: Input features [n_samples, seq_len, d_feat] or [seq_len, d_feat].
            **kwargs:
                return_signal (bool): If True, return AlphaSignal. Default True.
                regime (int): Current regime state for signal generation. Default 0.

        Returns:
            AlphaSignal if return_signal=True, else raw float prediction.
        """
        return_signal = kwargs.get("return_signal", True)
        regime = kwargs.get("regime", 0)

        X = np.asarray(X, dtype=np.float32)
        single_sample = False
        if X.ndim == 2:
            X = X[np.newaxis, :]
            single_sample = True

        # Collect predictions and uncertainties
        result = self._collect_predictions(X, with_uncertainty=True)
        predictions = result["predictions"]
        uncertainties = result["uncertainties"]
        mean_uncertainty = result["mean_uncertainty"]

        # Build meta-feature vector
        meta_features = self._build_meta_features(predictions, uncertainties)

        # Dual-mode inference
        if mean_uncertainty < self.uncertainty_threshold and self.lgbm_stacker is not None:
            # Low uncertainty: use LightGBM stacking
            ensemble_prediction = self.lgbm_stacker.predict(meta_features)
            inference_mode = "stacking"
        else:
            # High uncertainty: BMA fallback
            if self.bma is not None:
                # Average over samples
                ensemble_prediction = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    sample_preds = {
                        name: float(predictions[name][i])
                        for name in predictions
                        if predictions[name].ndim == 1
                    }
                    ensemble_prediction[i] = self.bma.average(sample_preds)
            else:
                # Final fallback: simple average
                pred_values = [predictions[n] for n in predictions if predictions[n].ndim == 1]
                if pred_values:
                    ensemble_prediction = np.mean(np.stack(pred_values, axis=0), axis=0)
                else:
                    ensemble_prediction = np.zeros(X.shape[0])
            inference_mode = "bma"

        logger.debug(
            "Ensemble inference completed",
            mode=inference_mode,
            mean_uncertainty=mean_uncertainty,
            threshold=self.uncertainty_threshold
        )

        if single_sample:
            pred_val = float(ensemble_prediction[0])
        else:
            pred_val = float(np.mean(ensemble_prediction))

        if not return_signal:
            return pred_val

        # Compute confidence from ensemble agreement
        pred_values_list = [
            float(np.mean(predictions[n]))
            for n in predictions
            if predictions[n].ndim == 1
        ]
        if len(pred_values_list) >= 2:
            signs = [1 if p > 0 else (-1 if p < 0 else 0) for p in pred_values_list]
            agreement = sum(1 for s in signs if s == (1 if pred_val > 0 else -1)) / len(signs)
            confidence = float(np.clip(agreement, 0.0, 1.0))
        else:
            confidence = 0.5

        sub_model_preds = {
            name: float(np.mean(predictions[name]))
            for name in predictions
        }

        signal = self.signal_generator.generate(
            prediction=pred_val,
            confidence=confidence,
            uncertainty=mean_uncertainty,
            regime=regime,
            sub_model_predictions=sub_model_preds
        )

        return signal

    def save(self, path: str, **kwargs: Any) -> None:
        """
        Serialize the ensemble aggregator state.

        Saves:
            - LightGBM stacker model (via joblib)
            - BMA state (via pickle)
            - Configuration and metadata
        """
        import joblib

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        state = {
            "name": self.name,
            "config": self.config,
            "uncertainty_threshold": self.uncertainty_threshold,
            "direction_threshold": self.direction_threshold,
            "n_mc_passes": self.n_mc_passes,
            "stacking_n_splits": self.stacking_n_splits,
            "lgbm_params": self.lgbm_params,
            "meta_feature_names": self._meta_feature_names,
            "torch_models": self._torch_models,
            "sub_model_names": list(self.sub_models.keys())
        }

        # Save metadata
        with open(path + ".meta", "wb") as f:
            pickle.dump(state, f)

        # Save LightGBM model
        if self.lgbm_stacker is not None:
            joblib.dump(self.lgbm_stacker, path + ".lgbm")

        # Save BMA state
        if self.bma is not None:
            with open(path + ".bma", "wb") as f:
                pickle.dump({
                    "model_names": self.bma.model_names,
                    "temperature": self.bma.temperature
                }, f)

        logger.info("EnsembleAggregator saved successfully", destination=path)

    def load(self, path: str, **kwargs: Any) -> None:
        """
        Deserialize the ensemble aggregator state.

        Note: Sub-models must be re-registered after loading since they are
        not serialized with the aggregator.
        """
        import joblib

        # Load metadata
        with open(path + ".meta", "rb") as f:
            state = pickle.load(f)

        self.name = state["name"]
        self.config = state["config"]
        self.uncertainty_threshold = state["uncertainty_threshold"]
        self.direction_threshold = state["direction_threshold"]
        self.n_mc_passes = state["n_mc_passes"]
        self.stacking_n_splits = state["stacking_n_splits"]
        self.lgbm_params = state["lgbm_params"]
        self._meta_feature_names = state["meta_feature_names"]
        self._torch_models = state["torch_models"]

        # Reload components
        self.mc_estimator = MCDropoutEstimator(n_forward_passes=self.n_mc_passes)
        self.signal_generator = SignalGenerator(direction_threshold=self.direction_threshold)

        # Load LightGBM
        lgbm_path = path + ".lgbm"
        if os.path.exists(lgbm_path):
            self.lgbm_stacker = joblib.load(lgbm_path)

        # Load BMA
        bma_path = path + ".bma"
        if os.path.exists(bma_path):
            with open(bma_path, "rb") as f:
                bma_state = pickle.load(f)
            self.bma = BayesianModelAverager(
                model_names=bma_state["model_names"],
                temperature=bma_state["temperature"]
            )

        logger.info(
            "EnsembleAggregator loaded successfully",
            source=path,
            n_meta_features=len(self._meta_feature_names)
        )
