"""Leakage-resistant, uncertainty-aware model ensemble for FX research.

This module intentionally separates research prediction quality from trading
execution. It produces audited out-of-sample forecasts and abstention decisions;
it contains no broker, order, or position-management integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.splits import ExpandingPurgedWalkForwardSplit
from research.training import _regression_metrics


class ModelSchemaError(ValueError):
    """Raised when inference data do not exactly match a trained feature schema."""


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered numeric feature contract persisted with every trained ensemble."""

    columns: tuple[str, ...]
    sha256: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FeatureSchema":
        columns = tuple(str(column) for column in frame.columns)
        if not columns:
            raise ModelSchemaError("A model schema requires at least one feature column.")
        digest = sha256("\n".join(columns).encode("utf-8")).hexdigest()
        return cls(columns=columns, sha256=digest)

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise ModelSchemaError("Inference features must be a pandas DataFrame.")
        received = tuple(str(column) for column in frame.columns)
        missing = [column for column in self.columns if column not in frame.columns]
        unexpected = [column for column in received if column not in self.columns]
        if missing or unexpected:
            raise ModelSchemaError(
                f"Feature schema mismatch; missing={missing}, unexpected={unexpected}."
            )
        ordered = frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        values = ordered.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ModelSchemaError("Inference features contain non-finite values.")
        return ordered


@dataclass(frozen=True)
class ConformalIntervalCalibrator:
    """Symmetric split-conformal interval calibrated on prior OOS residuals."""

    absolute_error_quantile: float
    target_coverage: float
    calibration_rows: int

    @classmethod
    def fit(
        cls,
        prediction: np.ndarray,
        target: np.ndarray,
        target_coverage: float,
    ) -> "ConformalIntervalCalibrator":
        if not 0.0 < target_coverage < 1.0:
            raise ValueError("target_coverage must be between zero and one.")
        prediction = np.asarray(prediction, dtype=float)
        target = np.asarray(target, dtype=float)
        if len(prediction) != len(target) or len(target) == 0:
            raise ValueError("Calibration requires equal non-empty prediction and target arrays.")
        errors = np.abs(target - prediction)
        if not np.isfinite(errors).all():
            raise ValueError("Calibration residuals must be finite.")
        # The "higher" rule avoids under-covering due to interpolation.
        try:
            quantile = float(np.quantile(errors, target_coverage, method="higher"))
        except TypeError:  # pragma: no cover - compatibility with older NumPy
            quantile = float(np.quantile(errors, target_coverage, interpolation="higher"))
        return cls(
            absolute_error_quantile=quantile,
            target_coverage=target_coverage,
            calibration_rows=len(errors),
        )

    def interval(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(prediction, dtype=float)
        width = self.absolute_error_quantile
        return prediction - width, prediction + width


@dataclass(frozen=True)
class HardenedEnsembleConfig:
    """Reproducible configuration for the cross-fitted ensemble."""

    split: ExpandingPurgedWalkForwardSplit
    ridge_alpha: float = 3.0
    elasticnet_alpha: float = 1e-4
    elasticnet_l1_ratio: float = 0.15
    hgb_max_iter: int = 150
    hgb_learning_rate: float = 0.04
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 64
    meta_ridge_alpha: float = 10.0
    conformal_coverage: float = 0.80
    minimum_meta_rows: int = 256
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.minimum_meta_rows < 1:
            raise ValueError("minimum_meta_rows must be positive.")
        if not 0.0 < self.conformal_coverage < 1.0:
            raise ValueError("conformal_coverage must be between zero and one.")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["split"] = asdict(self.split)
        return payload


@dataclass(frozen=True)
class HardenedEnsembleResult:
    """Saved model lineage and genuinely chronological OOS diagnostics."""

    run_id: str
    artifact_dir: str
    rows: int
    feature_schema_sha256: str
    fold_metrics: tuple[dict[str, float], ...]
    aggregate_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HardenedCrossFittedEnsemble:
    """Cross-fitted base models plus a chronological meta-model and abstention.

    Base models are refit inside every fold. The meta-model and conformal
    calibrator for a validation fold use only *earlier* completed OOS folds.
    This guards against the common error of training a stacker on the same
    predictions it is being judged on.
    """

    base_model_names = ("ridge", "elastic_net", "hist_gradient_boosting")

    def __init__(self, config: HardenedEnsembleConfig) -> None:
        self.config = config
        self.schema: FeatureSchema | None = None
        self.base_models: dict[str, Any] = {}
        self.meta_model: Pipeline | None = None
        self.calibrator: ConformalIntervalCalibrator | None = None
        self.is_fitted = False

    def _new_base_models(self, fold_number: int) -> dict[str, Any]:
        seed = self.config.random_seed + fold_number
        return {
            "ridge": Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(alpha=self.config.ridge_alpha))]
            ),
            "elastic_net": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=self.config.elasticnet_alpha,
                            l1_ratio=self.config.elasticnet_l1_ratio,
                            max_iter=10_000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                learning_rate=self.config.hgb_learning_rate,
                max_iter=self.config.hgb_max_iter,
                max_leaf_nodes=self.config.hgb_max_leaf_nodes,
                min_samples_leaf=self.config.hgb_min_samples_leaf,
                l2_regularization=1e-3,
                random_state=seed,
            ),
        }

    @staticmethod
    def _predict_matrix(models: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for name in HardenedCrossFittedEnsemble.base_model_names:
            prediction = np.asarray(models[name].predict(X), dtype=float).reshape(-1)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Base model '{name}' generated non-finite predictions.")
            predictions.append(prediction)
        return np.column_stack(predictions)

    def _new_meta_model(self) -> Pipeline:
        return Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=self.config.meta_ridge_alpha))]
        )

    @staticmethod
    def _diagnostics(
        prediction: np.ndarray,
        target: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        base_prediction_matrix: np.ndarray,
    ) -> dict[str, float]:
        metrics = _regression_metrics(prediction, target)
        interval_available = np.isfinite(lower) & np.isfinite(upper)
        abstain = (lower <= 0.0) & (upper >= 0.0)
        coverage = (target >= lower) & (target <= upper)
        metrics.update(
            {
                "conformal_coverage": float(np.mean(coverage[interval_available]))
                if interval_available.any()
                else 0.0,
                "interval_available_rate": float(np.mean(interval_available)),
                "abstention_rate": float(np.mean(abstain)),
                "mean_base_prediction_std": float(
                    np.mean(np.std(base_prediction_matrix, axis=1, ddof=0))
                ),
            }
        )
        eligible = ~abstain
        if eligible.any():
            metrics["eligible_directional_accuracy"] = float(
                np.mean(np.sign(prediction[eligible]) == np.sign(target[eligible]))
            )
            metrics["eligible_rows"] = float(eligible.sum())
        else:
            metrics["eligible_directional_accuracy"] = 0.0
            metrics["eligible_rows"] = 0.0
        return metrics

    def fit_cross_fitted(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, tuple[dict[str, float], ...]]:
        """Produce chronological OOS predictions, then fit final inference artifacts."""
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be a DataFrame and y must be a Series.")
        if not X.index.equals(y.index):
            raise ValueError("X and y must share an identical timestamp index.")
        self.schema = FeatureSchema.from_frame(X)
        X = self.schema.validate(X)
        y = pd.to_numeric(y, errors="coerce").astype(float)
        if not np.isfinite(y.to_numpy()).all():
            raise ValueError("Training target contains non-finite values.")

        oos_rows: list[pd.DataFrame] = []
        prior_meta_features: list[np.ndarray] = []
        prior_targets: list[np.ndarray] = []
        prior_predictions: list[np.ndarray] = []
        fold_metrics: list[dict[str, float]] = []

        for fold_number, (train_idx, validation_idx) in enumerate(
            self.config.split.split(len(X))
        ):
            models = self._new_base_models(fold_number)
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[validation_idx], y.iloc[validation_idx]
            for model in models.values():
                model.fit(X_train, y_train)
            base_matrix = self._predict_matrix(models, X_val)

            previous_rows = sum(len(values) for values in prior_targets)
            if previous_rows >= self.config.minimum_meta_rows:
                meta_model = self._new_meta_model()
                meta_model.fit(np.vstack(prior_meta_features), np.concatenate(prior_targets))
                prediction = np.asarray(meta_model.predict(base_matrix), dtype=float)
            else:
                # The first chronological fold cannot use future validation rows
                # to train a stacker, so it uses a transparent equal-weight mean.
                prediction = np.mean(base_matrix, axis=1)

            if prior_predictions:
                calibrator = ConformalIntervalCalibrator.fit(
                    np.concatenate(prior_predictions),
                    np.concatenate(prior_targets),
                    self.config.conformal_coverage,
                )
                lower, upper = calibrator.interval(prediction)
            else:
                # No historical OOS residuals are available yet. Abstain every
                # first-fold point rather than manufacturing confidence.
                lower = np.full_like(prediction, -np.inf)
                upper = np.full_like(prediction, np.inf)

            diagnostics = self._diagnostics(
                prediction,
                y_val.to_numpy(dtype=float),
                lower,
                upper,
                base_matrix,
            )
            diagnostics.update(
                {
                    "fold": float(fold_number),
                    "train_rows": float(len(train_idx)),
                    "validation_rows": float(len(validation_idx)),
                    "meta_rows_available": float(previous_rows),
                }
            )
            fold_metrics.append(diagnostics)
            oos_payload: dict[str, Any] = {
                "target": y_val.to_numpy(dtype=float),
                "prediction": prediction,
                "interval_lower": lower,
                "interval_upper": upper,
                "abstain": (lower <= 0.0) & (upper >= 0.0),
                "base_prediction_std": np.std(base_matrix, axis=1, ddof=0),
                "fold": fold_number,
            }
            for column_number, model_name in enumerate(self.base_model_names):
                oos_payload[f"base_{model_name}"] = base_matrix[:, column_number]
            oos_rows.append(pd.DataFrame(oos_payload, index=X_val.index))
            prior_meta_features.append(base_matrix)
            prior_targets.append(y_val.to_numpy(dtype=float))
            prior_predictions.append(prediction)

        if not oos_rows:
            raise ValueError("The configured split produced no valid OOS folds.")
        oos = pd.concat(oos_rows).sort_index()

        # Final model artifacts use all valid research data only after the OOS
        # evidence above has been recorded. These artifacts are for observation;
        # they do not alter the OOS result.
        self.base_models = self._new_base_models(fold_number=len(fold_metrics))
        for model in self.base_models.values():
            model.fit(X, y)
        self.meta_model = self._new_meta_model()
        self.meta_model.fit(
            np.vstack(prior_meta_features), np.concatenate(prior_targets)
        )
        self.calibrator = ConformalIntervalCalibrator.fit(
            oos["prediction"].to_numpy(dtype=float),
            oos["target"].to_numpy(dtype=float),
            self.config.conformal_coverage,
        )
        self.is_fitted = True
        return oos, tuple(fold_metrics)

    def predict_with_diagnostics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return point forecasts, calibrated intervals and conservative abstention."""
        if not self.is_fitted or self.schema is None or self.meta_model is None or self.calibrator is None:
            raise RuntimeError("The ensemble must be fitted before inference.")
        X = self.schema.validate(X)
        base_matrix = self._predict_matrix(self.base_models, X)
        prediction = np.asarray(self.meta_model.predict(base_matrix), dtype=float)
        lower, upper = self.calibrator.interval(prediction)
        abstain = (lower <= 0.0) & (upper >= 0.0)
        return pd.DataFrame(
            {
                "prediction": prediction,
                "interval_lower": lower,
                "interval_upper": upper,
                "abstain": abstain,
                "base_prediction_std": np.std(base_matrix, axis=1, ddof=0),
                "actionable_prediction": np.where(abstain, 0.0, prediction),
            },
            index=X.index,
        )


def run_hardened_ensemble_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    config: HardenedEnsembleConfig,
    artifact_root: str | Path = "artifacts/experiments",
    research_metadata: dict[str, Any] | None = None,
) -> HardenedEnsembleResult:
    """Train, evaluate and persist a strict research-only ensemble artifact."""
    np.random.seed(config.random_seed)
    ensemble = HardenedCrossFittedEnsemble(config)
    oos, fold_metrics = ensemble.fit_cross_fitted(X, y)
    finite_interval = np.isfinite(oos["interval_lower"]) & np.isfinite(oos["interval_upper"])
    aggregate_metrics = HardenedCrossFittedEnsemble._diagnostics(
        oos["prediction"].to_numpy(dtype=float),
        oos["target"].to_numpy(dtype=float),
        np.where(finite_interval, oos["interval_lower"], -np.inf),
        np.where(finite_interval, oos["interval_upper"], np.inf),
        oos[[f"base_{name}" for name in HardenedCrossFittedEnsemble.base_model_names]].to_numpy(
            dtype=float
        ),
    )
    aggregate_metrics["mean_base_prediction_std"] = float(
        oos["base_prediction_std"].mean()
    )
    run_id = f"hardened-ensemble-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(ensemble, artifact_dir / "model.joblib")
    oos.to_csv(artifact_dir / "oos_predictions.csv", index_label="timestamp")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ensemble_config": config.to_dict(),
        "feature_schema": asdict(ensemble.schema),
        "fold_metrics": list(fold_metrics),
        "aggregate_metrics": aggregate_metrics,
        "research_metadata": research_metadata or {},
        "research_only": True,
        "execution_ready": False,
        "promotion_eligible": False,
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return HardenedEnsembleResult(
        run_id=run_id,
        artifact_dir=str(artifact_dir),
        rows=len(X),
        feature_schema_sha256=ensemble.schema.sha256,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
    )
