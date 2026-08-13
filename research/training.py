"""Reproducible baseline model training for causal FX research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.contracts import MarketDataContract, build_dataset_manifest
from research.labels import ForwardReturnLabelSpec, build_forward_return_labels
from research.splits import ExpandingPurgedWalkForwardSplit


DEFAULT_APPROVED_FEATURE_PREFIXES: tuple[str, ...] = (
    "volatility_",
    "momentum_",
    "mean_reversion_",
    "trend_",
    "volume_",
    "wavelet_",
    "kalman_",
)


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Configuration for an auditable Ridge baseline experiment."""

    pair: str
    provider: str
    label_spec: ForwardReturnLabelSpec
    split: ExpandingPurgedWalkForwardSplit
    ridge_alpha: float = 1.0
    random_seed: int = 7
    feature_prefixes: tuple[str, ...] = DEFAULT_APPROVED_FEATURE_PREFIXES
    min_rows: int = 512

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["label_spec"] = self.label_spec.to_dict()
        payload["split"] = asdict(self.split)
        return payload


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Results and lineage for a completed walk-forward baseline experiment."""

    run_id: str
    artifact_dir: str
    rows_after_alignment: int
    feature_columns: tuple[str, ...]
    fold_metrics: tuple[dict[str, float], ...]
    aggregate_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_approved_features(
    features: pd.DataFrame,
    prefixes: Sequence[str] = DEFAULT_APPROVED_FEATURE_PREFIXES,
) -> list[str]:
    """Select only documented, directly computable research features by default."""
    selected = [
        str(column)
        for column in features.columns
        if any(str(column).startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        raise ValueError("No approved feature columns were selected.")
    return selected


def _feature_schema_hash(columns: Iterable[str]) -> str:
    return sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    pred_std = float(np.std(prediction))
    target_std = float(np.std(target))
    if pred_std == 0.0 or target_std == 0.0:
        information_coefficient = 0.0
    else:
        information_coefficient = float(np.corrcoef(prediction, target)[0, 1])
    directional_accuracy = float(np.mean(np.sign(prediction) == np.sign(target)))
    return {
        "rmse": rmse,
        "mae": mae,
        "information_coefficient": information_coefficient,
        "directional_accuracy": directional_accuracy,
    }


def build_research_matrix(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    config: BaselineTrainingConfig,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, object]]:
    """Validate, align and construct a finite, causally-labelled training matrix.

    Rows with unavailable rolling features or unavailable future labels are dropped;
    they are never backward-filled, forward-filled or converted to zero.
    """
    contract = MarketDataContract(pair=config.pair, provider=config.provider)
    validated_raw = contract.validate(raw)
    manifest = build_dataset_manifest(validated_raw, contract)

    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("Feature frame must use a DatetimeIndex.")
    if features.index.tz is None:
        raise ValueError("Feature timestamps must be timezone-aware.")
    feature_frame = features.copy()
    feature_frame.index = feature_frame.index.tz_convert("UTC")
    feature_frame = feature_frame.sort_index()
    if feature_frame.index.has_duplicates:
        raise ValueError("Feature timestamps must be unique.")

    selected_features = select_approved_features(
        feature_frame, config.feature_prefixes
    )
    numeric_features = feature_frame[selected_features].apply(
        pd.to_numeric, errors="coerce"
    )
    labels = build_forward_return_labels(validated_raw, config.label_spec)
    merged = numeric_features.join(labels, how="inner")

    finite_feature_mask = np.isfinite(
        merged[selected_features].to_numpy(dtype=float)
    ).all(axis=1)
    target_column = config.label_spec.target_name
    complete_mask = (
        finite_feature_mask
        & np.isfinite(merged[target_column].to_numpy(dtype=float))
        & merged[f"{target_column}_available_at"].notna().to_numpy()
    )
    clean = merged.loc[complete_mask].copy()
    if len(clean) < config.min_rows:
        raise ValueError(
            f"Only {len(clean)} fully observed rows remain; need at least {config.min_rows}."
        )

    X = clean[selected_features]
    y = clean[target_column]
    metadata = {
        "dataset_manifest": manifest.to_dict(),
        "feature_schema_sha256": _feature_schema_hash(selected_features),
        "rows_before_alignment": int(len(merged)),
        "rows_after_alignment": int(len(clean)),
        "dropped_rows": int(len(merged) - len(clean)),
        "first_training_timestamp": clean.index[0].isoformat(),
        "last_training_timestamp": clean.index[-1].isoformat(),
    }
    return X, y, selected_features, metadata


def run_ridge_walk_forward(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    config: BaselineTrainingConfig,
    artifact_root: str | Path = "artifacts/experiments",
) -> BaselineTrainingResult:
    """Train a transparent baseline and persist full research lineage.

    This routine intentionally provides only a regression-quality baseline. It
    makes no trading-performance claim and must be followed by cost-aware
    backtesting and risk validation before any paper-execution stage.
    """
    np.random.seed(config.random_seed)
    X, y, selected_features, metadata = build_research_matrix(raw, features, config)
    splitter = config.split
    fold_metrics: list[dict[str, float]] = []
    oos_predictions = pd.Series(index=X.index, dtype=float, name="prediction")

    for fold_number, (train_idx, validation_idx) in enumerate(splitter.split(len(X))):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=config.ridge_alpha)),
            ]
        )
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_validation = X.iloc[validation_idx]
        y_validation = y.iloc[validation_idx]
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        metrics = _regression_metrics(prediction, y_validation.to_numpy(dtype=float))
        metrics.update(
            {
                "fold": float(fold_number),
                "train_rows": float(len(train_idx)),
                "validation_rows": float(len(validation_idx)),
            }
        )
        fold_metrics.append(metrics)
        oos_predictions.loc[X_validation.index] = prediction

    if not fold_metrics:
        raise ValueError("The configured split produced no valid folds.")

    valid_prediction_mask = oos_predictions.notna()
    aggregate_metrics = _regression_metrics(
        oos_predictions.loc[valid_prediction_mask].to_numpy(dtype=float),
        y.loc[valid_prediction_mask].to_numpy(dtype=float),
    )

    final_model = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=config.ridge_alpha))]
    )
    final_model.fit(X, y)

    run_id = f"ridge-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(final_model, artifact_dir / "model.joblib")
    pd.DataFrame(
        {
            "target": y.loc[valid_prediction_mask],
            "prediction": oos_predictions.loc[valid_prediction_mask],
        }
    ).to_csv(artifact_dir / "oos_predictions.csv", index_label="timestamp")

    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_config": config.to_dict(),
        "research_matrix": metadata,
        "feature_columns": selected_features,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "research_only": True,
        "execution_ready": False,
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    return BaselineTrainingResult(
        run_id=run_id,
        artifact_dir=str(artifact_dir),
        rows_after_alignment=int(metadata["rows_after_alignment"]),
        feature_columns=tuple(selected_features),
        fold_metrics=tuple(fold_metrics),
        aggregate_metrics=aggregate_metrics,
    )
