"""Leakage-safe temporal-model training built on the research-core contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.temporal.combined import TemporalFusionModel
from research.training import (
    BaselineTrainingConfig,
    BaselineTrainingResult,
    _regression_metrics,
    build_research_matrix,
)


@dataclass(frozen=True)
class TemporalTrainingConfig:
    """Small, explicit temporal-model configuration for a research run."""

    sequence_length: int = 48
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    d_model: int = 32
    num_channels: tuple[int, ...] = (16, 32)
    nhead: int = 4
    num_layers: int = 1
    dim_feedforward: int = 64
    dropout: float = 0.1
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")

    def to_model_config(self) -> dict[str, Any]:
        return {
            "temporal_fusion": {
                "d_model": self.d_model,
                "num_channels": list(self.num_channels),
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "lr": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            }
        }


def build_causal_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create sequences ending at each decision time, without future features."""
    if not features.index.equals(target.index):
        raise ValueError("Features and target must have identical timestamps.")
    if len(features) < sequence_length:
        raise ValueError("Not enough samples for the requested sequence length.")
    values = features.to_numpy(dtype=np.float64, copy=True)
    targets = target.to_numpy(dtype=np.float64, copy=True)
    samples = len(features) - sequence_length + 1
    sequence_tensor = np.empty(
        (samples, sequence_length, values.shape[1]), dtype=np.float32
    )
    for sample_index in range(samples):
        sequence_tensor[sample_index] = values[
            sample_index : sample_index + sequence_length
        ]
    return (
        sequence_tensor,
        targets[sequence_length - 1 :].astype(np.float32),
        features.index[sequence_length - 1 :],
    )


def run_temporal_walk_forward(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    data_config: BaselineTrainingConfig,
    temporal_config: TemporalTrainingConfig,
    artifact_root: str | Path = "artifacts/experiments",
) -> BaselineTrainingResult:
    """Train/evaluate a temporal model using fold-local scaling and OOS predictions."""
    X, y, selected_features, metadata = build_research_matrix(raw, features, data_config)
    sequences, targets, timestamps = build_causal_sequences(
        X, y, temporal_config.sequence_length
    )
    if len(sequences) <= data_config.split.min_train_size:
        raise ValueError("Not enough sequence samples for the configured training window.")

    np.random.seed(temporal_config.random_seed)
    torch.manual_seed(temporal_config.random_seed)
    fold_metrics: list[dict[str, float]] = []
    predictions = pd.Series(index=timestamps, dtype=float, name="prediction")

    for fold_number, (train_idx, validation_idx) in enumerate(
        data_config.split.split(len(sequences))
    ):
        # Fit each scaler solely on observations ending within this fold's
        # historical training period; no validation data inform centring/scaling.
        train_sequence_end = int(train_idx.max()) + temporal_config.sequence_length
        scaler = StandardScaler().fit(X.iloc[:train_sequence_end])
        scaled_features = scaler.transform(X).astype(np.float32)
        scaled_sequences, _, _ = build_causal_sequences(
            pd.DataFrame(scaled_features, index=X.index, columns=X.columns),
            y,
            temporal_config.sequence_length,
        )

        torch.manual_seed(temporal_config.random_seed + fold_number)
        model = TemporalFusionModel(
            name=f"temporal_fold_{fold_number}",
            config=temporal_config.to_model_config(),
        )
        model.fit(scaled_sequences[train_idx], targets[train_idx])
        prediction = model.predict(scaled_sequences[validation_idx]).astype(float)
        validation_target = targets[validation_idx].astype(float)
        metrics = _regression_metrics(prediction, validation_target)
        metrics.update(
            {
                "fold": float(fold_number),
                "train_rows": float(len(train_idx)),
                "validation_rows": float(len(validation_idx)),
            }
        )
        fold_metrics.append(metrics)
        predictions.iloc[validation_idx] = prediction

    if not fold_metrics:
        raise ValueError("The configured split produced no temporal-model folds.")
    oos_mask = predictions.notna()
    aggregate_metrics = _regression_metrics(
        predictions.loc[oos_mask].to_numpy(dtype=float),
        targets[oos_mask.to_numpy()].astype(float),
    )

    # Persist a final model and matching scaler, while retaining the OOS result
    # as the only model-quality diagnostic used for subsequent validation gates.
    final_scaler = StandardScaler().fit(X)
    final_features = pd.DataFrame(
        final_scaler.transform(X), index=X.index, columns=X.columns
    )
    final_sequences, _, _ = build_causal_sequences(
        final_features, y, temporal_config.sequence_length
    )
    torch.manual_seed(temporal_config.random_seed)
    final_model = TemporalFusionModel(
        name="temporal_final", config=temporal_config.to_model_config()
    )
    final_model.fit(final_sequences, targets)

    run_id = f"temporal-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    final_model.save(str(artifact_dir / "model.pt"))
    joblib.dump(final_scaler, artifact_dir / "scaler.joblib")
    pd.DataFrame(
        {
            "target": targets[oos_mask.to_numpy()],
            "prediction": predictions.loc[oos_mask].to_numpy(dtype=float),
        },
        index=timestamps[oos_mask.to_numpy()],
    ).to_csv(artifact_dir / "oos_predictions.csv", index_label="timestamp")

    metadata_payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_config": data_config.to_dict(),
        "temporal_config": asdict(temporal_config),
        "research_matrix": metadata,
        "feature_columns": selected_features,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "research_only": True,
        "execution_ready": False,
    }
    (artifact_dir / "run_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return BaselineTrainingResult(
        run_id=run_id,
        artifact_dir=str(artifact_dir),
        rows_after_alignment=int(metadata["rows_after_alignment"]),
        feature_columns=tuple(selected_features),
        fold_metrics=tuple(fold_metrics),
        aggregate_metrics=aggregate_metrics,
    )
