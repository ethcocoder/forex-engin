from __future__ import annotations

import numpy as np
import pandas as pd

from models.temporal.trainer import TimeSeriesPurgedTrainer
from research.temporal import TemporalTrainingConfig, build_causal_sequences


def test_causal_sequences_end_at_their_decision_timestamp() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    features = pd.DataFrame({"momentum_x": np.arange(6)}, index=index)
    target = pd.Series(np.arange(10, 16), index=index, dtype=float)

    sequences, targets, timestamps = build_causal_sequences(features, target, 3)

    assert sequences.shape == (4, 3, 1)
    np.testing.assert_array_equal(sequences[0, :, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(sequences[-1, :, 0], np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(targets, np.array([12.0, 13.0, 14.0, 15.0]))
    assert timestamps[0] == index[2]


def test_temporal_training_config_is_explicit_and_small_by_default() -> None:
    config = TemporalTrainingConfig()
    model_config = config.to_model_config()["temporal_fusion"]
    assert model_config["d_model"] == 32
    assert model_config["epochs"] == 5
    assert model_config["batch_size"] == 64


def test_legacy_purged_trainer_never_uses_future_indices() -> None:
    trainer = TimeSeriesPurgedTrainer(
        n_splits=3, label_horizon=4, embargo_pct=0.05
    )
    for train_indices, validation_indices in trainer.get_purged_splits(120):
        assert train_indices.max() + 4 < validation_indices.min()
