from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from research.contracts import DataContractError, MarketDataContract, build_dataset_manifest
from research.labels import ForwardReturnLabelSpec, build_forward_return_labels
from research.splits import ExpandingPurgedWalkForwardSplit
from research.training import BaselineTrainingConfig, build_research_matrix, run_ridge_walk_forward


def _raw_frame(rows: int = 48) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = 1.10 + np.linspace(0.0, 0.02, rows)
    return pd.DataFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": np.arange(rows, dtype=float) + 100.0,
        },
        index=index,
    )


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "momentum_roc": np.linspace(-1.0, 1.0, len(frame)),
            "volatility_cc": np.linspace(0.01, 0.02, len(frame)),
            "kalman_velocity": np.linspace(-0.001, 0.001, len(frame)),
            # Must not be selected under the strict approved-feature default.
            "sentiment_score_ema": np.linspace(-0.5, 0.5, len(frame)),
        },
        index=frame.index,
    )


def _config() -> BaselineTrainingConfig:
    return BaselineTrainingConfig(
        pair="EUR_USD",
        provider="unit_test",
        label_spec=ForwardReturnLabelSpec(horizon_bars=1, entry_lag_bars=1),
        split=ExpandingPurgedWalkForwardSplit(
            n_splits=2,
            label_horizon=2,
            embargo_bars=1,
            min_train_size=12,
        ),
        min_rows=24,
    )


def test_market_data_contract_rejects_duplicate_timestamps() -> None:
    raw = _raw_frame(8)
    raw.index = raw.index.where(raw.index != raw.index[3], raw.index[2])
    with pytest.raises(DataContractError, match="Duplicate timestamps"):
        MarketDataContract(pair="EUR_USD").validate(raw)


def test_label_has_entry_lag_and_preserves_unavailable_tail() -> None:
    raw = _raw_frame(6)
    labels = build_forward_return_labels(
        raw, ForwardReturnLabelSpec(horizon_bars=1, entry_lag_bars=1)
    )
    expected = np.log(raw["close"].iloc[2] / raw["close"].iloc[1])
    assert labels["forward_return"].iloc[0] == pytest.approx(expected)
    assert labels["forward_return"].iloc[-1] != labels["forward_return"].iloc[-1]
    assert labels["forward_return"].iloc[-2] != labels["forward_return"].iloc[-2]


def test_purged_split_has_no_target_overlap_with_validation() -> None:
    splitter = ExpandingPurgedWalkForwardSplit(
        n_splits=3, label_horizon=2, embargo_bars=1, min_train_size=12
    )
    folds = list(splitter.split(48))
    assert len(folds) >= 2
    for train_idx, validation_idx in folds:
        assert train_idx.max() + 2 < validation_idx.min()
        assert train_idx.max() >= 11


def test_research_matrix_excludes_unapproved_and_nonfinite_features() -> None:
    raw = _raw_frame()
    features = _features(raw)
    features.loc[features.index[0], "momentum_roc"] = np.nan
    X, y, selected, metadata = build_research_matrix(raw, features, _config())
    assert "sentiment_score_ema" not in selected
    assert list(X.columns) == selected
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    assert np.isfinite(y.to_numpy(dtype=float)).all()
    assert metadata["dropped_rows"] >= 3
    manifest = build_dataset_manifest(raw, MarketDataContract(pair="EUR_USD"))
    assert manifest.rows == len(raw)


def test_walk_forward_baseline_persists_auditable_artifacts(tmp_path) -> None:
    raw = _raw_frame()
    result = run_ridge_walk_forward(raw, _features(raw), _config(), tmp_path)
    artifact_dir = tmp_path / result.run_id
    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "oos_predictions.csv").exists()
    metadata = json.loads((artifact_dir / "run_metadata.json").read_text())
    assert metadata["research_only"] is True
    assert metadata["execution_ready"] is False
    assert metadata["feature_columns"] == [
        "momentum_roc",
        "volatility_cc",
        "kalman_velocity",
    ]
    assert np.isfinite(metadata["aggregate_metrics"]["rmse"])
