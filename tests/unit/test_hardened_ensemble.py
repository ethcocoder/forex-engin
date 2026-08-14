from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.hardened_ensemble import (
    ConformalIntervalCalibrator,
    FeatureSchema,
    HardenedCrossFittedEnsemble,
    HardenedEnsembleConfig,
    ModelSchemaError,
    run_hardened_ensemble_walk_forward,
)
from research.splits import ExpandingPurgedWalkForwardSplit


def _dataset(rows: int = 144) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    x1 = np.linspace(-1.0, 1.0, rows)
    x2 = np.sin(np.linspace(0.0, 6.0, rows))
    x3 = np.cos(np.linspace(0.0, 3.0, rows))
    features = pd.DataFrame(
        {"momentum_x": x1, "volatility_x": x2, "kalman_x": x3}, index=index
    )
    target = pd.Series(0.002 * x1 - 0.001 * x2 + 0.0003 * x3, index=index)
    return features, target


def _config() -> HardenedEnsembleConfig:
    return HardenedEnsembleConfig(
        split=ExpandingPurgedWalkForwardSplit(
            n_splits=2,
            label_horizon=1,
            embargo_bars=1,
            min_train_size=40,
        ),
        hgb_max_iter=20,
        hgb_min_samples_leaf=5,
        minimum_meta_rows=10,
        random_seed=17,
    )


def test_feature_schema_rejects_reordered_missing_and_nonfinite_values() -> None:
    features, _ = _dataset(8)
    schema = FeatureSchema.from_frame(features)
    validated = schema.validate(features[["kalman_x", "momentum_x", "volatility_x"]])
    assert list(validated.columns) == ["momentum_x", "volatility_x", "kalman_x"]
    with pytest.raises(ModelSchemaError, match="missing"):
        schema.validate(features.drop(columns="kalman_x"))
    invalid = features.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ModelSchemaError, match="non-finite"):
        schema.validate(invalid)


def test_conformal_calibrator_produces_symmetric_intervals() -> None:
    calibrator = ConformalIntervalCalibrator.fit(
        np.array([0.0, 0.1, -0.1, 0.2]),
        np.array([0.1, 0.0, -0.2, 0.4]),
        target_coverage=0.75,
    )
    lower, upper = calibrator.interval(np.array([0.05, -0.05]))
    assert np.allclose(upper - lower, 2.0 * calibrator.absolute_error_quantile)
    assert calibrator.calibration_rows == 4


def test_cross_fitted_ensemble_persists_artifacts_and_supports_abstention(tmp_path) -> None:
    features, target = _dataset()
    result = run_hardened_ensemble_walk_forward(
        features, target, _config(), tmp_path, research_metadata={"source": "unit_test"}
    )
    artifact_dir = tmp_path / result.run_id
    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "oos_predictions.csv").exists()
    assert result.aggregate_metrics["abstention_rate"] >= 0.0

    import joblib

    loaded: HardenedCrossFittedEnsemble = joblib.load(artifact_dir / "model.joblib")
    diagnostics = loaded.predict_with_diagnostics(features.iloc[-3:])
    assert list(diagnostics.columns) == [
        "prediction",
        "interval_lower",
        "interval_upper",
        "abstain",
        "base_prediction_std",
        "actionable_prediction",
    ]
    assert diagnostics.index.equals(features.index[-3:])
    assert (diagnostics.loc[diagnostics["abstain"], "actionable_prediction"] == 0.0).all()
    oos = pd.read_csv(artifact_dir / "oos_predictions.csv")
    assert {"base_ridge", "base_elastic_net", "base_hist_gradient_boosting"}.issubset(
        oos.columns
    )


def test_cross_fitted_ensemble_is_deterministic_for_fixed_seed() -> None:
    features, target = _dataset()
    left = HardenedCrossFittedEnsemble(_config())
    right = HardenedCrossFittedEnsemble(_config())
    left_oos, _ = left.fit_cross_fitted(features, target)
    right_oos, _ = right.fit_cross_fitted(features, target)
    np.testing.assert_allclose(left_oos["prediction"], right_oos["prediction"])
    np.testing.assert_allclose(left_oos["interval_lower"], right_oos["interval_lower"])
