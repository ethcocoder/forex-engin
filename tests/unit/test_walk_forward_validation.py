import numpy as np
import pandas as pd

from models.feature_pipeline import LeakageSafeFeaturePipeline
from models.train_harness import ModelTrainingHarness


def real_quote_fixture(rows: int = 600) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02", periods=rows, freq="min", tz="UTC")
    trend = 1.1 + np.linspace(0.0, 0.002, rows)
    oscillation = 0.002 * np.sin(np.linspace(0.0, 80.0, rows))
    mid = trend + oscillation
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid": mid - 0.00005,
            "ask": mid + 0.00005,
            "mid": mid,
        }
    )


def test_executable_labels_use_bid_ask_and_drop_incomplete_horizon():
    pipeline = LeakageSafeFeaturePipeline(window_sizes=(5, 10))
    features = pipeline.compute_features(real_quote_fixture(80))
    labelled = pipeline.attach_executable_labels(features, horizon_bars=3)

    assert len(labelled) == len(features) - 3
    assert labelled["label_end_timestamp"].gt(labelled["timestamp"]).all()
    assert set(labelled["label"].unique()).issubset({-1, 0, 1})
    assert (labelled["long_return_h"] <= (labelled["mid"].shift(-3).fillna(labelled["mid"]) - labelled["ask"]) / labelled["ask"] + 0.01).all()


def test_purged_folds_do_not_allow_training_labels_to_overlap_test_time():
    pipeline = LeakageSafeFeaturePipeline(window_sizes=(5, 15))
    labelled = pipeline.attach_executable_labels(
        pipeline.compute_features(real_quote_fixture()), horizon_bars=4
    )
    splits = pipeline.purged_walk_forward_split(labelled, n_splits=3, purge_window=15)

    assert len(splits) == 3
    for train, test in splits:
        assert train["label_end_timestamp"].max() < test["timestamp"].min()


def test_harness_reports_research_only_and_never_returns_fitted_model():
    pipeline = LeakageSafeFeaturePipeline(window_sizes=(5, 15))
    labelled = pipeline.attach_executable_labels(
        pipeline.compute_features(real_quote_fixture()), horizon_bars=3
    )
    result = ModelTrainingHarness(n_estimators=10).evaluate_purged_walk_forward(
        labelled, pipeline, n_splits=2, purge_bars=15
    )

    assert result["status"] == "RESEARCH_ONLY"
    assert result["successful_folds"] >= 1
    assert all("model" not in fold for fold in result["folds"])
