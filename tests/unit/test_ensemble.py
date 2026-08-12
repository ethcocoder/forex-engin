import os
import pytest
import numpy as np
import torch
import torch.nn as nn

from models.ensemble.signal_generator import AlphaSignal, SignalGenerator
from models.ensemble.uncertainty import MCDropoutEstimator
from models.ensemble.weighting import DynamicWeightTracker, BayesianModelAverager
from models.ensemble.aggregator import EnsembleAggregator


class DummyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.dropout(self.fc(x)).squeeze(-1)


class DummyBaseModel:
    def __init__(self, name="dummy"):
        self.name = name
        self.model = DummyTorchModel()
        
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).numpy()
            
    def fit(self, X, y):
        pass


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    seq_len = 10
    d_feat = 5
    X = np.random.randn(n_samples, seq_len, d_feat)
    y = np.random.randn(n_samples)
    return X, y


def test_alpha_signal_contract():
    signal = AlphaSignal(
        direction=1,
        magnitude=0.8,
        confidence=0.9,
        uncertainty=0.1,
        expected_decay_steps=5,
        regime=0,
        timestamp=1234567890.0,
        metadata={"test": True}
    )
    assert signal.direction == 1
    assert signal.magnitude == 0.8
    
    with pytest.raises(ValueError):
        AlphaSignal(direction=2, magnitude=0.5, confidence=0.5, uncertainty=0.1, expected_decay_steps=1, regime=0, timestamp=0.0)
        
    with pytest.raises(ValueError):
        AlphaSignal(direction=1, magnitude=1.5, confidence=0.5, uncertainty=0.1, expected_decay_steps=1, regime=0, timestamp=0.0)


def test_signal_generator():
    gen = SignalGenerator(direction_threshold=0.01, decay_lookback=10)
    
    # Strong positive
    s1 = gen.generate(prediction=0.03, confidence=0.8, uncertainty=0.1, regime=1)
    assert s1.direction == 1
    assert s1.magnitude == 1.0  # capped
    assert s1.expected_decay_steps == 9  # int(10 * (1 - 0.1))
    
    # Weak negative
    s2 = gen.generate(prediction=-0.005, confidence=0.5, uncertainty=0.8, regime=0)
    assert s2.direction == 0
    assert 0.0 < s2.magnitude < 0.2
    assert s2.expected_decay_steps == 2


def test_mc_dropout_estimator(sample_data):
    X, _ = sample_data
    model = DummyTorchModel()
    model.eval()
    
    estimator = MCDropoutEstimator(n_forward_passes=10)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    mean_pred, uncertainty = estimator.estimate(model, X_tensor)
    
    assert mean_pred.shape == (100,)
    assert uncertainty.shape == (100,)
    assert np.any(uncertainty > 0)  # dropout should cause variance
    
    # After estimation, model should be back in eval mode
    assert not model.dropout.training


def test_dynamic_weight_tracker():
    tracker = DynamicWeightTracker(["m1", "m2"], lookback=50)
    
    # Feed some correlated and uncorrelated data
    # We need at least 30 samples to get non-zero ICs
    for i in range(40):
        actual = float(np.sin(i))
        preds = {
            "m1": actual + 0.1 * np.random.randn(),  # Highly correlated
            "m2": np.random.randn()  # Uncorrelated
        }
        tracker.update(preds, actual)
        
    weights = tracker.get_weights(temperature=1.0)
    assert weights["m1"] > weights["m2"]
    assert np.isclose(sum(weights.values()), 1.0)


def test_ensemble_aggregator(sample_data, tmp_path):
    X, y = sample_data
    
    agg = EnsembleAggregator(config={"ensemble": {"stacking_n_splits": 2, "lgbm_params": {"n_estimators": 10}}})
    agg.register_model("dummy1", DummyBaseModel(), is_torch=True)
    agg.register_model("dummy2", DummyBaseModel(), is_torch=False)
    
    # Build deterministic stand-in OOF meta-features; production callers must
    # create these through the purged walk-forward harness.
    cluster_preds, cluster_uncerts, _ = agg._collect_meta_data(X)
    oof_meta_features = agg._build_meta_features(cluster_preds, cluster_uncerts)
    oof_feature_names = agg._generate_meta_feature_names(cluster_preds)
    provenance = {
        "validation_type": "purged_walk_forward",
        "fold_count": 2,
        "embargo_rows": 10,
        "label_horizon_rows": 5,
        "data_manifest_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="skip_oos"):
        agg.fit(X, y, skip_oos=True)
    agg.fit(
        X,
        y,
        oof_meta_features=oof_meta_features,
        oof_feature_names=oof_feature_names,
        oof_provenance=provenance,
    )
    
    # Predict (returns AlphaSignal by default)
    signal = agg.predict(X[:5])
    assert isinstance(signal, AlphaSignal)
    assert "dummy1" in signal.metadata["sub_model_predictions"]
    
    # Predict raw
    raw = agg.predict(X[:5], return_signal=False)
    assert isinstance(raw, float)
    
    # Test serialization
    save_path = os.path.join(tmp_path, "ensemble")
    agg.save(save_path)
    
    loaded = EnsembleAggregator()
    loaded.load(save_path)
    
    assert loaded.uncertainty_threshold == agg.uncertainty_threshold
    assert loaded.n_mc_passes == agg.n_mc_passes
    assert loaded.lgbm_stacker is not None
    with pytest.raises(RuntimeError, match="model set"):
        loaded.predict(X[:5])

    restored = EnsembleAggregator()
    restored.register_model("dummy1", DummyBaseModel(), is_torch=True)
    restored.register_model("dummy2", DummyBaseModel(), is_torch=False)
    restored.load(save_path)
    assert isinstance(restored.predict(X[:5]), AlphaSignal)
