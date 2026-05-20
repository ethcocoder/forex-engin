import os
import pytest
import numpy as np
import torch

from models.meta_learner.maml import MAMLModel
from models.meta_learner.online_adapter import OnlineAdapter
from models.meta_learner.trainer import MAMLTrainer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples, seq_len, d_feat = 200, 10, 5
    X = np.random.randn(n_samples, seq_len, d_feat)
    y = np.random.randn(n_samples)
    return X, y


def test_maml_network_forward(sample_data):
    X, _ = sample_data
    d_feat = X.shape[-1]
    
    model = MAMLModel(config={"maml": {"meta_epochs": 1}})
    model._init_network(d_feat)
    
    # Forward pass
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = model.model(X_tensor)
    
    assert preds.shape == (200,)
    assert preds.requires_grad


def test_maml_meta_training_and_adaptation(sample_data):
    X, y = sample_data
    
    model = MAMLModel(config={"maml": {"meta_epochs": 2, "support_size": 20, "query_size": 10}})
    
    # Meta-train
    model.fit(X, y)
    
    # Check that model learned something (loss reduction during inner loop)
    # We test this via adaptation
    X_support, y_support = X[:20], y[:20]
    X_query = X[20:30]
    
    # Zero-shot prediction
    preds_zero = model.predict(X_query)
    
    # Adapt
    model.adapt(X_support, y_support)
    
    # Few-shot prediction
    preds_few = model.predict(X_query)
    
    assert preds_zero.shape == (10,)
    assert preds_few.shape == (10,)
    
    # Check that adaptation actually changed the predictions
    assert not np.allclose(preds_zero, preds_few)


def test_online_adapter(sample_data):
    X, y = sample_data
    model = MAMLModel(config={"maml": {"support_size": 20}})
    model._init_network(X.shape[-1])
    
    adapter = OnlineAdapter(model, buffer_size=20)
    
    for i in range(15):
        adapter.update(X[i], y[i])
        
    adapter.adapt_now()
    # Shouldn't adapt because buffer < support_size
    assert model.adapted_model is None
    
    for i in range(15, 25):
        adapter.update(X[i], y[i])
        
    adapter.adapt_now()
    # Now it should have adapted
    assert model.adapted_model is not None
    
    # Prediction
    pred = adapter.get_adapted_prediction(X[25:30])
    assert pred.shape == (5,)


def test_maml_serialization(sample_data, tmp_path):
    X, y = sample_data
    model = MAMLModel()
    model.fit(X, y)
    
    save_path = os.path.join(tmp_path, "maml.pt")
    model.save(save_path)
    
    loaded = MAMLModel()
    loaded.load(save_path)
    
    assert loaded.d_feat == model.d_feat
    assert loaded.inner_lr == model.inner_lr
    
    pred_orig = model.predict(X[:5])
    pred_loaded = loaded.predict(X[:5])
    
    np.testing.assert_allclose(pred_orig, pred_loaded)
