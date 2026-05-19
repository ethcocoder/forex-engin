import os
import tempfile
import numpy as np
import torch
import pytest

from models.temporal.transformer import TransformerEncoderModel, TransformerEncoderNet
from models.temporal.tcn import TCNModel, TCNNet
from models.temporal.combined import TemporalFusionModel, TemporalFusionNet
from models.temporal.trainer import TimeSeriesPurgedTrainer

from models.regime.hmm import GaussianHMMRegimeEstimator
from models.regime.lstm_classifier import LSTMRegimeClassifier
from models.regime.combined import RegimeEnsembleEstimator
from models.regime.trainer import RegimeTrainer


@pytest.fixture
def synthetic_time_series_data():
    """Generates synthetic sequence dataset for temporal and regime model testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 150
    seq_len = 10
    d_feat = 8
    
    X = np.random.randn(n_samples, seq_len, d_feat).astype(np.float32)
    # Regression target: continuous forward returns
    y_reg = np.random.randn(n_samples).astype(np.float32)
    # Classification target: discrete regimes (0, 1, 2, 3)
    y_cls = np.random.randint(0, 4, size=n_samples)
    
    return X, y_reg, y_cls


def test_temporal_models_instantiation_and_shape(synthetic_time_series_data):
    """Verifies that temporal models can be instantiated, fitted, and predict correct shapes."""
    X, y_reg, _ = synthetic_time_series_data
    n_samples, seq_len, d_feat = X.shape
    
    config = {
        "transformer": {
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
            "epochs": 2,
            "batch_size": 16
        },
        "tcn": {
            "num_channels": [16, 16],
            "kernel_size": 2,
            "epochs": 2,
            "batch_size": 16
        },
        "temporal_fusion": {
            "d_model": 16,
            "num_channels": [16, 16],
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
            "kernel_size": 2,
            "epochs": 2,
            "batch_size": 16
        }
    }
    
    # 1. Test Transformer Encoder Model
    transformer = TransformerEncoderModel(config=config)
    transformer.fit(X, y_reg)
    preds_trans = transformer.predict(X)
    assert preds_trans.shape == (n_samples,)
    
    # 2. Test TCN Model
    tcn = TCNModel(config=config)
    tcn.fit(X, y_reg)
    preds_tcn = tcn.predict(X)
    assert preds_tcn.shape == (n_samples,)
    
    # 3. Test Temporal Fusion Model
    fusion = TemporalFusionModel(config=config)
    fusion.fit(X, y_reg)
    preds_fusion = fusion.predict(X)
    assert preds_fusion.shape == (n_samples,)


def test_temporal_models_causal_masking_no_leakage():
    """
    Mathematically verifies that future steps (> t) in a sequence
    have zero influence on the hidden states at or before step t.
    """
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 10
    d_feat = 8
    split_step = 5  # index t where inputs will diverge
    
    # Generate S1 and S2 which are identical up to split_step, but diverge after
    x1 = torch.randn(batch_size, seq_len, d_feat)
    x2 = x1.clone()
    x2[:, split_step:, :] = torch.randn(batch_size, seq_len - split_step, d_feat)
    
    # Assert that they are indeed identical up to split_step and different after
    assert torch.equal(x1[:, :split_step, :], x2[:, :split_step, :])
    assert not torch.equal(x1[:, split_step:, :], x2[:, split_step:, :])
    
    # 1. Test Transformer Causal Masking
    net_trans = TransformerEncoderNet(d_feat=d_feat, d_model=16, nhead=2, num_layers=1)
    net_trans.eval()
    
    # Generate causal mask
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    
    # Run projection & positional encoding manually to examine raw encoder output
    with torch.no_grad():
        p1 = net_trans.pos_encoder(net_trans.projection(x1))
        p2 = net_trans.pos_encoder(net_trans.projection(x2))
        
        out1 = net_trans.transformer_encoder(p1, mask=mask)
        out2 = net_trans.transformer_encoder(p2, mask=mask)
        
        # Hidden representations up to step `split_step` MUST be perfectly identical
        torch.testing.assert_close(out1[:, :split_step, :], out2[:, :split_step, :])

    # 2. Test TCN Causal Padding
    net_tcn = TCNNet(d_feat=d_feat, num_channels=[16, 16], kernel_size=3)
    net_tcn.eval()
    
    with torch.no_grad():
        # Transpose to [batch, channels, seq_len]
        tx1 = x1.transpose(1, 2)
        tx2 = x2.transpose(1, 2)
        
        out1_tcn = net_tcn.tcn(tx1)
        out2_tcn = net_tcn.tcn(tx2)
        
        # Output sequence length dimension is at index 2
        # Hidden states up to `split_step` MUST be perfectly identical
        torch.testing.assert_close(out1_tcn[:, :, :split_step], out2_tcn[:, :, :split_step])

    # 3. Test Temporal Fusion Cross-Attention Causal Fusion
    net_fusion = TemporalFusionNet(d_feat=d_feat, d_model=16, num_channels=[16, 16], nhead=2)
    net_fusion.eval()
    
    with torch.no_grad():
        # 1. Transformer pass
        trans1 = net_fusion.projection(x1)
        trans1 = net_fusion.pos_encoder(trans1)
        trans_out1 = net_fusion.transformer_backbone(trans1, mask=mask)
        
        trans2 = net_fusion.projection(x2)
        trans2 = net_fusion.pos_encoder(trans2)
        trans_out2 = net_fusion.transformer_backbone(trans2, mask=mask)
        
        # 2. TCN pass
        tcn_out1 = net_fusion.tcn_backbone(x1.transpose(1, 2)).transpose(1, 2)
        tcn_out2 = net_fusion.tcn_backbone(x2.transpose(1, 2)).transpose(1, 2)
        
        # 3. Cross-Attention
        fused1, _ = net_fusion.cross_attention(query=trans_out1, key=tcn_out1, value=tcn_out1, attn_mask=mask)
        fused2, _ = net_fusion.cross_attention(query=trans_out2, key=tcn_out2, value=tcn_out2, attn_mask=mask)
        
        # Fused hidden representations up to `split_step` MUST be perfectly identical
        torch.testing.assert_close(fused1[:, :split_step, :], fused2[:, :split_step, :])


def test_purged_cross_validation_splits():
    """Verifies that walk-forward purged and embargoed cross-validation splits eliminate overlap."""
    n_samples = 120
    label_horizon = 6
    embargo_pct = 0.05  # 5% of 120 = 6 samples
    n_splits = 3
    
    trainer = TimeSeriesPurgedTrainer(n_splits=n_splits, label_horizon=label_horizon, embargo_pct=embargo_pct)
    splits = trainer.get_purged_splits(n_samples)
    
    assert len(splits) == n_splits
    segment_size = n_samples // (n_splits + 1)  # 120 // 4 = 30
    embargo_size = int(n_samples * embargo_pct)  # 6
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        assert len(val_idx) == segment_size
        
        val_start = val_idx[0]
        val_end = val_idx[-1] + 1
        
        # 1. Overlap purging check:
        # Train index cannot lie in: [val_start - label_horizon, val_end - 1]
        purged_range = set(range(val_start - label_horizon, val_end))
        # 2. Embargoing check:
        # Train index cannot lie in: [val_end, val_end + embargo_size - 1]
        embargo_range = set(range(val_end, val_end + embargo_size))
        
        excluded_range = purged_range.union(embargo_range)
        
        # Verify no training indices fall inside the excluded range
        for idx in train_idx:
            assert idx not in excluded_range


def test_gaussian_hmm_regime_estimator():
    """Tests fitting, predictions, transition matrices, and state alignment in GaussianHMMRegimeEstimator."""
    np.random.seed(42)
    # Generate 150 points with distinct volatility features
    low_vol = np.random.randn(80, 3) * 0.2
    high_vol = np.random.randn(70, 3) * 1.5
    X = np.vstack([low_vol, high_vol])
    
    config = {
        "hmm": {
            "n_components": 2,
            "covariance_type": "full",
            "n_iter": 50,
            "tol": 1e-3,
            "random_state": 42
        }
    }
    
    estimator = GaussianHMMRegimeEstimator(config=config)
    estimator.fit(X)
    
    # 1. State Alignment validation
    # Verify states are ordered by volatility: state 0 (low vol) < state 1 (high vol)
    covs = estimator.model.covars_
    vol0 = np.trace(covs[0])
    vol1 = np.trace(covs[1])
    assert vol0 < vol1
    
    # 2. Transitions validation
    transmat = estimator.get_transition_matrix()
    assert transmat.shape == (2, 2)
    np.testing.assert_allclose(transmat.sum(axis=1), 1.0, atol=1e-6)
    
    # 3. Decoded predictions and posteriors shape check
    preds = estimator.predict(X)
    assert preds.shape == (150,)
    assert set(preds).issubset({0, 1})
    
    probs = estimator.predict(X, return_proba=True)
    assert probs.shape == (150, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_lstm_regime_classifier(synthetic_time_series_data):
    """Tests fitting, predicting, and shape outputs for LSTMRegimeClassifier."""
    X, _, y_cls = synthetic_time_series_data
    n_samples, seq_len, d_feat = X.shape
    
    config = {
        "lstm_regime": {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_classes": 4,
            "epochs": 2,
            "batch_size": 16
        }
    }
    
    classifier = LSTMRegimeClassifier(config=config)
    classifier.fit(X, y_cls)
    
    # Test point prediction (argmax)
    preds = classifier.predict(X)
    assert preds.shape == (n_samples,)
    
    # Test probability output
    probs = classifier.predict(X, return_proba=True)
    assert probs.shape == (n_samples, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_regime_ensemble_estimator(synthetic_time_series_data):
    """Tests unified fit/predict workflow for the fused RegimeEnsembleEstimator."""
    X, _, _ = synthetic_time_series_data
    n_samples, seq_len, d_feat = X.shape
    
    config = {
        "ensemble": {
            "w_hmm": 0.4,
            "w_lstm": 0.6
        },
        "hmm": {
            "n_components": 3,
            "random_state": 42
        },
        "lstm_regime": {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_classes": 3,
            "epochs": 2,
            "batch_size": 16
        }
    }
    
    ensemble = RegimeEnsembleEstimator(config=config)
    ensemble.fit(X)
    
    # Verify predictions
    preds = ensemble.predict(X)
    assert preds.shape == (n_samples,)
    
    # Verify combined posterior probabilities
    probs = ensemble.predict(X, return_proba=True)
    assert probs.shape == (n_samples, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_regime_trainer(synthetic_time_series_data):
    """Tests the RegimeTrainer walk-forward CV and dynamic analysis reports."""
    X, _, _ = synthetic_time_series_data
    
    config = {
        "hmm": {
            "n_components": 3,
            "random_state": 42
        },
        "lstm_regime": {
            "hidden_dim": 16,
            "num_layers": 1,
            "num_classes": 3,
            "epochs": 2,
            "batch_size": 16
        }
    }
    
    ensemble = RegimeEnsembleEstimator(config=config)
    trainer = RegimeTrainer(n_splits=2)
    
    # 1. Test walk-forward cross validation
    cv_metrics = trainer.evaluate_cv(ensemble, X)
    assert "mean_alignment" in cv_metrics
    assert "mean_entropy" in cv_metrics
    assert len(cv_metrics["fold_alignments"]) == 2
    
    # 2. Test regime structure profiling
    report = trainer.analyze_regimes(ensemble, X)
    assert "distribution" in report
    assert "transition_matrix" in report
    assert "profiles" in report
    assert len(report["distribution"]) <= 3
    assert len(report["profiles"]) == 3


def test_model_serialization_and_deserialization(synthetic_time_series_data):
    """Verifies that all models can be serialized and deserialized back to identical outputs."""
    X, y_reg, _ = synthetic_time_series_data
    
    config = {
        "transformer": {"d_model": 16, "nhead": 2, "epochs": 1, "batch_size": 16},
        "tcn": {"num_channels": [16], "epochs": 1, "batch_size": 16},
        "hmm": {"n_components": 2, "random_state": 42},
        "lstm_regime": {"hidden_dim": 16, "num_classes": 2, "epochs": 1, "batch_size": 16},
        "ensemble": {"w_hmm": 0.5, "w_lstm": 0.5}
    }
    
    # Setup temporary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Test Transformer Serialization
        trans_model = TransformerEncoderModel(config=config)
        trans_model.fit(X, y_reg)
        trans_preds_before = trans_model.predict(X)
        
        path_trans = os.path.join(tmp_dir, "trans.pt")
        trans_model.save(path_trans)
        
        trans_model_new = TransformerEncoderModel(config=config)
        trans_model_new.load(path_trans)
        trans_preds_after = trans_model_new.predict(X)
        np.testing.assert_allclose(trans_preds_before, trans_preds_after, atol=1e-5)
        
        # 2. Test TCN Serialization
        tcn_model = TCNModel(config=config)
        tcn_model.fit(X, y_reg)
        tcn_preds_before = tcn_model.predict(X)
        
        path_tcn = os.path.join(tmp_dir, "tcn.pt")
        tcn_model.save(path_tcn)
        
        tcn_model_new = TCNModel(config=config)
        tcn_model_new.load(path_tcn)
        tcn_preds_after = tcn_model_new.predict(X)
        np.testing.assert_allclose(tcn_preds_before, tcn_preds_after, atol=1e-5)
        
        # 3. Test HMM Serialization
        hmm_model = GaussianHMMRegimeEstimator(config=config)
        hmm_model.fit(X[:, -1, :])
        hmm_preds_before = hmm_model.predict(X[:, -1, :])
        
        path_hmm = os.path.join(tmp_dir, "hmm.pkl")
        hmm_model.save(path_hmm)
        
        hmm_model_new = GaussianHMMRegimeEstimator(config=config)
        hmm_model_new.load(path_hmm)
        hmm_preds_after = hmm_model_new.predict(X[:, -1, :])
        np.testing.assert_allclose(hmm_preds_before, hmm_preds_after)
        
        # 4. Test LSTM Serialization
        lstm_model = LSTMRegimeClassifier(config=config)
        y_pseudo = hmm_preds_before
        lstm_model.fit(X, y_pseudo)
        lstm_preds_before = lstm_model.predict(X)
        
        path_lstm = os.path.join(tmp_dir, "lstm.pt")
        lstm_model.save(path_lstm)
        
        lstm_model_new = LSTMRegimeClassifier(config=config)
        lstm_model_new.load(path_lstm)
        lstm_preds_after = lstm_model_new.predict(X)
        np.testing.assert_allclose(lstm_preds_before, lstm_preds_after)
        
        # 5. Test Ensemble Serialization
        ensemble_model = RegimeEnsembleEstimator(config=config)
        ensemble_model.fit(X)
        ens_preds_before = ensemble_model.predict(X)
        
        path_ens = os.path.join(tmp_dir, "ensemble.pkl")
        ensemble_model.save(path_ens)
        
        ensemble_model_new = RegimeEnsembleEstimator(config=config)
        ensemble_model_new.load(path_ens)
        ens_preds_after = ensemble_model_new.predict(X)
        np.testing.assert_allclose(ens_preds_before, ens_preds_after)
