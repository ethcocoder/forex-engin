import numpy as np
from typing import Any, Dict, List, Tuple
import structlog

from models.base_model import BaseModel
from models.regime.combined import RegimeEnsembleEstimator

logger = structlog.get_logger()


class RegimeTrainer:
    """
    Harness for training and evaluating market regime classification models.
    Supports sequential HMM unsupervised learning, pseudo-label generation,
    LSTM sequence model training, and dynamic regime stability evaluation.
    """
    def __init__(self, n_splits: int = 4) -> None:
        self.n_splits = n_splits

    def get_walk_forward_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates standard walk-forward cross-validation splits.
        Returns a list of tuples containing (train_indices, val_indices).
        """
        splits = []
        segment_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            val_start = (i + 1) * segment_size
            val_end = min(val_start + segment_size, n_samples)
            
            train_indices = np.arange(0, val_start)
            val_indices = np.arange(val_start, val_end)
            
            splits.append((train_indices, val_indices))
            
        return splits

    def evaluate_cv(
        self,
        ensemble: RegimeEnsembleEstimator,
        X: np.ndarray,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Runs walk-forward cross-validation for the Regime Ensemble.
        Evaluates out-of-sample consistency (alignment between LSTM and HMM)
        and transition stability.
        """
        n_samples = len(X)
        splits = self.get_walk_forward_splits(n_samples)
        
        fold_alignments = []
        fold_state_entropies = []
        
        logger.info("Beginning regime cross-validation", folds=self.n_splits)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
                
            X_train = X[train_idx]
            X_val = X[val_idx]
            
            logger.info(
                "Training regime fold",
                fold=fold,
                train_samples=len(X_train),
                val_samples=len(X_val)
            )
            
            # Train the ensemble on the training set
            ensemble.fit(X_train)
            
            # Evaluate out-of-sample predictions on the validation set
            # HMM point-in-time predictions
            X_val_hmm = X_val[:, -1, :]
            y_hmm_val = ensemble.hmm.predict(X_val_hmm)
            
            # LSTM predictions
            y_lstm_val = ensemble.lstm.predict(X_val)
            
            # Calculate alignment (fraction of identical state assignments)
            alignment = np.mean(y_hmm_val == y_lstm_val)
            fold_alignments.append(alignment)
            
            # Calculate state entropy to check for degenerate states
            _, counts = np.unique(y_lstm_val, return_counts=True)
            probs = counts / len(y_lstm_val)
            entropy = -np.sum(probs * np.log2(probs + 1e-15))
            fold_state_entropies.append(entropy)
            
            logger.info(
                "Fold evaluation completed",
                fold=fold,
                lstm_hmm_alignment=float(alignment),
                state_entropy=float(entropy)
            )
            
        # Re-fit ensemble on the full dataset at the end
        logger.info("Fitting regime ensemble on full dataset")
        ensemble.fit(X)
        
        return {
            "mean_alignment": float(np.mean(fold_alignments)),
            "mean_entropy": float(np.mean(fold_state_entropies)),
            "fold_alignments": fold_alignments,
            "fold_entropies": fold_state_entropies
        }

    def analyze_regimes(self, ensemble: RegimeEnsembleEstimator, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the distribution, transition matrices, and feature profiles
        of the states predicted by the ensemble on a given dataset X.
        """
        # Predict regimes
        y_pred = ensemble.predict(X)
        n_samples = len(y_pred)
        
        # Calculate state distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        distribution = {int(k): float(v / n_samples) for k, v in zip(unique, counts)}
        
        # Check transition matrix from HMM
        transmat = ensemble.hmm.get_transition_matrix()
        
        # Calculate feature profiles per state
        # X is [n_samples, seq_len, d_feat], extract last step for point-in-time features
        X_pit = X[:, -1, :]
        d_feat = X_pit.shape[1]
        
        profiles = {}
        for state in range(ensemble.hmm.n_components):
            state_mask = (y_pred == state)
            if np.sum(state_mask) > 0:
                state_feats = X_pit[state_mask]
                profiles[state] = {
                    "mean": state_feats.mean(axis=0).tolist(),
                    "std": state_feats.std(axis=0).tolist(),
                    "count": int(np.sum(state_mask))
                }
            else:
                profiles[state] = {
                    "mean": [0.0] * d_feat,
                    "std": [0.0] * d_feat,
                    "count": 0
                }
                
        logger.info(
            "Regime analysis completed",
            state_distribution=distribution,
            num_states=len(unique)
        )
        
        return {
            "distribution": distribution,
            "transition_matrix": transmat.tolist(),
            "profiles": profiles
        }
