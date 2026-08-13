import numpy as np
import torch
from typing import Any, List, Tuple, Dict
import structlog
from models.base_model import BaseModel

logger = structlog.get_logger()


class TimeSeriesPurgedTrainer:
    """
    Harness for walk-forward time-series model training and cross-validation.
    Implements purging (discarding training samples that overlap with validation targets)
    and embargoing (discarding training samples immediately following validation to avoid
    autocorrelation bias) as described by Marcos Lopez de Prado.
    """
    def __init__(
        self,
        n_splits: int = 4,
        label_horizon: int = 24,
        embargo_pct: float = 0.01
    ) -> None:
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_pct = embargo_pct

    def get_purged_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates purged and embargoed cross-validation splits.
        Returns a list of tuples containing (train_indices, val_indices).
        """
        splits = []
        # Divide sample space into equal segments for walk-forward validation
        segment_size = n_samples // (self.n_splits + 1)
        embargo = int(n_samples * self.embargo_pct)
        
        logger.info(
            "Generating purged cross-validation splits",
            total_samples=n_samples,
            segment_size=segment_size,
            label_horizon=self.label_horizon,
            embargo=embargo
        )
        
        for i in range(self.n_splits):
            # Validation set represents the (i+1)-th segment
            val_start = (i + 1) * segment_size
            val_end = min(val_start + segment_size, n_samples)
            val_indices = np.arange(val_start, val_end)
            
            # Expanding-window training: do not train on future observations.
            # A label created at index i reaches i + label_horizon, so retain only
            # samples whose label completes strictly before validation begins.
            train_end = max(0, val_start - self.label_horizon)
            train_indices = np.arange(train_end, dtype=np.int64)

            # `embargo` is retained in the public configuration for compatibility,
            # but is naturally satisfied here because future samples are never used
            # in an expanding-window fold. A subsequent fold begins later in time.
            splits.append((train_indices, val_indices))
            
        return splits

    def evaluate_cv(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Trains and validates a given BaseModel using Purged Cross-Validation.
        Computes out-of-sample MSE, RMSE, and track performance across folds.
        """
        n_samples = len(X)
        splits = self.get_purged_splits(n_samples)
        
        fold_mses = []
        logger.info("Beginning cross-validation evaluation", model=model.name, folds=self.n_splits)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                logger.warning("Empty indices generated for cross-validation split", fold=fold)
                continue
                
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            logger.info(
                "Training fold",
                fold=fold,
                train_samples=len(X_train),
                val_samples=len(X_val)
            )
            
            # Fit model on training set
            model.fit(X_train, y_train, **kwargs)
            
            # Predict out-of-sample
            preds = model.predict(X_val)
            
            # Calculate validation metric (Mean Squared Error)
            mse = np.mean((preds - y_val) ** 2)
            fold_mses.append(mse)
            
            logger.info("Fold evaluation completed", fold=fold, val_mse=float(mse))
            
        mean_mse = np.mean(fold_mses)
        mean_rmse = np.sqrt(mean_mse)
        
        logger.info(
            "Cross-validation complete",
            mean_mse=float(mean_mse),
            mean_rmse=float(mean_rmse),
            all_fold_mses=[float(m) for m in fold_mses]
        )
        
        return {
            "mean_mse": mean_mse,
            "mean_rmse": mean_rmse,
            "fold_mses": fold_mses
        }
