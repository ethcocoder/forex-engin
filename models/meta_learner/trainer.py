import numpy as np
import structlog
from typing import Any, Dict

from models.meta_learner.maml import MAMLModel

logger = structlog.get_logger()


class MAMLTrainer:
    """
    Harness for meta-training and evaluating MAML models using
    walk-forward meta-validation.
    """

    def __init__(self, n_splits: int = 3, holdout_pct: float = 0.2) -> None:
        self.n_splits = n_splits
        self.holdout_pct = holdout_pct
        
        logger.info(
            "MAMLTrainer initialized",
            n_splits=n_splits,
            holdout_pct=holdout_pct
        )

    def evaluate_meta_learning(
        self,
        maml_model: MAMLModel,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Walk-forward evaluation of MAML adaptation quality.
        
        Splits data sequentially. For each split, meta-trains on the first portion
        and tests few-shot adaptation on the holdout portion.
        """
        n_samples = X.shape[0]
        fold_size = n_samples // self.n_splits
        
        metrics = {
            "mean_query_loss": [],
            "adaptation_speed": []
        }
        
        logger.info("Starting MAML walk-forward meta-validation")
        
        for i in range(self.n_splits):
            logger.info(f"Processing fold {i + 1}/{self.n_splits}")
            
            fold_end = fold_size * (i + 1)
            X_fold = X[:fold_end]
            y_fold = y[:fold_end]
            
            split_point = int(len(X_fold) * (1.0 - self.holdout_pct))
            
            X_train, y_train = X_fold[:split_point], y_fold[:split_point]
            X_val, y_val = X_fold[split_point:], y_fold[split_point:]
            
            # Reset model for fresh meta-training
            if maml_model.d_feat is not None:
                maml_model._init_network(maml_model.d_feat)
                
            # Meta-train
            maml_model.fit(X_train, y_train)
            
            # Evaluate adaptation on validation episodes
            episodes = maml_model._construct_episodes(X_val, y_val)
            if not episodes:
                continue
                
            fold_q_loss = 0.0
            fold_adapt_speed = 0.0
            
            import torch
            import torch.nn as nn
            loss_fn = nn.MSELoss()
            
            maml_model.model.eval()
            with torch.no_grad():
                for s_X, s_y, q_X, q_y in episodes:
                    # Pre-adaptation query loss (zero-shot)
                    zero_shot_preds = maml_model.model(q_X)
                    l0 = loss_fn(zero_shot_preds, q_y).item()
                    
                    # Adapt
                    adapted = maml_model._inner_loop_adapt(maml_model.model, s_X, s_y)
                    adapted.eval()
                    
                    # Post-adaptation query loss (few-shot)
                    few_shot_preds = adapted(q_X)
                    lK = loss_fn(few_shot_preds, q_y).item()
                    
                    fold_q_loss += lK
                    if l0 > 1e-6:
                        fold_adapt_speed += (l0 - lK) / l0
                        
            metrics["mean_query_loss"].append(fold_q_loss / len(episodes))
            metrics["adaptation_speed"].append(fold_adapt_speed / len(episodes))
            
        results = {
            "mean_query_loss": float(np.mean(metrics["mean_query_loss"])) if metrics["mean_query_loss"] else 0.0,
            "mean_adaptation_speed": float(np.mean(metrics["adaptation_speed"])) if metrics["adaptation_speed"] else 0.0,
            "fold_query_losses": metrics["mean_query_loss"],
            "fold_adaptation_speeds": metrics["adaptation_speed"]
        }
        
        logger.info("MAML walk-forward meta-validation completed", results=results)
        return results

    def analyze_adaptation(
        self,
        maml_model: MAMLModel,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze step-by-step loss reduction during inner-loop adaptation.
        """
        maml_model.fit(X, y)
        episodes = maml_model._construct_episodes(X, y)
        
        if not episodes:
            return {"loss_curve": []}
            
        import torch
        import torch.nn as nn
        import torch.optim as optim
        loss_fn = nn.MSELoss()
        
        avg_losses = np.zeros(maml_model.num_inner_steps + 1)
        
        for s_X, s_y, q_X, q_y in episodes:
            clone = maml_model._clone_model(maml_model.model)
            inner_opt = optim.SGD(clone.parameters(), lr=maml_model.inner_lr)
            clone.eval()
            
            with torch.no_grad():
                l0 = loss_fn(clone(q_X), q_y).item()
            avg_losses[0] += l0
            
            clone.train()
            for step in range(maml_model.num_inner_steps):
                inner_opt.zero_grad()
                s_loss = loss_fn(clone(s_X), s_y)
                s_loss.backward()
                inner_opt.step()
                
                clone.eval()
                with torch.no_grad():
                    lk = loss_fn(clone(q_X), q_y).item()
                avg_losses[step + 1] += lk
                clone.train()
                
        avg_losses /= len(episodes)
        
        return {
            "loss_curve": avg_losses.tolist()
        }
