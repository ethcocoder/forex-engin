import numpy as np
import torch
import torch.nn as nn
import structlog
from typing import Tuple

logger = structlog.get_logger()


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty quantification for PyTorch neural networks.

    Enables dropout layers at inference time and performs N stochastic forward
    passes to estimate predictive mean and standard deviation. This provides
    a computationally tractable approximation to Bayesian neural network
    posterior inference.

    The predictive uncertainty is computed as:
        ŷ = (1/N) Σ f_θ(x; z_i)
        σ_pred = sqrt((1/(N-1)) Σ (f_θ(x; z_i) - ŷ)²)

    where z_i represents different dropout masks.
    """

    def __init__(self, n_forward_passes: int = 30) -> None:
        """
        Args:
            n_forward_passes: Number of stochastic forward passes for MC estimation.
                Higher values give more accurate uncertainty estimates but increase
                latency linearly. Default 30 balances quality vs. speed (~15ms total).
        """
        self.n_forward_passes = n_forward_passes

        logger.info(
            "MCDropoutEstimator initialized",
            n_forward_passes=n_forward_passes
        )

    def _enable_dropout(self, model: nn.Module) -> None:
        """
        Enable dropout layers while keeping all other layers in eval mode.
        This is the key trick: we want stochastic dropout masks at inference time
        but deterministic batch norm statistics, etc.
        """
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def _disable_dropout(self, model: nn.Module) -> None:
        """Restore all dropout layers back to eval mode."""
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.eval()

    def estimate(
        self,
        model: nn.Module,
        X: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate predictive mean and uncertainty via MC Dropout.

        Args:
            model: A PyTorch nn.Module with dropout layers. Must be in eval mode
                   before calling (this method handles dropout activation internally).
            X: Input tensor of shape [n_samples, ...] or [seq_len, d_feat] etc.

        Returns:
            Tuple of:
                mean_prediction: np.ndarray of shape [n_samples] — MC average prediction.
                uncertainty: np.ndarray of shape [n_samples] — MC predictive std deviation.
        """
        was_training = model.training
        model.eval()
        self._enable_dropout(model)

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                pred = model(X)

                # Handle various output shapes
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if pred.dim() > 1:
                    pred = pred.squeeze(-1)

                predictions.append(pred.cpu().numpy())

        self._disable_dropout(model)

        # Restore original training state
        if was_training:
            model.train()

        # Stack: [n_forward_passes, n_samples]
        predictions_stack = np.stack(predictions, axis=0)

        # Predictive mean: [n_samples]
        mean_prediction = np.mean(predictions_stack, axis=0)

        # Predictive std: [n_samples]
        if self.n_forward_passes > 1:
            uncertainty = np.std(predictions_stack, axis=0, ddof=1)
        else:
            uncertainty = np.zeros_like(mean_prediction)

        logger.debug(
            "MC Dropout estimation completed",
            n_passes=self.n_forward_passes,
            mean_uncertainty=float(np.mean(uncertainty)),
            max_uncertainty=float(np.max(uncertainty)) if len(uncertainty) > 0 else 0.0
        )

        return mean_prediction, uncertainty
