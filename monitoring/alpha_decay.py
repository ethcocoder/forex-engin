import numpy as np
import structlog
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = structlog.get_logger()

class AlphaDecayMonitor:
    """
    Elite Alpha Decay Monitoring System.
    Detects when the model's predictive power is fading relative to historical benchmarks.
    """

    def __init__(
        self,
        lookback_windows: List[int] = [100, 500, 1000],
        decay_threshold: float = 0.7,
        retrain_callback: Any = None
    ) -> None:
        self.lookback_windows = lookback_windows
        self.decay_threshold = decay_threshold
        self.retrain_callback = retrain_callback
        
        self.prediction_history: List[float] = []
        self.actual_history: List[float] = []
        self.timestamps: List[datetime] = []
        
        logger.info("AlphaDecayMonitor initialized", threshold=decay_threshold)

    def log_prediction(self, predicted_return: float, actual_return: float) -> None:
        """Logs a prediction vs actual outcome to track correlation."""
        self.prediction_history.append(predicted_return)
        self.actual_history.append(actual_return)
        self.timestamps.append(datetime.utcnow())
        
        # Check for decay periodically
        if len(self.prediction_history) % 50 == 0:
            self._check_for_decay()

    def _check_for_decay(self) -> None:
        """Calculates Information Coefficient (IC) across multiple windows."""
        if len(self.prediction_history) < max(self.lookback_windows):
            return

        ic_values = {}
        for window in self.lookback_windows:
            preds = np.array(self.prediction_history[-window:])
            actuals = np.array(self.actual_history[-window:])
            
            # Correlation between predicted and actual returns (IC)
            correlation = np.corrcoef(preds, actuals)[0, 1]
            ic_values[window] = correlation

        # Compare short-term IC vs long-term IC
        short_term_ic = ic_values[self.lookback_windows[0]]
        long_term_ic = ic_values[self.lookback_windows[-1]]
        
        decay_ratio = short_term_ic / (long_term_ic + 1e-6)
        
        logger.info(
            "Alpha Decay Check",
            short_ic=f"{short_term_ic:.4f}",
            long_ic=f"{long_term_ic:.4f}",
            decay_ratio=f"{decay_ratio:.2f}"
        )

        if decay_ratio < self.decay_threshold and long_term_ic > 0.05:
            logger.warning("CRITICAL: Alpha decay detected! Triggering retrain sequence.")
            if self.retrain_callback:
                self.retrain_callback()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current health status of the Alpha signal."""
        if not self.prediction_history:
            return {"status": "WARMUP"}
            
        return {
            "history_size": len(self.prediction_history),
            "current_ic": np.corrcoef(self.prediction_history[-100:], self.actual_history[-100:])[0, 1] if len(self.prediction_history) >= 100 else 0.0
        }
