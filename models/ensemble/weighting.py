import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


class DynamicWeightTracker:
    """
    Tracks rolling Information Coefficients (IC) for each sub-model
    and converts them to dynamic ensemble weights via softmax.

    The IC for model m is defined as:
        IC_m = Corr(ŷ_m, y_actual)

    computed over a rolling lookback window with exponential decay.

    Weights are computed via temperature-scaled softmax:
        w_m = exp(IC_m / τ) / Σ exp(IC_j / τ)
    """

    def __init__(
        self,
        model_names: List[str],
        lookback: int = 720,
        decay: float = 0.99
    ) -> None:
        """
        Args:
            model_names: List of sub-model identifiers.
            lookback: Rolling window size for IC computation (default 720 = ~30 days at 1h).
            decay: Exponential decay factor for recency-weighting observations.
        """
        self.model_names = list(model_names)
        self.lookback = lookback
        self.decay = decay

        # Per-model prediction histories
        self.prediction_histories: Dict[str, deque] = {
            name: deque(maxlen=lookback) for name in self.model_names
        }
        # Shared actual returns history
        self.actual_history: deque = deque(maxlen=lookback)

        self._lock = threading.Lock()

        logger.info(
            "DynamicWeightTracker initialized",
            models=self.model_names,
            lookback=lookback,
            decay=decay
        )

    def update(self, predictions_dict: Dict[str, float], actual_return: float) -> None:
        """
        Record a new prediction-actual pair for all models.

        Args:
            predictions_dict: Mapping of model_name -> predicted return.
            actual_return: The realized actual return for this timestep.
        """
        with self._lock:
            self.actual_history.append(actual_return)

            for name in self.model_names:
                pred = predictions_dict.get(name, 0.0)
                self.prediction_histories[name].append(pred)

    def get_rolling_ics(self) -> Dict[str, float]:
        """
        Compute Pearson correlation (IC) between each model's predictions
        and actual returns over the rolling buffer.

        Returns:
            Dict mapping model_name -> IC float.
            Returns 0.0 for models with insufficient data (< 30 samples).
        """
        with self._lock:
            n = len(self.actual_history)
            if n < 30:
                return {name: 0.0 for name in self.model_names}

            actuals = np.array(self.actual_history)
            ics = {}

            for name in self.model_names:
                preds = np.array(self.prediction_histories[name])

                # Handle constant predictions or actuals
                if np.std(preds) < 1e-12 or np.std(actuals) < 1e-12:
                    ics[name] = 0.0
                else:
                    # Apply exponential decay weights for recency bias
                    weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
                    weights /= weights.sum()

                    # Weighted Pearson correlation
                    mean_p = np.average(preds, weights=weights)
                    mean_a = np.average(actuals, weights=weights)

                    cov = np.average((preds - mean_p) * (actuals - mean_a), weights=weights)
                    std_p = np.sqrt(np.average((preds - mean_p) ** 2, weights=weights))
                    std_a = np.sqrt(np.average((actuals - mean_a) ** 2, weights=weights))

                    if std_p < 1e-12 or std_a < 1e-12:
                        ics[name] = 0.0
                    else:
                        ics[name] = float(cov / (std_p * std_a))

            return ics

    def get_weights(self, temperature: float = 1.0) -> Dict[str, float]:
        """
        Compute softmax-normalized ensemble weights from rolling ICs.

        Args:
            temperature: Softmax temperature. Lower = sharper concentration
                on the best model; higher = more uniform weights.

        Returns:
            Dict mapping model_name -> weight in [0, 1], summing to 1.0.
        """
        ics = self.get_rolling_ics()

        # Softmax with temperature scaling
        ic_values = np.array([ics[name] for name in self.model_names])

        if temperature < 1e-12:
            # Degenerate: winner-takes-all
            weights_arr = np.zeros(len(self.model_names))
            weights_arr[np.argmax(ic_values)] = 1.0
        else:
            # Numerical stability: subtract max before exp
            scaled = ic_values / temperature
            scaled -= np.max(scaled)
            exp_vals = np.exp(scaled)
            weights_arr = exp_vals / exp_vals.sum()

        weights = {name: float(w) for name, w in zip(self.model_names, weights_arr)}

        logger.debug(
            "Dynamic weights computed",
            ics=ics,
            weights=weights,
            temperature=temperature
        )

        return weights


class BayesianModelAverager:
    """
    Bayesian Model Averaging (BMA) ensemble fallback.

    Used when prediction uncertainty is too high for the LightGBM stacking
    layer. Computes weighted averages of sub-model predictions using rolling
    Information Coefficients as weights.

    Weight computation:
        IC_m = Corr(ŷ_m, y_actual)  [rolling, exponentially decayed]
        w_m = softmax(IC_m / τ)     [temperature-scaled normalization]
    """

    def __init__(
        self,
        model_names: List[str],
        lookback: int = 720,
        temperature: float = 1.0
    ) -> None:
        """
        Args:
            model_names: List of sub-model identifiers.
            lookback: Rolling window size for IC computation.
            temperature: Softmax temperature for weight normalization.
        """
        self.model_names = list(model_names)
        self.temperature = temperature
        self.tracker = DynamicWeightTracker(
            model_names=self.model_names,
            lookback=lookback
        )

        logger.info(
            "BayesianModelAverager initialized",
            models=self.model_names,
            lookback=lookback,
            temperature=temperature
        )

    def update(self, predictions_dict: Dict[str, float], actual_return: float) -> None:
        """
        Record a new prediction-actual pair.

        Args:
            predictions_dict: Mapping of model_name -> predicted return.
            actual_return: The realized actual return.
        """
        self.tracker.update(predictions_dict, actual_return)

    def average(self, predictions_dict: Dict[str, float]) -> float:
        """
        Compute BMA-weighted average of sub-model predictions.

        Args:
            predictions_dict: Mapping of model_name -> predicted return.

        Returns:
            Weighted average prediction as a float.
        """
        weights = self.get_weights()

        weighted_sum = 0.0
        for name in self.model_names:
            pred = predictions_dict.get(name, 0.0)
            weighted_sum += weights.get(name, 0.0) * pred

        return float(weighted_sum)

    def get_weights(self) -> Dict[str, float]:
        """
        Returns current BMA weights (softmax-normalized ICs).

        Returns:
            Dict mapping model_name -> weight in [0, 1], summing to 1.0.
        """
        return self.tracker.get_weights(temperature=self.temperature)
