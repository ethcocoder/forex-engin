import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class AlphaSignal:
    """
    Frozen dataclass representing the unified output of the Neural Ensemble.
    This is the locked API contract between the Ensemble Aggregator (Layer 2)
    and the Risk Engine (Layer 4).

    Fields:
        direction: Trade direction {-1 (short), 0 (flat), 1 (long)}.
        magnitude: Normalized signal strength in [0.0, 1.0].
        confidence: Ensemble agreement / model confidence in [0.0, 1.0].
        uncertainty: MC Dropout predictive standard deviation.
        expected_decay_steps: Estimated signal half-life in bars.
        regime: Current HMM regime state index.
        timestamp: Signal generation Unix timestamp.
        metadata: Sub-model contributions, raw predictions, diagnostics.
    """
    direction: int
    magnitude: float
    confidence: float
    uncertainty: float
    expected_decay_steps: int
    regime: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field constraints on frozen dataclass."""
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be in {{-1, 0, 1}}, got {self.direction}")
        if not (0.0 <= self.magnitude <= 1.0):
            raise ValueError(f"magnitude must be in [0.0, 1.0], got {self.magnitude}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


class SignalGenerator:
    """
    Converts raw ensemble numerical predictions into structured AlphaSignal objects.

    Maps continuous magnitude predictions to discrete direction via configurable
    thresholds, estimates signal decay from model uncertainty, and attaches
    regime state from the regime classifier.
    """

    def __init__(
        self,
        direction_threshold: float = 0.002,
        decay_lookback: int = 10
    ) -> None:
        self.direction_threshold = direction_threshold
        self.decay_lookback = decay_lookback

        logger.info(
            "SignalGenerator initialized",
            direction_threshold=direction_threshold,
            decay_lookback=decay_lookback
        )

    def generate(
        self,
        prediction: float,
        confidence: float,
        uncertainty: float,
        regime: int,
        sub_model_predictions: Optional[Dict[str, float]] = None
    ) -> AlphaSignal:
        """
        Convert a raw ensemble prediction into a structured AlphaSignal.

        Args:
            prediction: Raw float prediction of forward returns from ensemble.
            confidence: Model confidence / ensemble agreement score in [0, 1].
            uncertainty: MC Dropout predictive standard deviation.
            regime: Current HMM regime state index.
            sub_model_predictions: Dict mapping sub-model names to their raw predictions.

        Returns:
            AlphaSignal with all fields populated.
        """
        # Direction classification
        if prediction > self.direction_threshold:
            direction = 1
        elif prediction < -self.direction_threshold:
            direction = -1
        else:
            direction = 0

        # Magnitude normalization: scale by 3x threshold, cap at 1.0
        magnitude = min(abs(prediction) / (3.0 * self.direction_threshold), 1.0)

        # Clamp confidence to valid range
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Signal decay estimation: higher uncertainty → faster decay
        clamped_uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        expected_decay_steps = max(1, int(round(self.decay_lookback * (1.0 - clamped_uncertainty))))

        # Build metadata
        metadata = {
            "raw_prediction": float(prediction),
            "sub_model_predictions": sub_model_predictions or {}
        }

        signal = AlphaSignal(
            direction=direction,
            magnitude=float(magnitude),
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            expected_decay_steps=expected_decay_steps,
            regime=int(regime),
            timestamp=time.time(),
            metadata=metadata
        )

        logger.debug(
            "AlphaSignal generated",
            direction=signal.direction,
            magnitude=signal.magnitude,
            confidence=signal.confidence,
            uncertainty=signal.uncertainty,
            decay=signal.expected_decay_steps,
            regime=signal.regime
        )

        return signal
