import threading
from typing import Any

import numpy as np
import structlog

from models.meta_learner.maml import MAMLModel

logger = structlog.get_logger()


class OnlineAdapter:
    """
    Live adaptation wrapper for the MAML Meta-Learner.
    Maintains a rolling buffer of recent observations and triggers few-shot
    adaptation of the meta-learned initialization weights.
    """

    def __init__(self, maml_model: MAMLModel, buffer_size: int = 50) -> None:
        self.maml_model = maml_model
        self.buffer_size = buffer_size
        
        self.observations = []
        self.targets = []
        self._lock = threading.Lock()
        
        logger.info(
            "OnlineAdapter initialized",
            buffer_size=self.buffer_size,
            maml_model=self.maml_model.name
        )

    def update(self, observation: Any, target: float) -> None:
        """
        Add a new observation-target pair to the rolling buffer.
        """
        with self._lock:
            self.observations.append(observation)
            self.targets.append(target)
            
            # Trim buffer
            if len(self.observations) > self.buffer_size:
                self.observations = self.observations[-self.buffer_size:]
                self.targets = self.targets[-self.buffer_size:]

    def adapt_now(self) -> None:
        """
        Trigger few-shot adaptation using the current buffer.
        Only adapts if the buffer is full.
        """
        with self._lock:
            if len(self.observations) < self.maml_model.support_size:
                logger.debug(
                    "Skipping adaptation, insufficient data",
                    current=len(self.observations),
                    required=self.maml_model.support_size
                )
                return
                
            X_support = np.array(self.observations[-self.maml_model.support_size:])
            y_support = np.array(self.targets[-self.maml_model.support_size:])
            
        logger.info("Triggering online MAML adaptation")
        self.maml_model.adapt(X_support, y_support)

    def get_adapted_prediction(self, X: Any) -> np.ndarray:
        """
        Get prediction using the currently adapted model state.
        """
        with self._lock:
            return self.maml_model.predict(X)

    def reset(self) -> None:
        """
        Clear the rolling buffer and reset the adapted state back to the meta-initialization.
        """
        with self._lock:
            self.observations.clear()
            self.targets.clear()
            self.maml_model.adapted_model = None
            logger.info("OnlineAdapter buffer reset")
