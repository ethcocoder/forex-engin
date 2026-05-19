from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseRiskEngine(ABC):
    """
    Abstract Base Class for the Risk Engine gating system.
    Decoupled interface that intercepts signals and returns gated/sized orders.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def gate(self, signal: Any, portfolio_state: Any) -> Optional[Any]:
        """
        Evaluate an alpha signal against portfolio risk metrics and circuit breakers.
        Returns a validated and sized Order object, or None if rejected.
        """
        pass
