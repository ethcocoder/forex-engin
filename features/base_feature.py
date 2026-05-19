from abc import ABC, abstractmethod
from typing import Any


class BaseFeature(ABC):
    """
    Abstract Base Class that all feature extractors must inherit.
    Enforces validation and deterministic computation of features.
    """

    def __init__(self, name: str, config: Any) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def compute(self, df: Any, **kwargs: Any) -> Any:
        """
        Compute features based on historical market data (e.g., pandas DataFrame).
        Should return computed features as a Series or DataFrame.
        """
        pass

    @abstractmethod
    def validate(self, df: Any) -> bool:
        """
        Validate inputs before calculation.
        Checks for missing columns, data gaps, or shape mismatch.
        Should raise a ValueError or return False on invalid schemas.
        """
        pass
