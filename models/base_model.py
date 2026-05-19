from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModel(ABC):
    """
    Abstract Base Class that all models in the neural ensemble must inherit.
    Enforces unified interfaces for training, inference, and serialization.
    """

    def __init__(self, name: str, config: Any) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs: Any) -> Any:
        """
        Train the model using inputs X and targets y.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """
        Run forward inference on X and return predictions.
        """
        pass

    @abstractmethod
    def save(self, path: str, **kwargs: Any) -> None:
        """
        Serialize model state and save weights to the specified path.
        """
        pass

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> None:
        """
        Deserialize and load model weights from the specified path.
        """
        pass
