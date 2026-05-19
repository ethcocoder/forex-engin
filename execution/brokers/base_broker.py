from abc import ABC, abstractmethod
from typing import Any, List


class BaseBroker(ABC):
    """
    Abstract Base Class representing an external broker connection.
    Guarantees standard trade interaction and connection handling.
    """

    def __init__(self, name: str, config: Any) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the broker's endpoints (WebSocket + REST).
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Safely terminate broker connections and clear listeners.
        """
        pass

    @abstractmethod
    def place_order(self, order: Any) -> Any:
        """
        Submit a trade order (Market, Limit, Stop, etc.) to the broker.
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Any]:
        """
        Query all active open positions in the brokerage account.
        """
        pass

    @abstractmethod
    def get_account_balance(self) -> float:
        """
        Retrieve current cash balance or net asset value of the account.
        """
        pass
