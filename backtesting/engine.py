from abc import ABC, abstractmethod
from typing import Dict, Any, List
import structlog

from backtesting.data_handler import BaseDataHandler
from backtesting.portfolio import BacktestPortfolio

logger = structlog.get_logger()


class BaseBacktestEngine(ABC):
    """
    Abstract Base Class for the historical backtesting environment.
    """

    def __init__(
        self,
        data_handler: BaseDataHandler,
        portfolio: BacktestPortfolio,
        config: Dict[str, Any]
    ) -> None:
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.config = config
        
        self.trades: List[Dict[str, Any]] = []
        
        logger.info(
            "BacktestEngine initialized",
            data_handler=self.data_handler.__class__.__name__,
            portfolio=self.portfolio.__class__.__name__
        )

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Executes the backtest loop and returns performance results.
        """
        pass
