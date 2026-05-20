from typing import Any
import structlog

from risk.risk_engine import OrderRequest
from execution.brokers.base_broker import BaseBroker
from execution.routing.twap import TWAPRouter

logger = structlog.get_logger()


class SmartRouter:
    """
    Smart Order Router.
    
    Dynamically decides the execution algorithm based on order size.
    Small orders -> Direct Market Execution.
    Large orders -> TWAP Slicing.
    """

    def __init__(self, large_order_threshold: float = 500000.0, twap_config: dict = None) -> None:
        """
        Args:
            large_order_threshold: Order size above which TWAP is triggered.
            twap_config: Configuration dictionary for the TWAP router fallback.
        """
        self.large_order_threshold = large_order_threshold
        twap_config = twap_config or {"slices": 5, "duration_seconds": 60, "randomize": True}
        
        self.twap = TWAPRouter(**twap_config)
        
        logger.info(
            "SmartRouter initialized",
            large_order_threshold=self.large_order_threshold
        )

    def route(self, order: OrderRequest, broker: BaseBroker) -> bool:
        """
        Route the order intelligently.
        """
        if order.size >= self.large_order_threshold:
            logger.info("Order exceeds threshold, routing to TWAP", size=order.size, threshold=self.large_order_threshold)
            return self.twap.route(order, broker)
        else:
            logger.debug("Order below threshold, direct execution", size=order.size)
            result = broker.place_order(order)
            return result is not None and result.get("status") in ["FILLED", "PENDING"]
