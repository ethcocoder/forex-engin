from typing import Any, Dict, Optional
import time
import structlog

from risk.risk_engine import OrderRequest
from execution.brokers.base_broker import BaseBroker

logger = structlog.get_logger()


class ExecutionEngine:
    """
    Execution Orchestrator (Layer 5).
    Takes a gated OrderRequest from the Risk Engine and routes it to the correct
    broker, optionally handling algorithmic execution (TWAP/VWAP).
    """

    def __init__(self, broker: BaseBroker, router: Any = None) -> None:
        """
        Args:
            broker: The active BaseBroker instance (PaperBroker, OandaBroker, etc).
            router: Optional algorithmic router (SmartRouter, TWAPRouter).
        """
        self.broker = broker
        self.router = router
        
        # Track active pending orders and executed fills
        self.active_orders: Dict[str, Any] = {}
        
        logger.info(
            "ExecutionEngine initialized",
            broker=self.broker.name,
            router=self.router.__class__.__name__ if self.router else "Direct"
        )

    def execute(self, order: OrderRequest) -> bool:
        """
        Process the approved OrderRequest.
        If a router is attached, hands off to the router. Otherwise executes directly.
        
        Returns:
            True if successfully submitted, False otherwise.
        """
        logger.info(
            "ExecutionEngine received order",
            pair=order.pair,
            direction=order.direction,
            size=order.size
        )
        
        if self.router:
            # Algorithmic routing (e.g. slicing into TWAP)
            logger.debug("Delegating order to router")
            return self.router.route(order, self.broker)
        else:
            # Direct execution
            return self._execute_direct(order)
            
    def _execute_direct(self, order: OrderRequest, max_retries: int = 3) -> bool:
        """
        Executes a trade directly on the broker with exponential backoff retry.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # Place order on broker
                result = self.broker.place_order(order)
                
                if result and result.get("status") in ["FILLED", "PENDING"]:
                    logger.info(
                        "Order successfully placed",
                        pair=order.pair,
                        size=order.size,
                        broker_status=result.get("status")
                    )
                    return True
                else:
                    logger.warning(
                        "Order rejected by broker",
                        pair=order.pair,
                        result=result
                    )
                    return False
                    
            except Exception as e:
                attempt += 1
                wait_time = 2 ** attempt
                logger.error(
                    "Network error during execution, retrying...",
                    attempt=attempt,
                    wait_time=wait_time,
                    error=str(e)
                )
                time.sleep(wait_time)
                
        logger.critical("Max retries exceeded, order execution failed", pair=order.pair)
        return False

    def sync_portfolio_state(self) -> Dict[str, float]:
        """
        Re-synchronizes internal portfolio state with the true broker state.
        This is critical after a crash to avoid state desync.
        """
        try:
            positions = self.broker.get_positions()
            logger.info("Portfolio state synced with broker", positions=positions)
            return positions
        except Exception as e:
            logger.error("Failed to sync portfolio state", error=str(e))
            return {}
