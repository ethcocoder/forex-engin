import time
import threading
from typing import Any, Dict, Optional
import structlog

from risk.risk_engine import OrderRequest
from execution.brokers.base_broker import BaseBroker

logger = structlog.get_logger()


class IcebergRouter:
    """
    Iceberg Order Routing Algorithm.
    
    Splits a large parent order into a visible slice (display size) and
    a hidden remainder. The router submits a child order for the display size,
    and only submits the next slice once the broker confirms the previous slice
    has been fully filled. This masks the true order volume from the order book.
    """

    def __init__(
        self,
        display_size: float = 50000.0,
        poll_interval: float = 0.5,
        timeout: float = 300.0
    ) -> None:
        """
        Args:
            display_size: The maximum visible size submitted as a single child order.
            poll_interval: How often (in seconds) to check order execution status for pending fills.
            timeout: Maximum time (in seconds) allowed for the entire iceberg order to execute.
        """
        self.display_size = max(1.0, display_size)
        self.poll_interval = max(0.1, poll_interval)
        self.timeout = max(1.0, timeout)
        
        logger.info(
            "IcebergRouter initialized",
            display_size=self.display_size,
            poll_interval=self.poll_interval,
            timeout=self.timeout
        )

    def route(self, order: OrderRequest, broker: BaseBroker) -> bool:
        """
        Accepts the parent order, starts a background execution thread, and returns True immediately.
        """
        if order.size <= self.display_size:
            logger.info("Order size below display size, placing directly", size=order.size)
            res = broker.place_order(order)
            return res is not None and res.get("status") in ["FILLED", "PENDING"]

        iceberg_thread = threading.Thread(
            target=self._execute_iceberg,
            args=(order, broker),
            daemon=True
        )
        iceberg_thread.start()
        
        logger.info(
            "Iceberg routing started in background",
            pair=order.pair,
            parent_size=order.size,
            display_size=self.display_size
        )
        return True

    def _execute_iceberg(self, parent_order: OrderRequest, broker: BaseBroker) -> None:
        remaining_size = parent_order.size
        start_time = time.time()
        slice_idx = 1

        while remaining_size > 0:
            if time.time() - start_time > self.timeout:
                logger.error("Iceberg parent order timed out", pair=parent_order.pair, remaining=remaining_size)
                break

            current_slice_size = min(self.display_size, remaining_size)

            # Create child order
            child_order = OrderRequest(
                pair=parent_order.pair,
                direction=parent_order.direction,
                size=current_slice_size,
                order_type=parent_order.order_type,
                limit_price=parent_order.limit_price,
                stop_loss=parent_order.stop_loss,
                take_profit=parent_order.take_profit,
                metadata={**parent_order.metadata, "iceberg_slice": slice_idx}
            )

            logger.debug("Placing iceberg slice", slice_idx=slice_idx, size=current_slice_size)

            try:
                res = broker.place_order(child_order)
                if not res:
                    logger.error("Broker failed to accept iceberg slice, aborting execution", slice_idx=slice_idx)
                    break

                status = res.get("status")

                if status == "FILLED":
                    # Instant fill (e.g. market order or simulated environment)
                    remaining_size -= current_slice_size
                    logger.debug("Iceberg slice filled immediately", slice_idx=slice_idx, size=current_slice_size)
                    slice_idx += 1
                elif status == "PENDING":
                    # Wait for fill
                    filled = self._wait_for_fill(broker, parent_order.pair, current_slice_size, start_time)
                    if filled:
                        remaining_size -= current_slice_size
                        logger.debug("Iceberg slice filled after polling", slice_idx=slice_idx, size=current_slice_size)
                        slice_idx += 1
                    else:
                        logger.error("Iceberg slice execution failed or timed out during wait", slice_idx=slice_idx)
                        break
                else:
                    logger.error("Iceberg slice rejected or failed", status=status)
                    break
            except Exception as e:
                logger.error("Exception placing iceberg slice", error=str(e))
                break

        logger.info("Iceberg order execution run completed", pair=parent_order.pair, remaining=remaining_size)

    def _wait_for_fill(self, broker: BaseBroker, pair: str, target_size: float, parent_start_time: float) -> bool:
        """
        Polls broker state to determine when a slice is filled.
        """
        slice_start_time = time.time()
        
        try:
            initial_positions = broker.get_positions()
            initial_pos = initial_positions.get(pair, 0.0)
        except Exception as e:
            logger.error("Failed to get initial positions for iceberg check", error=str(e))
            initial_pos = 0.0

        while True:
            if time.time() - parent_start_time > self.timeout:
                return False
            if time.time() - slice_start_time > 60.0:  # Individual slice timeout
                return False

            time.sleep(self.poll_interval)

            try:
                current_positions = broker.get_positions()
                current_pos = current_positions.get(pair, 0.0)

                # Check if position has changed by approximately the target size in target direction
                if abs(current_pos - initial_pos) >= (target_size * 0.99):
                    return True
            except Exception as e:
                logger.error("Error polling position for iceberg fill check", error=str(e))


# Legacy alias for compatibility with execution.routing imports
IcebergOrder = IcebergRouter
