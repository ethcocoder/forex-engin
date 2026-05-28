import time
import threading
import random
from typing import Any, List
import structlog

from risk.risk_engine import OrderRequest
from execution.brokers.base_broker import BaseBroker

logger = structlog.get_logger()


class TWAPRouter:
    """
    Time-Weighted Average Price (TWAP) Execution Algo.
    
    Splits a large parent order into N smaller child slices, executing them
    over a specified time window. Uses randomization on both slice size and
    interval timing to mask algorithmic footprint from institutional hunters.
    """

    def __init__(self, slices: int = 5, duration_seconds: int = 60, randomize: bool = True) -> None:
        """
        Args:
            slices: Number of child orders to split the parent order into.
            duration_seconds: Total time window to execute all slices.
            randomize: If True, staggers slice sizes and sleep times by +/- 20%.
        """
        self.slices = max(1, slices)
        self.duration_seconds = max(1, duration_seconds)
        self.randomize = randomize
        
        self.base_interval = self.duration_seconds / self.slices
        
        logger.info(
            "TWAPRouter initialized",
            slices=self.slices,
            duration=self.duration_seconds,
            randomize=self.randomize
        )

    def route(self, order: OrderRequest, broker: BaseBroker) -> bool:
        """
        Takes the parent order and spawns a background thread to execute the TWAP.
        Returns True immediately indicating the TWAP was accepted and started.
        """
        if self.slices == 1:
            # Degenerate case, just execute immediately
            return broker.place_order(order) is not None
            
        # Compute slices
        child_sizes = self._compute_slice_sizes(order.size)
        
        # Fire and forget thread
        twap_thread = threading.Thread(
            target=self._execute_slices,
            args=(order, child_sizes, broker),
            daemon=True
        )
        twap_thread.start()
        
        logger.info(
            "TWAP execution started in background",
            pair=order.pair,
            parent_size=order.size,
            slices=self.slices
        )
        return True

    def _compute_slice_sizes(self, total_size: float) -> List[float]:
        base_size = total_size / self.slices
        
        if not self.randomize:
            return [base_size] * self.slices
            
        # Randomized slicing: +/- 20% variance while maintaining total sum
        sizes = []
        remaining_size = total_size
        
        for i in range(self.slices - 1):
            variance = random.uniform(0.8, 1.2)
            slice_size = base_size * variance
            
            # Ensure we don't overshoot before the last slice
            slice_size = min(slice_size, remaining_size * 0.9)
            sizes.append(slice_size)
            remaining_size -= slice_size
            
        # The final slice cleans up whatever is left exactly
        sizes.append(remaining_size)
        return sizes

    def _execute_slices(self, parent_order: OrderRequest, child_sizes: List[float], broker: BaseBroker) -> None:
        """
        The background loop that executes the child orders.
        """
        for i, size in enumerate(child_sizes):
            # Create child order
            child = OrderRequest(
                pair=parent_order.pair,
                direction=parent_order.direction,
                size=size,
                order_type=parent_order.order_type,
                metadata={**parent_order.metadata, "twap_slice": i+1, "twap_total": self.slices}
            )
            
            # Execute
            try:
                result = broker.place_order(child)
                logger.debug(
                    "TWAP slice executed",
                    slice_idx=i+1,
                    size=size,
                    status=result.get("status") if result else "FAILED"
                )
            except Exception as e:
                logger.error("TWAP slice failed", slice_idx=i+1, error=str(e))
                
            # Sleep until next slice (unless it's the last one)
            if i < len(child_sizes) - 1:
                sleep_time = self.base_interval
                if self.randomize:
                    # Randomize interval by +/- 20%
                    sleep_time *= random.uniform(0.8, 1.2)
                    
                time.sleep(sleep_time)
                
        logger.info("TWAP execution completed", pair=parent_order.pair)


# Legacy alias for compatibility with execution.routing imports
TWAPOrder = TWAPRouter
