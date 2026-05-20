import time
import datetime
import random
import threading
from typing import Any, List, Dict, Optional
import structlog

from risk.risk_engine import OrderRequest
from execution.brokers.base_broker import BaseBroker

logger = structlog.get_logger()

# Intraday hourly volume profile (0-23 UTC) representing institutional FX volume
# Peaks during London open (08:00-10:00 UTC) and US overlap (12:00-16:00 UTC)
HISTORICAL_VOLUME_PROFILE = [
    0.015, 0.015, 0.018, 0.020, 0.022, 0.025,  # 00:00 - 05:00 UTC (Asian Session)
    0.035, 0.055, 0.075, 0.080, 0.070, 0.060,  # 06:00 - 11:00 UTC (European Open / London)
    0.085, 0.095, 0.090, 0.075, 0.050, 0.035,  # 12:00 - 17:00 UTC (US Open / Overlap)
    0.025, 0.020, 0.018, 0.015, 0.012, 0.010   # 18:00 - 23:00 UTC (Sydney / Tokyo Close)
]


class VWAPRouter:
    """
    Volume-Weighted Average Price (VWAP) Execution Router.
    
    Slices a large parent order into N smaller child orders to match the
    historical intraday volume distribution curve of the global FX market.
    Helps minimize price impact when executing large block trades.
    """

    def __init__(
        self,
        slices: int = 5,
        duration_seconds: int = 3600,
        randomize: bool = True,
        volume_profile: Optional[List[float]] = None
    ) -> None:
        """
        Args:
            slices: Number of child slices to segment the parent order.
            duration_seconds: Total time window to distribute execution.
            randomize: Whether to add random variance (+/- 15%) to mask footprints.
            volume_profile: Optional custom 24-item list representing hourly volume weights.
        """
        self.slices = max(1, slices)
        self.duration_seconds = max(1, duration_seconds)
        self.randomize = randomize
        self.volume_profile = volume_profile or HISTORICAL_VOLUME_PROFILE
        
        # Verify custom volume profile validity
        if len(self.volume_profile) != 24:
            logger.warning("Custom volume profile size is not 24, falling back to standard profile")
            self.volume_profile = HISTORICAL_VOLUME_PROFILE
            
        self.base_interval = self.duration_seconds / self.slices
        
        logger.info(
            "VWAPRouter initialized",
            slices=self.slices,
            duration=self.duration_seconds,
            randomize=self.randomize
        )

    def route(self, order: OrderRequest, broker: BaseBroker) -> bool:
        """
        Accepts the parent order, schedules slices in a background thread, and returns True immediately.
        """
        if self.slices == 1:
            logger.info("VWAP only contains 1 slice, executing directly")
            res = broker.place_order(order)
            return res is not None and res.get("status") in ["FILLED", "PENDING"]

        child_sizes = self._compute_vwap_slices(order.size)

        vwap_thread = threading.Thread(
            target=self._execute_slices,
            args=(order, child_sizes, broker),
            daemon=True
        )
        vwap_thread.start()

        logger.info(
            "VWAP execution started in background",
            pair=order.pair,
            parent_size=order.size,
            slices=self.slices,
            duration_sec=self.duration_seconds
        )
        return True

    def _compute_vwap_slices(self, total_size: float) -> List[float]:
        """
        Calculates slice sizes weighted by the volume profile beginning at the current UTC hour.
        """
        current_utc_hour = datetime.datetime.now(datetime.timezone.utc).hour
        
        # Calculate the corresponding UTC hour for each slice index
        weights = []
        for i in range(self.slices):
            # Calculate execution hour offset based on interval
            hour_offset = int((i * self.base_interval) / 3600.0)
            slice_hour = (current_utc_hour + hour_offset) % 24
            weights.append(self.volume_profile[slice_hour])

        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight <= 0.0:
            normalized_weights = [1.0 / self.slices] * self.slices
        else:
            normalized_weights = [w / total_weight for w in weights]

        # Distribute total size
        sizes = []
        remaining_size = total_size

        for i in range(self.slices - 1):
            slice_size = total_size * normalized_weights[i]
            
            if self.randomize:
                # Add +/- 15% random variance while keeping average clean
                variance = random.uniform(0.85, 1.15)
                slice_size *= variance

            # Clamp slice size to make sure we don't overshoot
            slice_size = min(slice_size, remaining_size * 0.9)
            sizes.append(slice_size)
            remaining_size -= slice_size

        # Last slice fills the remaining size exactly
        sizes.append(remaining_size)
        return sizes

    def _execute_slices(self, parent_order: OrderRequest, child_sizes: List[float], broker: BaseBroker) -> None:
        """
        Executes child orders sequentially on the background thread.
        """
        for i, size in enumerate(child_sizes):
            child = OrderRequest(
                pair=parent_order.pair,
                direction=parent_order.direction,
                size=size,
                order_type=parent_order.order_type,
                limit_price=parent_order.limit_price,
                stop_loss=parent_order.stop_loss,
                take_profit=parent_order.take_profit,
                metadata={**parent_order.metadata, "vwap_slice": i+1, "vwap_total": self.slices}
            )

            try:
                res = broker.place_order(child)
                logger.debug(
                    "VWAP slice executed",
                    slice_idx=i+1,
                    size=size,
                    status=res.get("status") if res else "FAILED"
                )
            except Exception as e:
                logger.error("VWAP slice execution failed", slice_idx=i+1, error=str(e))

            # Sleep between slices, except for the last one
            if i < len(child_sizes) - 1:
                sleep_time = self.base_interval
                if self.randomize:
                    # Randomize interval by +/- 15% to mask footprint
                    sleep_time *= random.uniform(0.85, 1.15)
                time.sleep(sleep_time)

        logger.info("VWAP execution finished", pair=parent_order.pair)
