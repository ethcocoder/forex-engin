from datetime import datetime, time
from typing import Any, Dict, Optional

import structlog

from models.ensemble.signal_generator import AlphaSignal

logger = structlog.get_logger()


class SessionFilter:
    """
    Forex Session Filter.
    
    Rejects trades during illiquid or highly erratic periods, such as the
    daily rollover (e.g., 21:00-22:00 UTC) or weekends.
    """

    def __init__(
        self,
        rollover_start: time = time(21, 0),
        rollover_end: time = time(22, 0),
        trade_weekends: bool = False
    ) -> None:
        """
        Args:
            rollover_start: UTC time for the start of the illiquid rollover period.
            rollover_end: UTC time for the end of the rollover period.
            trade_weekends: Whether to permit trading on Saturday/Sunday.
        """
        self.rollover_start = rollover_start
        self.rollover_end = rollover_end
        self.trade_weekends = trade_weekends
        
        logger.info(
            "SessionFilter initialized",
            rollover_window=f"{self.rollover_start}-{self.rollover_end}",
            trade_weekends=self.trade_weekends
        )

    def check(self, signal: AlphaSignal, pair: str, portfolio_state_or_market_data: Any, market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Evaluate if the current UTC time is safe for trading.
        """
        if market_data is None:
            market_data = portfolio_state_or_market_data
            
        # Convert signal timestamp to UTC datetime
        dt = datetime.utcfromtimestamp(signal.timestamp)
        
        # Weekend check
        # weekday(): Monday is 0, Sunday is 6. Saturday is 5.
        if not self.trade_weekends and dt.weekday() >= 5:
            logger.debug("Signal rejected: Weekend trading disabled", current_day=dt.weekday())
            return False
            
        # Rollover check
        current_time = dt.time()
        
        if self.rollover_start <= current_time <= self.rollover_end:
            logger.debug(
                "Signal rejected: Rollover illiquidity window",
                current_time=current_time
            )
            return False
            
        return True
