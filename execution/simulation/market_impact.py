import numpy as np
import structlog

logger = structlog.get_logger()


class MarketImpactModel:
    """
    Simulates the price impact of trading large volumes.
    Uses a standard Square Root impact model:
    Impact = scalar * volatility * sqrt(Order Size / Average Daily Volume)
    """

    def __init__(self, impact_scalar: float = 0.1) -> None:
        """
        Args:
            impact_scalar: Calibration constant for the specific asset class.
        """
        self.impact_scalar = impact_scalar
        logger.info("MarketImpactModel initialized", impact_scalar=self.impact_scalar)

    def calculate_impact(
        self,
        order_size: float,
        average_daily_volume: float,
        volatility: float = 1.0
    ) -> float:
        """
        Calculate expected market impact in pips.
        """
        if average_daily_volume <= 0.0:
            logger.warning("Invalid ADV provided, defaulting to zero impact")
            return 0.0
            
        if order_size <= 0.0:
            return 0.0
            
        # The fraction of the market our order represents
        participation_rate = order_size / average_daily_volume
        
        # Square root law of market impact
        impact = self.impact_scalar * volatility * np.sqrt(participation_rate)
        
        # Scale to pips (assuming impact scalar is calibrated for this)
        # For a standard FX pair, this scalar is usually around 0.1 to 0.5
        
        logger.debug(
            "Market impact calculated",
            size=order_size,
            adv=average_daily_volume,
            participation=participation_rate,
            impact_pips=impact
        )
        
        return float(impact)
