from typing import Any, Dict

import structlog

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState

logger = structlog.get_logger()


class FixedFractionalSizer:
    """
    Fixed Fractional Position Sizing.
    Sizes every trade to a fixed fraction of the current account equity.
    """

    def __init__(self, fraction: float = 0.02) -> None:
        """
        Args:
            fraction: The fixed fraction of equity to risk per trade (e.g., 0.02 = 2%).
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0.0, 1.0], got {fraction}")
            
        self.fraction = fraction
        logger.info("FixedFractionalSizer initialized", fraction=self.fraction)

    def calculate_size(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculates position size as a simple fraction of equity, modulated by signal magnitude.
        """
        base_size = portfolio_state.current_equity * self.fraction
        
        # Scale down by the signal's magnitude [0.0, 1.0]
        final_size = base_size * signal.magnitude
        
        logger.debug(
            "Fixed fractional size calculated",
            pair=pair,
            equity=portfolio_state.current_equity,
            base_size=base_size,
            magnitude=signal.magnitude,
            final_size=final_size
        )
        
        return final_size
