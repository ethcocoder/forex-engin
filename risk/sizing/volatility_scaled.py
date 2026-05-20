from typing import Any, Dict

import structlog

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState

logger = structlog.get_logger()


class VolatilitySizer:
    """
    Volatility Scaled Position Sizing.
    
    Sizes positions inversely proportional to the asset's current volatility,
    ensuring a constant risk profile across different market conditions.
    """

    def __init__(self, risk_pct: float = 0.01) -> None:
        """
        Args:
            risk_pct: The target risk percentage of equity per trade (e.g., 0.01 = 1%).
        """
        self.risk_pct = risk_pct
        logger.info("VolatilitySizer initialized", risk_pct=self.risk_pct)

    def calculate_size(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate size based on equity, target risk, and current volatility.
        
        Formula: Size = (Equity * Risk_Pct) / (Volatility * Point_Value)
        """
        # Extract volatility from market_data (e.g., ATR or standard deviation)
        volatility = market_data.get("volatility")
        point_value = market_data.get("point_value", 1.0)
        
        if volatility is None or volatility <= 0.0:
            logger.warning("Missing or invalid volatility data, returning 0 size", pair=pair)
            return 0.0
            
        risk_amount = portfolio_state.current_equity * self.risk_pct
        
        # Base size
        base_size = risk_amount / (volatility * point_value)
        
        # Scale by signal magnitude and inverse uncertainty
        signal_scalar = signal.magnitude * (1.0 - signal.uncertainty)
        final_size = base_size * signal_scalar
        
        logger.debug(
            "Volatility size calculated",
            pair=pair,
            equity=portfolio_state.current_equity,
            volatility=volatility,
            risk_amount=risk_amount,
            signal_scalar=signal_scalar,
            final_size=final_size
        )
        
        return final_size
