from typing import Any, Dict

import structlog

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState

logger = structlog.get_logger()


class KellySizer:
    """
    Fractional Kelly Criterion Position Sizing.
    
    Formula: f* = (p * b - q) / b
    Where:
        p = probability of win (win rate)
        q = probability of loss (1 - p)
        b = ratio of average win to average loss
        
    Uses a fractional multiplier to reduce volatility and risk of ruin.
    """

    def __init__(self, fraction: float = 0.25, max_risk_pct: float = 0.05) -> None:
        """
        Args:
            fraction: The Kelly fraction multiplier (e.g., 0.25 for quarter-Kelly).
            max_risk_pct: Hard cap on the maximum percentage of equity to risk per trade.
        """
        self.fraction = fraction
        self.max_risk_pct = max_risk_pct
        logger.info(
            "KellySizer initialized",
            fraction=self.fraction,
            max_risk_pct=self.max_risk_pct
        )

    def calculate_size(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate position size using the Kelly criterion based on portfolio history.
        """
        p = portfolio_state.win_rate
        b = portfolio_state.win_loss_ratio
        
        # Cold start fallback: if win_rate and win_loss_ratio are at initial neutral settings
        if p == 0.5 and b == 1.0:
            kelly_f = 0.20  # 20% Kelly target as standard fallback
        else:
            # Edge cases: no history or degenerate stats
            if p <= 0.0 or b <= 0.0:
                logger.debug("Degenerate Kelly inputs, falling back to minimum risk fraction", p=p, b=b)
                kelly_f = 0.08
            else:
                q = 1.0 - p
                
                # Kelly fraction (f*)
                kelly_f = (p * b - q) / b
            
        if kelly_f <= 0.0:
            logger.debug("Negative or zero Kelly fraction, falling back to minimum risk fraction", kelly_f=kelly_f)
            kelly_f = 0.08  # safe minimum raw Kelly fraction (e.g., 2% risk with 0.25 multiplier)
            

        
        # Apply fractional multiplier
        fractional_kelly = kelly_f * self.fraction
        
        # Scale by signal magnitude, confidence, and inverse uncertainty
        signal_scalar = signal.magnitude * signal.confidence * (1.0 - signal.uncertainty)
        adjusted_f = fractional_kelly * signal_scalar

        # Cap at maximum allowed risk per trade
        final_f = min(adjusted_f, self.max_risk_pct)
        
        final_size = portfolio_state.current_equity * final_f
        
        logger.debug(
            "Kelly size calculated",
            pair=pair,
            p=p,
            b=b,
            kelly_f=kelly_f,
            fractional_kelly=fractional_kelly,
            adjusted_f=adjusted_f,
            final_f=final_f,
            final_size=final_size
        )
        
        return final_size
