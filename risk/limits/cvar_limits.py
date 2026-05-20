from typing import Any, Dict

import numpy as np
import structlog

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState

logger = structlog.get_logger()


class CVaRFilter:
    """
    Conditional Value at Risk (Expected Shortfall) Filter.
    
    Computes historical simulation CVaR at a specified confidence level.
    If the portfolio's current CVaR exceeds the risk budget, the filter
    rejects risk-increasing signals.
    """

    def __init__(self, confidence_level: float = 0.95, max_cvar_pct: float = 0.05) -> None:
        """
        Args:
            confidence_level: The confidence level for CVaR (e.g., 0.95 means worst 5%).
            max_cvar_pct: The maximum allowed Expected Shortfall as a percentage of equity.
        """
        self.confidence_level = confidence_level
        self.max_cvar_pct = max_cvar_pct
        logger.info(
            "CVaRFilter initialized",
            confidence=self.confidence_level,
            max_cvar=self.max_cvar_pct
        )

    def check(self, signal: AlphaSignal, pair: str, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> bool:
        """
        Evaluate if the current portfolio risk allows for new exposure.
        """
        returns = portfolio_state.historical_returns
        
        if returns is None or len(returns) < 100:
            logger.debug("Insufficient history for CVaR, defaulting to Pass")
            return True
            
        # Sort returns ascending (worst to best)
        sorted_returns = np.sort(returns)
        
        # Identify the cutoff index for the tail
        cutoff_idx = int((1.0 - self.confidence_level) * len(sorted_returns))
        cutoff_idx = max(1, cutoff_idx)
        
        # Calculate CVaR (mean of returns below the cutoff)
        cvar = np.mean(sorted_returns[:cutoff_idx])
        
        # cvar is typically negative. We compare its absolute value to max_cvar_pct.
        current_cvar_pct = abs(cvar)
        
        if current_cvar_pct > self.max_cvar_pct:
            logger.warning(
                "CVaR limit breached",
                current_cvar=current_cvar_pct,
                limit=self.max_cvar_pct
            )
            return False
            
        return True
