from typing import Any, Dict

import structlog

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState

logger = structlog.get_logger()


class CorrelationFilter:
    """
    Cross-Pair Correlation Limits.
    
    Prevents taking trades that would significantly increase exposure to highly
    correlated assets. If a proposed trade is in the same direction as an existing
    open position, and the correlation between the two pairs is > threshold,
    the trade is rejected.
    """

    def __init__(self, max_correlation: float = 0.75) -> None:
        """
        Args:
            max_correlation: The maximum allowed Pearson correlation between two
                concurrently open pairs in the same directional exposure.
        """
        self.max_correlation = max_correlation
        logger.info("CorrelationFilter initialized", max_correlation=self.max_correlation)

    def check(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate correlation exposure.
        """
        correlation_matrix = market_data.get("correlation_matrix", {})
        if not portfolio_state.open_positions:
            return True
            
        proposed_dir = signal.direction
        
        for open_pair, open_size in portfolio_state.open_positions.items():
            if open_pair == pair or open_size == 0.0:
                continue
                
            open_dir = 1 if open_size > 0 else -1
            
            # We only care if we are adding exposure in the correlated direction
            # If going Long A and Long B, and A & B are positively correlated -> Risk
            # If going Long A and Short B, and A & B are negatively correlated -> Risk
            
            corr = correlation_matrix.get(pair, {}).get(open_pair, 0.0)
            
            if proposed_dir == open_dir and corr > self.max_correlation:
                logger.warning(
                    "Correlation limit breached (positive correlation)",
                    pair1=pair,
                    pair2=open_pair,
                    correlation=corr,
                    limit=self.max_correlation
                )
                return False
                
            if proposed_dir != open_dir and corr < -self.max_correlation:
                logger.warning(
                    "Correlation limit breached (negative correlation)",
                    pair1=pair,
                    pair2=open_pair,
                    correlation=corr,
                    limit=-self.max_correlation
                )
                return False
                
        return True
