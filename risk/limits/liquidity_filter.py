from typing import Any, Dict

import structlog

from models.ensemble.signal_generator import AlphaSignal

logger = structlog.get_logger()


class SpreadFilter:
    """
    Liquidity Gating via Bid/Ask Spread.
    
    Rejects signals if the real-time spread exceeds a pair-specific threshold,
    preventing execution during extreme market volatility or illiquidity.
    """

    def __init__(self, default_max_spread_pips: float = 3.0, pair_overrides: Dict[str, float] = None) -> None:
        """
        Args:
            default_max_spread_pips: Default tolerable spread in pips.
            pair_overrides: Specific max spreads for exotic/minor pairs (e.g., {'GBPJPY': 5.0}).
        """
        self.default_max_spread = default_max_spread_pips
        self.pair_overrides = pair_overrides or {}
        
        logger.info(
            "SpreadFilter initialized",
            default_max_spread=self.default_max_spread,
            overrides=self.pair_overrides
        )

    def check(self, signal: AlphaSignal, pair: str, market_data: Dict[str, Any]) -> bool:
        """
        Evaluate real-time spread.
        """
        # We expect market_data to contain current L1 book or explicit spread
        current_spread_pips = market_data.get("spread_pips")
        
        if current_spread_pips is None:
            logger.warning("Spread data missing in market_data, failing filter safe", pair=pair)
            return False
            
        max_allowed = self.pair_overrides.get(pair, self.default_max_spread)
        
        if current_spread_pips > max_allowed:
            logger.debug(
                "Signal rejected: Spread too wide",
                pair=pair,
                spread=current_spread_pips,
                max_allowed=max_allowed
            )
            return False
            
        return True
