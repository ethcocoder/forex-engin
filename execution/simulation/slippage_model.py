import numpy as np
import structlog

logger = structlog.get_logger()


class SlippageModel:
    """
    Simulates real-world execution slippage for the PaperBroker.
    
    Total slippage is calculated as:
    Base Spread + Market Impact + Random Noise
    """

    def __init__(self, volatility_scalar: float = 0.5, noise_std: float = 0.1) -> None:
        """
        Args:
            volatility_scalar: Scales slippage based on current market volatility.
            noise_std: Standard deviation of the random noise added to simulate latency jitter.
        """
        self.volatility_scalar = volatility_scalar
        self.noise_std = noise_std
        
        logger.info(
            "SlippageModel initialized",
            volatility_scalar=self.volatility_scalar,
            noise_std=self.noise_std
        )

    def calculate_slippage(
        self,
        base_spread_pips: float,
        market_impact_pips: float,
        volatility: float = 1.0
    ) -> float:
        """
        Calculates the expected slippage in pips.
        
        Args:
            base_spread_pips: The current Level 1 bid/ask spread.
            market_impact_pips: The calculated impact of our order size.
            volatility: Current market volatility multiplier (default 1.0).
            
        Returns:
            Total slippage penalty in pips (always positive).
        """
        # Half the spread is the guaranteed execution penalty
        spread_penalty = base_spread_pips / 2.0
        
        # Volatility multiplier (higher vol = wider simulated execution)
        vol_penalty = market_impact_pips * (1.0 + (volatility * self.volatility_scalar))
        
        # Random noise to simulate millisecond latency differences (clamped >= 0)
        noise = float(np.random.normal(0, self.noise_std))
        noise_penalty = max(0.0, noise)
        
        total_slippage = spread_penalty + vol_penalty + noise_penalty
        
        logger.debug(
            "Slippage calculated",
            spread=spread_penalty,
            impact=vol_penalty,
            noise=noise_penalty,
            total=total_slippage
        )
        
        return total_slippage
