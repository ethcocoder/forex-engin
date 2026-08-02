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
        Calculate position size using the Kelly criterion with proper stop-loss-based
        forex position sizing.
        
        The risk fraction determines how many DOLLARS to risk, then the position size
        is computed as: position_size = risk_dollars / stop_loss_distance.
        This produces institutional-grade lot sizes instead of tiny fractional units.
        """
        import numpy as np
        
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
        
        # Scale by signal confidence and inverse uncertainty.
        # Magnitude is normalized via sigmoid so tiny forex return predictions
        # (0.0001–0.002) map into a usable 0.3–0.9 range instead of collapsing
        # the risk fraction to dust.
        mag_normalized = 2.0 / (1.0 + np.exp(-signal.magnitude * 2000.0)) - 1.0  # sigmoid → [0, 1)
        mag_normalized = float(np.clip(mag_normalized, 0.1, 1.0))  # floor at 10%
        
        signal_scalar = mag_normalized * signal.confidence * (1.0 - signal.uncertainty)
        adjusted_f = fractional_kelly * signal_scalar
        
        # Enforce minimum risk fraction so positions stay institutional-grade
        min_risk_f = 0.0005  # 0.05% of equity minimum risk per trade
        adjusted_f = max(adjusted_f, min_risk_f)

        # Cap at maximum allowed risk per trade
        final_f = min(adjusted_f, self.max_risk_pct)
        
        # ── Regime-Adaptive Sizing ───────────────────────────────────────────
        # Scale down sizing in volatile and choppy range regimes to protect capital
        regime_multipliers = {0: 1.0, 1: 0.5, 2: 0.35, 3: 0.8}
        regime_mult = regime_multipliers.get(signal.regime, 1.0)
        
        # risk_dollars = how much cash we're willing to lose if stopped out
        risk_dollars = portfolio_state.current_equity * final_f * regime_mult
        
        # Derive stop-loss distance from ATR or rolling volatility
        # ATR represents typical per-bar price movement; 2× ATR is a standard stop distance
        volatility = market_data.get("volatility", 0.0005)
        atr = market_data.get("atr", volatility)  # Fall back to rolling vol if no ATR
        stop_distance = max(atr * 2.0, 0.0002)    # Floor at 2 pips to prevent division blow-up
        
        # position_size = dollars_at_risk / price_distance_to_stop
        # This gives units in base currency (e.g., EUR for EURUSD)
        raw_size = risk_dollars / stop_distance
        
        # Hard caps: never exceed 20× equity notional (institutional leverage limit)
        price = market_data.get("price", market_data.get("close", market_data.get("mid_price", 1.0)))
        max_notional = portfolio_state.current_equity * 20.0  # 20:1 leverage cap
        max_units = max_notional / max(price, 0.01)
        final_size = min(raw_size, max_units)
        
        # Floor at 1 unit to avoid dust orders
        final_size = max(final_size, 1.0)
        
        logger.debug(
            "Kelly size calculated (stop-loss based)",
            pair=pair,
            p=p,
            b=b,
            kelly_f=kelly_f,
            fractional_kelly=fractional_kelly,
            adjusted_f=adjusted_f,
            final_f=final_f,
            risk_dollars=risk_dollars,
            stop_distance=stop_distance,
            raw_size=raw_size,
            final_size=final_size
        )
        
        return final_size
