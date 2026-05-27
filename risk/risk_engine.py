from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from models.ensemble.signal_generator import AlphaSignal

logger = structlog.get_logger()


@dataclass
class PortfolioState:
    """
    Represents the real-time state of the portfolio.
    Passed to the Risk Engine to evaluate circuit breakers and sizing limits.
    """
    current_equity: float
    open_positions: Dict[str, float]  # pair -> current_size
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    win_rate: float
    win_loss_ratio: float
    historical_returns: np.ndarray  # rolling window of recent returns for CVaR


@dataclass
class OrderRequest:
    """
    The final approved order resulting from the Risk Engine.
    """
    pair: str
    direction: int
    size: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRiskEngine(ABC):
    """
    Abstract Base Class for the Risk Engine gating system.
    """
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def gate(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> Optional[OrderRequest]:
        pass


class RiskEngine(BaseRiskEngine):
    """
    Concrete Risk Engine Orchestrator.
    Executes a rigid pipeline: Filters -> Limits -> Sizing.
    """

    def __init__(self, config: Any = None) -> None:
        config = config or {}
        super().__init__(config)
        
        self.filters = []
        self.limits = []
        self.sizer = None
        
        logger.info("RiskEngine orchestrator initialized")

    def register_filter(self, filter_obj: Any) -> None:
        """Add a pass/fail liquidity or session filter."""
        self.filters.append(filter_obj)

    def register_limit(self, limit_obj: Any) -> None:
        """Add a circuit breaker limit (Drawdown, CVaR, Correlation)."""
        self.limits.append(limit_obj)

    def set_sizer(self, sizer_obj: Any) -> None:
        """Set the position sizing algorithm."""
        self.sizer = sizer_obj

    def gate(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> Optional[OrderRequest]:
        """
        Evaluate an AlphaSignal through the full risk pipeline.
        
        Returns:
            OrderRequest if approved, None if rejected.
        """
        # 1. If signal is flat, we don't need to open a new order,
        # but we might need to close. (Closing logic is handled by execution manager).
        if signal.direction == 0:
            return None

        # 2. Hard Filters (Session, Liquidity)
        for f in self.filters:
            if not f.check(signal, pair, market_data):
                logger.debug(f"Signal rejected by filter: {f.__class__.__name__}")
                return None

        # 3. Circuit Breakers & Limits
        # If any limit is breached, we only allow risk-reducing trades (exits).
        current_exposure = portfolio_state.open_positions.get(pair, 0.0)
        is_risk_increasing = (signal.direction > 0 and current_exposure >= 0) or \
                             (signal.direction < 0 and current_exposure <= 0)
        
        for limit in self.limits:
            if not limit.check(signal, pair, portfolio_state, market_data):
                if is_risk_increasing:
                    logger.warning(
                        "Risk increasing signal rejected by limit",
                        limit=limit.__class__.__name__,
                        pair=pair,
                        direction=signal.direction
                    )
                    return None
                else:
                    logger.info("Risk reducing signal permitted despite limit breach")
                    break

        # 4. Position Sizing
        if self.sizer is None:
            logger.error("No position sizer configured in RiskEngine")
            return None
            
        proposed_size = self.sizer.calculate_size(signal, pair, portfolio_state, market_data)
        
        if proposed_size <= 0.0:
            logger.debug("Signal rejected: Calculated size is 0 or negative")
            return None

        # 5. Net Exposure Cap — prevent position overaccumulation
        # The proposed_size represents the TARGET total exposure, not an additive increment.
        # Only trade the difference between current exposure and target.
        current_exposure = portfolio_state.open_positions.get(pair, 0.0)
        if signal.direction > 0:
            # Want net long of proposed_size; if already long, only add the gap
            trade_size = max(0.0, proposed_size - max(current_exposure, 0.0))
        else:
            # Want net short of proposed_size; if already short, only add the gap
            trade_size = max(0.0, proposed_size - abs(min(current_exposure, 0.0)))

        if trade_size < 1.0:
            logger.debug("Signal rejected: Net exposure already at or above target",
                         proposed_size=proposed_size, current_exposure=current_exposure)
            return None

        # Create the final approved order
        order = OrderRequest(
            pair=pair,
            direction=signal.direction,
            size=trade_size,
            order_type="MARKET",
            metadata={"source": "EnsembleAggregator", "signal_confidence": signal.confidence}
        )
        
        logger.info(
            "Signal approved by RiskEngine",
            pair=pair,
            direction=order.direction,
            size=order.size
        )
        
        return order
