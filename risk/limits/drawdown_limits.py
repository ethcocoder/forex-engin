from typing import Any, Dict
import structlog

from models.ensemble.signal_generator import AlphaSignal
from risk.risk_engine import PortfolioState

logger = structlog.get_logger()


class DrawdownFilter:
    """
    Drawdown Circuit Breakers.
    
    Monitors daily, weekly, and monthly realized PnL against equity.
    If PnL drops below specified threshold percentages, rejects risk-increasing trades.
    """

    def __init__(
        self,
        max_daily_dd: float = 0.02,
        max_weekly_dd: float = 0.04,
        max_monthly_dd: float = 0.08
    ) -> None:
        """
        Args:
            max_daily_dd: Maximum allowed daily drawdown (e.g., 0.03 = 3%).
            max_weekly_dd: Maximum allowed weekly drawdown.
            max_monthly_dd: Maximum allowed monthly drawdown.
        """
        self.max_daily_dd = max_daily_dd
        self.max_weekly_dd = max_weekly_dd
        self.max_monthly_dd = max_monthly_dd
        
        logger.info(
            "DrawdownFilter initialized",
            daily=self.max_daily_dd,
            weekly=self.max_weekly_dd,
            monthly=self.max_monthly_dd
        )

    def check(self, signal: AlphaSignal, pair: str, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> bool:
        """
        Evaluate current drawdowns against limits.
        Drawdowns are represented as negative PnL.
        """
        equity = portfolio_state.current_equity
        if equity <= 0:
            return False
            
        daily_dd_pct = -portfolio_state.daily_pnl / equity
        weekly_dd_pct = -portfolio_state.weekly_pnl / equity
        monthly_dd_pct = -portfolio_state.monthly_pnl / equity
        
        if daily_dd_pct > self.max_daily_dd:
            logger.warning("Daily drawdown limit breached", dd=daily_dd_pct, limit=self.max_daily_dd)
            return False
            
        if weekly_dd_pct > self.max_weekly_dd:
            logger.warning("Weekly drawdown limit breached", dd=weekly_dd_pct, limit=self.max_weekly_dd)
            return False
            
        if monthly_dd_pct > self.max_monthly_dd:
            logger.warning("Monthly drawdown limit breached", dd=monthly_dd_pct, limit=self.max_monthly_dd)
            return False
            
        return True
