from typing import Any, Dict, Optional
import numpy as np
import structlog

from risk.risk_engine import PortfolioState
from risk.monitoring.alert_manager import AlertManager, AlertLevel

logger = structlog.get_logger()


class PortfolioMonitor:
    """
    Monitors portfolio risk metrics dynamically.
    Computes Drawdown, Leverage, Asset Concentration, and Parametric Value at Risk (VaR).
    Dispatches warning and critical alerts to the AlertManager when limits are breached.
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.alert_manager = alert_manager
        self.config = config or {}
        
        # Risk threshold config
        self.max_drawdown = self.config.get("max_drawdown", 0.15)  # 15% Max Drawdown
        self.max_leverage = self.config.get("max_leverage", 25.0)   # 25x Max Leverage
        self.max_concentration = self.config.get("max_concentration", 0.35)  # 35% max in single pair
        self.var_confidence = self.config.get("var_confidence", 0.95)  # 95% VaR
        
        # State tracking
        self.peak_equity = 0.0
        
        logger.info(
            "PortfolioMonitor initialized",
            max_drawdown=self.max_drawdown,
            max_leverage=self.max_leverage,
            max_concentration=self.max_concentration
        )

    def update(self, state: PortfolioState) -> Dict[str, Any]:
        """
        Processes a PortfolioState update. Calculates all risk metrics and runs boundary checks.
        
        Returns:
            A dictionary containing the computed metrics.
        """
        # 1. Track peak equity & drawdown
        if self.peak_equity <= 0.0 or state.current_equity > self.peak_equity:
            self.peak_equity = state.current_equity
            
        current_drawdown = 0.0
        if self.peak_equity > 0.0:
            current_drawdown = (self.peak_equity - state.current_equity) / self.peak_equity

        if current_drawdown > self.max_drawdown:
            self.alert_manager.trigger(
                level=AlertLevel.CRITICAL,
                metric_name="drawdown",
                value=current_drawdown,
                message=f"Drawdown {current_drawdown * 100:.2f}% breached limit of {self.max_drawdown * 100:.2f}%"
            )

        # 2. Leverage checks
        total_exposure = sum(abs(pos) for pos in state.open_positions.values())
        current_leverage = 0.0
        if state.current_equity > 0.0:
            current_leverage = total_exposure / state.current_equity

        if current_leverage > self.max_leverage:
            self.alert_manager.trigger(
                level=AlertLevel.CRITICAL,
                metric_name="leverage",
                value=current_leverage,
                message=f"Leverage {current_leverage:.2f}x breached limit of {self.max_leverage:.2f}x"
            )

        # 3. Concentration checks
        max_seen_concentration = 0.0
        concentrated_pair = ""
        for pair, size in state.open_positions.items():
            if state.current_equity > 0.0:
                concentration = abs(size) / state.current_equity
                if concentration > max_seen_concentration:
                    max_seen_concentration = concentration
                    concentrated_pair = pair

        if max_seen_concentration > self.max_concentration:
            self.alert_manager.trigger(
                level=AlertLevel.WARNING,
                metric_name="concentration",
                value=max_seen_concentration,
                message=f"Asset concentration in {concentrated_pair} reached {max_seen_concentration * 100:.2f}%, limit {self.max_concentration * 100:.2f}%"
            )

        # 4. Parametric Value at Risk (VaR)
        var_value = 0.0
        if len(state.historical_returns) > 10:
            # We assume historical_returns contains recent returns (e.g. daily or rolling intervals)
            mean_ret = np.mean(state.historical_returns)
            std_ret = np.std(state.historical_returns)
            
            # Map confidence level to Z-score
            z_score = 1.645 if self.var_confidence == 0.95 else 2.33
            
            # Parametric VaR fraction
            var_pct = -(mean_ret - z_score * std_ret)
            
            # Ensure it is positive representing a risk of loss
            var_value = max(0.0, var_pct * state.current_equity)
            
            # Warn if VaR is more than 5% of the total equity
            if var_value > state.current_equity * 0.05:
                self.alert_manager.trigger(
                    level=AlertLevel.WARNING,
                    metric_name="value_at_risk",
                    value=var_value,
                    message=f"VaR risk ({self.var_confidence * 100:.0f}%) reached {var_value:.2f} ({var_pct * 100:.2f}% of equity)"
                )

        metrics = {
            "drawdown": current_drawdown,
            "leverage": current_leverage,
            "max_concentration": max_seen_concentration,
            "concentrated_pair": concentrated_pair,
            "value_at_risk": var_value,
            "equity": state.current_equity,
            "peak_equity": self.peak_equity
        }
        
        logger.debug("Portfolio monitor metrics calculated", **metrics)
        return metrics
