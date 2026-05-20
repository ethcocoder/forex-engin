from typing import Dict, List, Any
import numpy as np
import structlog
from datetime import datetime

logger = structlog.get_logger()


class PerformanceTracker:
    """
    Generates an end-of-run statistical tear sheet.
    Calculates Sharpe Ratio, Max Drawdown, Win Rate, and Execution Slippage.
    """

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[float] = [datetime.utcnow().timestamp()]
        
        self.trades: List[Dict[str, Any]] = []
        self.total_slippage_pips: float = 0.0
        
        logger.info("PerformanceTracker initialized", initial_capital=initial_capital)

    def update_equity(self, current_equity: float, timestamp: float) -> None:
        """Log a new equity high/low."""
        self.equity_curve.append(current_equity)
        self.timestamps.append(timestamp)

    def log_trade(self, pair: str, direction: int, size: float, pnl: float, slippage_pips: float) -> None:
        """Record a closed trade."""
        self.trades.append({
            "pair": pair,
            "direction": direction,
            "size": size,
            "pnl": pnl,
            "slippage": slippage_pips
        })
        self.total_slippage_pips += slippage_pips

    def generate_tear_sheet(self) -> str:
        """Computes metrics and returns a formatted markdown summary."""
        if len(self.equity_curve) < 2:
            return "Insufficient data for tear sheet."

        # Arrays
        equity_arr = np.array(self.equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        
        # PnL Metrics
        total_return_pct = (equity_arr[-1] / self.initial_capital - 1.0) * 100.0
        
        # Max Drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (running_max - equity_arr) / running_max
        max_dd_pct = np.max(drawdown) * 100.0
        
        # Sharpe Ratio (Assuming zero risk-free rate for intraday, annualized approximation)
        # Note: If returns are per tick/minute, scaling factor is high. We'll present raw & roughly annualized.
        mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
        std_ret = np.std(returns) if len(returns) > 0 else 1e-6
        raw_sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
        
        # Trade Metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t["pnl"] > 0]
        losing_trades = [t for t in self.trades if t["pnl"] <= 0]
        
        win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
        
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = abs(np.mean([t["pnl"] for t in losing_trades])) if losing_trades else 1e-6
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        sheet = f"""
# Performance Tear Sheet

## 1. Portfolio Level
* **Initial Capital:** ${self.initial_capital:,.2f}
* **Ending Capital:** ${equity_arr[-1]:,.2f}
* **Total Return:** {total_return_pct:.2f}%
* **Max Drawdown:** -{max_dd_pct:.2f}%
* **Raw Sharpe Ratio:** {raw_sharpe:.4f}

## 2. Trade Execution Level
* **Total Trades Executed:** {total_trades}
* **Win Rate:** {win_rate:.1f}%
* **Win/Loss Ratio:** {win_loss_ratio:.2f}
* **Total Slippage Drag:** {self.total_slippage_pips:.2f} pips
"""
        return sheet
