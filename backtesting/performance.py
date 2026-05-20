import numpy as np
import pandas as pd
from typing import Dict, Any, List


class PerformanceCalculator:
    """
    Computes professional portfolio metrics from backtest runs.
    """

    @staticmethod
    def calculate_metrics(equity_curve: List[float], returns: List[float] = None) -> Dict[str, Any]:
        """
        Computes Sharpe, Sortino, Max Drawdown, Max Drawdown Duration, and total return.
        """
        equity = np.array(equity_curve)
        if len(equity) < 2:
            return {"status": "insufficient_data"}

        if returns is None:
            returns = np.diff(equity) / equity[:-1]
        else:
            returns = np.array(returns)

        total_return = (equity[-1] / equity[0] - 1.0) * 100.0

        # Drawdown calculations
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_dd = np.max(drawdowns) * 100.0

        # Calculate max drawdown duration
        # We find chunks where drawdown > 0 and compute max consecutive lengths
        dd_binary = drawdowns > 0
        max_dd_duration = 0
        current_duration = 0
        for val in dd_binary:
            if val:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        # Risk-adjusted metrics (assuming daily returns for scaling, standard 252 trading days)
        mean_return = np.mean(returns) if len(returns) > 0 else 0.0
        std_return = np.std(returns) if len(returns) > 0 else 1e-6
        
        # Annualized values (approximation assuming intraday samples represent standard spacing)
        raw_sharpe = mean_return / std_return if std_return > 0 else 0.0
        annualized_sharpe = raw_sharpe * np.sqrt(252) # Scaled to days

        # Downside deviation for Sortino
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        raw_sortino = mean_return / downside_std if downside_std > 0 else 0.0
        annualized_sortino = raw_sortino * np.sqrt(252)

        return {
            "total_return_pct": float(total_return),
            "max_drawdown_pct": float(max_dd),
            "max_drawdown_duration_bars": int(max_dd_duration),
            "raw_sharpe": float(raw_sharpe),
            "annualized_sharpe": float(annualized_sharpe),
            "raw_sortino": float(raw_sortino),
            "annualized_sortino": float(annualized_sortino)
        }

    @staticmethod
    def calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates trade-specific metrics: Win Rate, Profit Factor, Average Win/Loss.
        """
        if not trades:
            return {"status": "no_trades"}

        pnls = [t["pnl"] for t in trades]
        total_trades = len(pnls)
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
        
        gross_profits = sum(wins)
        gross_losses = abs(sum(losses))
        
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        return {
            "total_trades": total_trades,
            "win_rate_pct": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "gross_profits": float(gross_profits),
            "gross_losses": float(gross_losses)
        }
