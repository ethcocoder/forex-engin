import numpy as np
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()


class MonteCarloSimulator:
    """
    Evaluates system robustness by resampling strategy returns (bootstrapping)
    over N iterations to calculate probability of ruin and drawdown expectations.
    """

    def __init__(self, iterations: int = 1000) -> None:
        self.iterations = iterations
        logger.info("MonteCarloSimulator initialized", iterations=iterations)

    def run_simulations(self, strategy_returns: List[float], initial_capital: float) -> Dict[str, Any]:
        """
        Runs random resampling with replacement of historical return series.
        """
        returns_arr = np.array(strategy_returns)
        if len(returns_arr) == 0:
            return {"status": "no_data"}

        final_equities = []
        max_drawdowns = []
        ruin_events = 0 # Equity hits < 50% of starting capital

        n_samples = len(returns_arr)

        for _ in range(self.iterations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns_arr, size=n_samples, replace=True)
            
            # Reconstruct equity curve
            equity_curve = initial_capital * np.cumprod(1 + sampled_returns)
            
            final_equities.append(equity_curve[-1])
            
            # Calculate drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (running_max - equity_curve) / running_max
            max_drawdowns.append(np.max(drawdowns))
            
            # Ruin check (50% drawdown rule)
            if np.min(equity_curve) < (initial_capital * 0.5):
                ruin_events += 1

        prob_of_ruin = (ruin_events / self.iterations) * 100.0
        median_final_equity = np.median(final_equities)
        avg_max_drawdown = np.mean(max_drawdowns) * 100.0
        
        logger.info(
            "Monte Carlo run completed",
            prob_of_ruin=prob_of_ruin,
            median_equity=median_final_equity
        )

        return {
            "probability_of_ruin_pct": float(prob_of_ruin),
            "median_final_equity": float(median_final_equity),
            "average_max_drawdown_pct": float(avg_max_drawdown),
            "percentiles": {
                "5th": float(np.percentile(final_equities, 5)),
                "25th": float(np.percentile(final_equities, 25)),
                "75th": float(np.percentile(final_equities, 75)),
                "95th": float(np.percentile(final_equities, 95))
            }
        }
