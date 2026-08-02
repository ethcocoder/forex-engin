import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable
import structlog
from datetime import timedelta

logger = structlog.get_logger()

class WalkForwardEngine:
    """
    Elite Walk-Forward Optimization Engine.
    Implements a rolling window approach to validate model stability over time.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        train_window_days: int,
        test_window_days: int,
        step_days: int,
        train_func: Callable,
        test_func: Callable
    ) -> None:
        self.data = data
        self.train_window = timedelta(days=train_window_days)
        self.test_window = timedelta(days=test_window_days)
        self.step = timedelta(days=step_days)
        self.train_func = train_func
        self.test_func = test_func
        
        self.results = []
        
        logger.info(
            "WalkForwardEngine initialized",
            train_days=train_window_days,
            test_days=test_window_days,
            step_days=step_days
        )

    def run(self) -> List[Dict[str, Any]]:
        """
        Executes the walk-forward loop.
        """
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        
        current_train_start = start_date
        
        while True:
            current_train_end = current_train_start + self.train_window
            current_test_start = current_train_end
            current_test_end = current_test_start + self.test_window
            
            if current_test_end > end_date:
                logger.info("Walk-forward completed: Reached end of data")
                break
                
            logger.info(
                "Executing Walk-Forward Window",
                train_range=f"{current_train_start} to {current_train_end}",
                test_range=f"{current_test_start} to {current_test_end}"
            )
            
            # Slice data
            train_data = self.data.loc[current_train_start:current_train_end]
            test_data = self.data.loc[current_test_start:current_test_end]
            
            # 1. Train
            model = self.train_func(train_data)
            
            # 2. Test (Backtest)
            performance = self.test_func(model, test_data)
            
            self.results.append({
                "window_start": current_test_start,
                "window_end": current_test_end,
                "performance": performance
            })
            
            # Slide the window
            current_train_start += self.step
            
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregates performance across all walk-forward windows.
        """
        if not self.results:
            return {}
            
        sharpes = [r["performance"].get("sharpe", 0) for r in self.results]
        returns = [r["performance"].get("total_return", 0) for r in self.results]
        
        summary = {
            "avg_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "avg_return": np.mean(returns),
            "num_windows": len(self.results),
            "stability_score": np.mean(sharpes) / (np.std(sharpes) + 1e-6)
        }
        
        logger.info("Walk-Forward Summary calculated", stability=summary["stability_score"])
        return summary
