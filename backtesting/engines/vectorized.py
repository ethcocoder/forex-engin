from typing import Dict, Any, List
import pandas as pd
import numpy as np
import structlog

from backtesting.engine import BaseBacktestEngine
from backtesting.data_handler import BaseDataHandler
from backtesting.portfolio import BacktestPortfolio

logger = structlog.get_logger()


class VectorizedBacktestEngine(BaseBacktestEngine):
    """
    Extremely fast vectorized backtesting engine.
    Applies vector calculations directly on a DataFrame.
    Used for hyperparameter sweeps and initial strategy exploration.
    """

    def __init__(
        self,
        data_handler: BaseDataHandler,
        portfolio: BacktestPortfolio,
        config: Dict[str, Any] = None
    ) -> None:
        super().__init__(data_handler, portfolio, config or {})
        logger.info("VectorizedBacktestEngine initialized")

    def run(self) -> Dict[str, Any]:
        """
        Executes vectorized strategy math over the historical dataset.
        Expects strategy signals to be pre-computed as a column or array.
        """
        logger.info("Running vectorized backtest")
        
        # Pull the primary pair from config
        pair = self.config.get("primary_pair", "EURUSD")
        
        # Extract underlying dataframe
        # Vectorized requires loading the entire dataset into memory at once
        self.data_handler.load_data()
        df = self.data_handler.data.get(pair)
        
        if df is None or df.empty:
            logger.error("No data available for vectorized run", pair=pair)
            return {"status": "no_data"}
            
        # Calculate daily percentage returns
        df["market_returns"] = df["close"].pct_change()
        
        # Simple Mock Signal: Long if close > 20-day SMA, Short if below
        # In a real environment, strategy signals are passed in or generated here.
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["signal"] = np.where(df["close"] > df["sma_20"], 1, -1)
        
        # Shift signal by 1 bar to avoid lookahead bias (execute at next bar open)
        df["strategy_returns"] = df["signal"].shift(1) * df["market_returns"]
        
        # Calculate cumulative returns and equity curve
        df["cum_returns"] = (1 + df["strategy_returns"].fillna(0)).cumprod()
        equity_curve = self.portfolio.initial_capital * df["cum_returns"]
        
        # Track realized trades (transition points in signal)
        df["trade_trigger"] = df["signal"].diff().fillna(0)
        trades_triggered = df[df["trade_trigger"] != 0]
        
        # Generate raw trades list for metrics
        raw_trades = []
        for ts, row in trades_triggered.iterrows():
            raw_trades.append({
                "pair": pair,
                "direction": int(row["signal"]),
                "size": 10000.0, # default size
                "pnl": float(row["strategy_returns"] * self.portfolio.initial_capital),
                "timestamp": ts
            })

        self.portfolio.equity_history = equity_curve.tolist()
        
        from backtesting.performance import PerformanceCalculator
        perf_metrics = PerformanceCalculator.calculate_metrics(
            self.portfolio.equity_history,
            returns=df["strategy_returns"].dropna().tolist()
        )
        
        trade_metrics = PerformanceCalculator.calculate_trade_metrics(raw_trades)

        logger.info("Vectorized backtest complete", final_equity=self.portfolio.equity_history[-1])

        return {
            "performance": perf_metrics,
            "trades": trade_metrics,
            "final_equity": self.portfolio.equity_history[-1]
        }
