import queue
import time
from typing import Dict, Any, List
import numpy as np
import structlog

from backtesting.engine import BaseBacktestEngine
from backtesting.data_handler import BaseDataHandler
from backtesting.portfolio import BacktestPortfolio
from execution.simulation.slippage_model import SlippageModel
from execution.simulation.market_impact import MarketImpactModel
from risk.risk_engine import RiskEngine, OrderRequest, PortfolioState

logger = structlog.get_logger()


class EventDrivenBacktestEngine(BaseBacktestEngine):
    """
    Highly detailed, queue-based historical simulator.
    Uses exact same risk gating and slippage penalty mechanisms as live pipeline.
    """

    def __init__(
        self,
        data_handler: BaseDataHandler,
        portfolio: BacktestPortfolio,
        risk_engine: RiskEngine,
        strategy: Any, # The trading model or ensemble aggregator
        config: Dict[str, Any] = None
    ) -> None:
        super().__init__(data_handler, portfolio, config or {})
        self.risk_engine = risk_engine
        self.strategy = strategy
        
        self.events_queue: queue.Queue = queue.Queue()
        
        # Simulators
        self.slippage_model = SlippageModel()
        self.impact_model = MarketImpactModel()
        
        # Cache for latest prices
        self.latest_prices: Dict[str, float] = {}
        
        logger.info("EventDrivenBacktestEngine initialized")

    def run(self) -> Dict[str, Any]:
        """
        Runs the event-driven queue loop:
        Data Generator -> Market Event -> Strategy -> Signal Event -> Risk -> Order -> Fill.
        """
        logger.info("Starting historical event-driven simulation")
        self.data_handler.load_data()
        
        bar_generator = self.data_handler.stream_bars()
        
        for bar in bar_generator:
            pair = bar["pair"]
            self.latest_prices[pair] = bar["close"]
            
            # 1. Market Event: Enqueue
            self.events_queue.put({"type": "MARKET", "bar": bar})
            
            # Process any cascading events triggered by this bar
            while not self.events_queue.empty():
                event = self.events_queue.get()
                
                if event["type"] == "MARKET":
                    self._handle_market(event["bar"])
                elif event["type"] == "SIGNAL":
                    self._handle_signal(event)
                elif event["type"] == "ORDER":
                    self._handle_order(event["order_request"])
                elif event["type"] == "FILL":
                    self._handle_fill(event["fill"])

        # Final portfolio evaluation at the end of data
        # Use last known prices to mark-to-market
        end_timestamp = list(bar_generator)[-1]["timestamp"] if hasattr(bar_generator, "timestamp") else bar["timestamp"]
        self.portfolio.update_equity(end_timestamp, self.latest_prices)
        
        logger.info("Backtest event loop finished")
        
        # Calculate performance
        from backtesting.performance import PerformanceCalculator
        perf_metrics = PerformanceCalculator.calculate_metrics(
            self.portfolio.equity_history,
            returns=None
        )
        trade_metrics = PerformanceCalculator.calculate_trade_metrics(self.trades)
        
        return {
            "performance": perf_metrics,
            "trades": trade_metrics,
            "raw_trades": self.trades,
            "final_equity": self.portfolio.equity_history[-1]
        }

    def _handle_market(self, bar: Dict[str, Any]) -> None:
        """Passes bars to strategy to compute AlphaSignals."""
        pair = bar["pair"]
        
        # We can extract or generate mock features to pass into the strategy
        # In a real setup, features are generated from the data_handler's history
        mock_features = np.zeros(10) 
        
        # Strategy prediction
        signal = self.strategy.predict(mock_features, return_signal=True)
        
        if signal.direction != 0:
            self.events_queue.put({
                "type": "SIGNAL",
                "pair": pair,
                "signal": signal,
                "bar": bar
            })

    def _handle_signal(self, event: Dict[str, Any]) -> None:
        """Passes signal through the RiskEngine circuit breakers."""
        pair = event["pair"]
        signal = event["signal"]
        bar = event["bar"]
        
        # Track starting equities for drawdown limits
        if not hasattr(self, "start_of_day_equity"):
            self.last_day = None
            self.last_week = None
            self.last_month = None
            self.start_of_day_equity = None
            self.start_of_week_equity = None
            self.start_of_month_equity = None
            
        current_time = bar["timestamp"]
        current_equity = self.portfolio.cash
        
        # Check day boundary
        if self.start_of_day_equity is None or current_time.date() != self.last_day:
            self.start_of_day_equity = current_equity
            self.last_day = current_time.date()
            
        # Check week boundary
        current_week = current_time.isocalendar()[1]
        if self.start_of_week_equity is None or current_week != self.last_week:
            self.start_of_week_equity = current_equity
            self.last_week = current_week
            
        # Check month boundary
        if self.start_of_month_equity is None or current_time.month != self.last_month:
            self.start_of_month_equity = current_equity
            self.last_month = current_time.month
            
        daily_pnl = current_equity - self.start_of_day_equity
        weekly_pnl = current_equity - self.start_of_week_equity
        monthly_pnl = current_equity - self.start_of_month_equity
        
        # Calculate dynamic win_rate and win_loss_ratio from self.trades
        win_rate = 0.5
        win_loss_ratio = 1.0
        
        if len(self.trades) > 0:
            wins = [t for t in self.trades if t["pnl"] > 0]
            losses = [t for t in self.trades if t["pnl"] < 0]
            
            win_rate = len(wins) / len(self.trades)
            
            avg_win = np.mean([t["pnl"] for t in wins]) if len(wins) > 0 else 0.0
            avg_loss = abs(np.mean([t["pnl"] for t in losses])) if len(losses) > 0 else 0.0
            
            if avg_loss > 0:
                win_loss_ratio = avg_win / avg_loss
            elif avg_win > 0:
                win_loss_ratio = 10.0
                
        historical_returns = np.zeros(200)
        if len(self.trades) > 0:
            recent_trades = self.trades[-200:]
            returns_list = [t["pnl"] / self.portfolio.initial_capital for t in recent_trades]
            historical_returns[-len(returns_list):] = returns_list

        p_state = PortfolioState(
            current_equity=current_equity,
            open_positions=self.portfolio.positions.copy(),
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            historical_returns=historical_returns
        )
        
        market_data = {
            "mid_price": bar["close"],
            "spread_pips": 1.5, # default spread
            "adv": 1000000.0,
            "pip_value": 0.0001,
            "volatility": 0.001
        }
        
        order = self.risk_engine.gate(signal, pair, p_state, market_data)
        
        if order:
            self.events_queue.put({
                "type": "ORDER",
                "order_request": order
            })

    def _handle_order(self, order: OrderRequest) -> None:
        """Simulates physical execution using the Slippage and Impact Models."""
        pair = order.pair
        direction = order.direction
        size = order.size
        
        # Execute trade on historical tick
        mid_price = self.latest_prices[pair]
        
        # Calculate dynamic slippage
        impact = self.impact_model.calculate_impact(size, 1000000.0) # ADV = 1M
        slippage = self.slippage_model.calculate_slippage(base_spread_pips=1.5, market_impact_pips=impact)
        
        # Convert slippage pips to price delta
        pip_val = 0.0001
        slippage_price = slippage * pip_val
        
        fill_price = mid_price + slippage_price if direction == 1 else mid_price - slippage_price
        
        fill = {
            "pair": pair,
            "direction": direction,
            "size": size,
            "fill_price": fill_price,
            "slippage_pips": slippage,
            "timestamp": time.time()
        }
        
        self.events_queue.put({
            "type": "FILL",
            "fill": fill
        })

    def _handle_fill(self, fill: Dict[str, Any]) -> None:
        """Updates internal portfolio ledger with the trade results."""
        # Check if we are closing a position to capture the PnL
        pair = fill["pair"]
        size = fill["size"]
        direction = fill["direction"]
        
        current_pos = self.portfolio.positions.get(pair, 0.0)
        
        # If we have a position and this fill reduces/closes it, calculate realized trade PnL
        if current_pos != 0.0:
            if (current_pos > 0 and direction < 0) or (current_pos < 0 and direction > 0):
                closed_size = min(abs(current_pos), size)
                entry_price = self.portfolio.avg_entry.get(pair, fill["fill_price"])
                pnl_direction = 1 if current_pos > 0 else -1
                trade_pnl = closed_size * (fill["fill_price"] - entry_price) * pnl_direction
                
                # Log closed trade for stats
                self.trades.append({
                    "pair": pair,
                    "direction": direction,
                    "size": closed_size,
                    "pnl": trade_pnl,
                    "slippage_pips": fill["slippage_pips"]
                })
        
        self.portfolio.apply_fill(fill)
        self.portfolio.update_equity(fill["timestamp"], self.latest_prices)
