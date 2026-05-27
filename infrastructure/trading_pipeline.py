import time
import json
from typing import Any, Dict, Optional
import numpy as np
import structlog
from datetime import datetime

from risk.risk_engine import RiskEngine, PortfolioState
from execution.execution_engine import ExecutionEngine
from models.ensemble.aggregator import EnsembleAggregator
from models.ensemble.signal_generator import AlphaSignal

logger = structlog.get_logger()


class TradingPipeline:
    """
    The Master Orchestrator (Layer 1-5 Bridge).
    Connects Data -> Models -> Risk -> Execution into a unified event loop.
    """

    def __init__(
        self,
        ensemble: EnsembleAggregator,
        risk_engine: RiskEngine,
        execution_engine: ExecutionEngine,
        initial_capital: float = 100000.0
    ) -> None:
        self.ensemble = ensemble
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        
        # Internal state
        self.portfolio_state = PortfolioState(
            current_equity=initial_capital,
            open_positions={},
            daily_pnl=0.0,
            weekly_pnl=0.0,
            monthly_pnl=0.0,
            win_rate=0.5, # Assume neutral start
            win_loss_ratio=1.0,
            historical_returns=np.zeros(200) # Pre-fill for CVaR
        )
        
        # Rolling trade statistics for live Kelly updates
        self._trade_wins = []   # list of winning trade PnL amounts
        self._trade_losses = [] # list of losing trade PnL amounts (stored as positive)
        
        self.last_checkpoint_time = time.time()
        
        logger.info("TradingPipeline initialized", initial_capital=initial_capital)

    def process_tick(self, pair: str, features: np.ndarray, market_data: Dict[str, Any]) -> None:
        """
        Process a single tick or bar through the entire pipeline.
        
        Args:
            pair: "EURUSD"
            features: Real-time generated feature vector.
            market_data: Dictionary containing spread, volatility, mid_price.
        """
        logger.debug("Processing tick", pair=pair)
        
        # 1. Sync Portfolio State with broker before anything else to ensure fresh state
        actual_positions = self.execution_engine.sync_portfolio_state()
        self.portfolio_state.open_positions = actual_positions

        # 2. Ask Ensemble for AlphaSignal
        # We assume the features vector is properly formatted for the ensemble
        signal: AlphaSignal = self.ensemble.predict(features, return_signal=True)
        
        if signal.direction == 0:
            return  # No action required
            
        # 3. Gate the signal through the Risk Engine
        order = self.risk_engine.gate(signal, pair, self.portfolio_state, market_data)
        
        # 4. Execute if approved
        if order is not None:
            success = self.execution_engine.execute(order)
            if success:
                logger.info("Pipeline executed trade successfully", pair=pair, direction=order.direction)
            else:
                logger.warning("Execution engine failed to execute order", pair=pair)
                
        # 5. Periodic Checkpointing
        self._maybe_checkpoint()

    def update_pnl(self, realized_pnl: float, return_pct: float) -> None:
        """
        Update the portfolio metrics based on a closed trade.
        Also recalculates live win_rate and win_loss_ratio for Kelly sizing.
        """
        self.portfolio_state.current_equity += realized_pnl
        self.portfolio_state.daily_pnl += realized_pnl
        self.portfolio_state.weekly_pnl += realized_pnl
        self.portfolio_state.monthly_pnl += realized_pnl
        
        # Shift historical returns
        self.portfolio_state.historical_returns = np.roll(self.portfolio_state.historical_returns, -1)
        self.portfolio_state.historical_returns[-1] = return_pct
        
        # Track wins and losses for live Kelly computation
        if realized_pnl > 0:
            self._trade_wins.append(realized_pnl)
        elif realized_pnl < 0:
            self._trade_losses.append(abs(realized_pnl))
        
        # Update live statistics (require at least 5 trades before updating)
        total_trades = len(self._trade_wins) + len(self._trade_losses)
        if total_trades >= 5:
            self.portfolio_state.win_rate = len(self._trade_wins) / total_trades
            avg_win = np.mean(self._trade_wins) if self._trade_wins else 1.0
            avg_loss = np.mean(self._trade_losses) if self._trade_losses else 1.0
            self.portfolio_state.win_loss_ratio = avg_win / max(avg_loss, 1e-8)

    def _maybe_checkpoint(self) -> None:
        """Save state every 5 minutes (simulated or real time)."""
        now = time.time()
        if now - self.last_checkpoint_time > 300:
            self._save_state()
            self.last_checkpoint_time = now

    def _save_state(self) -> None:
        state_dict = {
            "equity": self.portfolio_state.current_equity,
            "daily_pnl": self.portfolio_state.daily_pnl,
            "timestamp": datetime.utcnow().isoformat()
        }
        # In a real system, write to Redis or state.json safely
        logger.debug("Checkpoint saved", state=state_dict)
