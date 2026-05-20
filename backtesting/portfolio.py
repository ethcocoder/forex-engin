from typing import Dict, Any, List
import structlog
from datetime import datetime

logger = structlog.get_logger()


class BacktestPortfolio:
    """
    Simulates a trading account ledger for offline backtesting.
    Tracks equity curves, cash reserves, position changes, and fees.
    """

    def __init__(self, initial_capital: float, commission_per_lot: float = 1.50) -> None:
        self.initial_capital = initial_capital
        self.commission_per_lot = commission_per_lot
        
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}       # pair -> net size
        self.avg_entry: Dict[str, float] = {}       # pair -> average entry price
        
        # History
        self.equity_history: List[float] = [initial_capital]
        self.timestamp_history: List[datetime] = []
        self.realized_pnl = 0.0
        self.total_commissions = 0.0

        logger.info("BacktestPortfolio initialized", capital=initial_capital, commission=commission_per_lot)

    def apply_fill(self, fill_event: Dict[str, Any]) -> None:
        """
        Adjusts cash, positions, and average entries based on a fill event.
        fill_event: {
            "pair": "EURUSD",
            "direction": 1 or -1,
            "size": float,
            "fill_price": float,
            "timestamp": datetime
        }
        """
        pair = fill_event["pair"]
        direction = fill_event["direction"]
        size = fill_event["size"]
        fill_price = fill_event["fill_price"]
        
        # Commission calculation (Forex 1 lot = 100k units)
        lots = size / 100000.0
        commission = lots * self.commission_per_lot
        self.total_commissions += commission
        self.cash -= commission

        current_size = self.positions.get(pair, 0.0)
        
        if current_size == 0.0:
            # New position
            self.positions[pair] = size * direction
            self.avg_entry[pair] = fill_price
        else:
            new_size = current_size + (size * direction)
            
            # Check if adding or closing/reducing
            if (current_size > 0 and direction > 0) or (current_size < 0 and direction < 0):
                # Adding to position: recalculate weighted average entry price
                total_cost = (abs(current_size) * self.avg_entry[pair]) + (size * fill_price)
                self.avg_entry[pair] = total_cost / (abs(current_size) + size)
                self.positions[pair] = new_size
            else:
                # Reducing or closing position: realize P&L
                closed_size = min(abs(current_size), size)
                entry_price = self.avg_entry[pair]
                
                # PnL = closed_size * (exit - entry) * direction_of_entry
                pnl_direction = 1 if current_size > 0 else -1
                trade_pnl = closed_size * (fill_price - entry_price) * pnl_direction
                
                self.cash += trade_pnl
                self.realized_pnl += trade_pnl
                
                self.positions[pair] = new_size
                
                # Cleanup if completely closed
                if self.positions[pair] == 0.0:
                    del self.positions[pair]
                    del self.avg_entry[pair]

        logger.debug(
            "Portfolio state updated on fill",
            cash=self.cash,
            positions=self.positions,
            commission=commission
        )

    def update_equity(self, timestamp: datetime, latest_prices: Dict[str, float]) -> float:
        """
        Calculates Net Asset Value (NAV) taking into account unrealized PnL.
        """
        unrealized = 0.0
        for pair, size in self.positions.items():
            current_price = latest_prices.get(pair)
            if current_price is None:
                continue
            entry = self.avg_entry[pair]
            # Unrealized = size * (current - entry) * direction
            # Note: size is positive for long, negative for short
            unrealized += size * (current_price - entry)
            
        current_equity = self.cash + unrealized
        self.equity_history.append(current_equity)
        self.timestamp_history.append(timestamp)
        return current_equity
