from typing import Any, Dict, List
import time
import structlog

from execution.brokers.base_broker import BaseBroker
from execution.simulation.slippage_model import SlippageModel
from execution.simulation.market_impact import MarketImpactModel
from risk.risk_engine import OrderRequest

logger = structlog.get_logger()


class PaperBroker(BaseBroker):
    """
    Internal simulator acting as a broker.
    Tracks cash and positions locally, applying realistic slippage.
    """

    def __init__(self, name: str = "paper_broker", config: Any = None) -> None:
        super().__init__(name, config or {})
        
        self.initial_capital = self.config.get("initial_capital", 100000.0)
        self.cash = self.initial_capital
        self.positions: Dict[str, float] = {}  # pair -> size (positive long, negative short)
        self.entry_prices: Dict[str, float] = {}
        
        self.slippage_model = SlippageModel()
        self.impact_model = MarketImpactModel()
        
        # We need a reference to market state to simulate fill prices
        self.latest_market_data: Dict[str, Any] = {}
        
        logger.info("PaperBroker initialized", initial_capital=self.initial_capital)

    def connect(self) -> bool:
        logger.info("PaperBroker connected (internal simulation)")
        return True

    def disconnect(self) -> None:
        logger.info("PaperBroker disconnected")

    def update_market_state(self, market_data: Dict[str, Any]) -> None:
        """Feed latest ticks into the simulator."""
        self.latest_market_data.update(market_data)

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Simulate an order fill immediately based on latest market data.
        """
        pair = order.pair
        direction = order.direction
        size = order.size
        
        # Get market data
        current_price = self.latest_market_data.get(pair, {}).get("mid_price", 1.0)
        spread_pips = self.latest_market_data.get(pair, {}).get("spread_pips", 2.0)
        adv = self.latest_market_data.get(pair, {}).get("adv", 1000000.0)
        pip_value = self.latest_market_data.get(pair, {}).get("pip_value", 0.0001)
        
        # Calculate Slippage
        impact_pips = self.impact_model.calculate_impact(size, adv)
        total_slippage_pips = self.slippage_model.calculate_slippage(spread_pips, impact_pips)
        
        # Determine fill price
        slippage_price_delta = total_slippage_pips * pip_value
        
        if direction == 1:
            # Buying (Long) - pay the ask + slippage
            fill_price = current_price + slippage_price_delta
        else:
            # Selling (Short) - receive the bid - slippage
            fill_price = current_price - slippage_price_delta
            
        # Update internal ledger
        current_pos = self.positions.get(pair, 0.0)
        new_pos = current_pos + (size * direction)
        
        # Simple PnL update (ignoring margin lockup for simplicity in paper broker)
        # If we are closing or reversing, calculate realized PnL
        if current_pos != 0.0:
            if (current_pos > 0 and direction < 0) or (current_pos < 0 and direction > 0):
                # We are closing some or all of the position
                close_size = min(abs(current_pos), size)
                entry = self.entry_prices.get(pair, current_price)
                
                # Realized PnL = size * (exit - entry) * direction
                pnl = close_size * (fill_price - entry) * (1 if current_pos > 0 else -1)
                self.cash += pnl
                
        self.positions[pair] = new_pos
        self.entry_prices[pair] = fill_price
        
        if self.positions[pair] == 0.0:
            del self.positions[pair]
            del self.entry_prices[pair]
            
        logger.info(
            "PaperBroker filled order",
            pair=pair,
            direction=direction,
            size=size,
            fill_price=fill_price,
            slippage_pips=total_slippage_pips
        )
        
        return {
            "status": "FILLED",
            "fill_price": fill_price,
            "fill_time": time.time(),
            "slippage_pips": total_slippage_pips
        }

    def get_positions(self) -> Dict[str, float]:
        return self.positions.copy()

    def get_account_balance(self) -> float:
        # For simplicity, returning cash. Actual NAV requires mark-to-market.
        return self.cash
