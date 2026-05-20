import time
from typing import Any, Dict, Optional
import structlog

from risk.risk_engine import OrderRequest
from execution.simulation.slippage_model import SlippageModel
from execution.simulation.market_impact import MarketImpactModel

logger = structlog.get_logger()


class OrderFillSimulator:
    """
    Order Fill Simulator (Layer 5 Simulation).
    
    Provides modular, realistic matching logic for MARKET, LIMIT, and STOP orders
    against tick, bar, or orderbook state. Integrates directly with SlippageModel
    and MarketImpactModel to compute precise fill execution details.
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        impact_model: Optional[MarketImpactModel] = None
    ) -> None:
        self.slippage_model = slippage_model or SlippageModel()
        self.impact_model = impact_model or MarketImpactModel()
        logger.info("OrderFillSimulator initialized")

    def simulate_fill(self, order: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an OrderRequest against current market data to determine execution status.
        
        Args:
            order: The requested order details.
            market_data: Dictionary containing current market details:
                - mid_price: Current mid price (float, required)
                - high: Current interval high price (float, optional)
                - low: Current interval low price (float, optional)
                - spread_pips: Level 1 bid/ask spread (float, default 2.0)
                - adv: Average Daily Volume (float, default 1,000,000)
                - pip_value: Asset pip scale (float, default 0.0001)
                - volatility: Current volatility index (float, default 1.0)
                
        Returns:
            Dict containing fill results:
                - status: 'FILLED', 'REJECTED', 'UNFILLED'
                - fill_price: Price at which order executed (float)
                - fill_time: Execution timestamp (float)
                - slippage_pips: Total slippage incurred (float)
        """
        order_type = (order.order_type or "MARKET").upper()
        
        if order_type == "MARKET":
            return self._fill_market(order, market_data)
        elif order_type == "LIMIT":
            return self._fill_limit(order, market_data)
        elif order_type == "STOP":
            return self._fill_stop(order, market_data)
        else:
            logger.error("Unsupported order type", order_type=order_type)
            return {
                "status": "REJECTED",
                "reason": f"Unsupported order type: {order_type}",
                "fill_price": 0.0,
                "fill_time": time.time(),
                "slippage_pips": 0.0
            }

    def _fill_market(self, order: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Immediately fills market orders, applying spread, impact, and slippage noise.
        """
        mid_price = market_data.get("mid_price", 1.0)
        spread_pips = market_data.get("spread_pips", 2.0)
        adv = market_data.get("adv", 1000000.0)
        pip_value = market_data.get("pip_value", 0.0001)
        volatility = market_data.get("volatility", 1.0)

        # Calculate impact and slippage
        impact_pips = self.impact_model.calculate_impact(order.size, adv, volatility)
        slippage_pips = self.slippage_model.calculate_slippage(spread_pips, impact_pips, volatility)
        slippage_price_delta = slippage_pips * pip_value

        # Execution direction logic (Buy at Ask, Sell at Bid)
        if order.direction == 1:
            fill_price = mid_price + slippage_price_delta
        else:
            fill_price = mid_price - slippage_price_delta

        logger.debug(
            "Market order fill simulated",
            pair=order.pair,
            direction=order.direction,
            mid_price=mid_price,
            fill_price=fill_price,
            slippage_pips=slippage_pips
        )

        return {
            "status": "FILLED",
            "fill_price": fill_price,
            "fill_time": time.time(),
            "slippage_pips": slippage_pips
        }

    def _fill_limit(self, order: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if limit price was reached/crossed during the price interval.
        """
        limit_price = order.limit_price
        if limit_price is None:
            logger.error("Limit price is missing for LIMIT order")
            return {
                "status": "REJECTED",
                "reason": "Missing limit_price",
                "fill_price": 0.0,
                "fill_time": time.time(),
                "slippage_pips": 0.0
            }

        high_price = market_data.get("high", market_data.get("mid_price", 1.0))
        low_price = market_data.get("low", market_data.get("mid_price", 1.0))

        is_filled = False
        if order.direction == 1:  # Buy Limit: fill if low <= limit_price
            if low_price <= limit_price:
                is_filled = True
        else:  # Sell Limit: fill if high >= limit_price
            if high_price >= limit_price:
                is_filled = True

        if is_filled:
            logger.debug("Limit order fill simulated", pair=order.pair, limit_price=limit_price)
            # Limit orders generally have 0 slippage or get filled at limit price
            return {
                "status": "FILLED",
                "fill_price": limit_price,
                "fill_time": time.time(),
                "slippage_pips": 0.0
            }

        return {
            "status": "UNFILLED",
            "fill_price": 0.0,
            "fill_time": time.time(),
            "slippage_pips": 0.0
        }

    def _fill_stop(self, order: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if stop trigger level was reached/crossed, then executes as a market order.
        """
        stop_price = order.limit_price  # Stop triggers are mapped to limit_price field in OrderRequest
        if stop_price is None:
            logger.error("Stop price is missing for STOP order")
            return {
                "status": "REJECTED",
                "reason": "Missing stop price trigger",
                "fill_price": 0.0,
                "fill_time": time.time(),
                "slippage_pips": 0.0
            }

        high_price = market_data.get("high", market_data.get("mid_price", 1.0))
        low_price = market_data.get("low", market_data.get("mid_price", 1.0))

        is_triggered = False
        if order.direction == 1:  # Buy Stop: trigger if high >= stop_price
            if high_price >= stop_price:
                is_triggered = True
        else:  # Sell Stop: trigger if low <= stop_price
            if low_price <= stop_price:
                is_triggered = True

        if is_triggered:
            # Triggered! Now fill as a market order with slippage penalty
            logger.debug("Stop order triggered, converting to market execution", pair=order.pair, stop_price=stop_price)
            return self._fill_market(order, market_data)

        return {
            "status": "UNFILLED",
            "fill_price": 0.0,
            "fill_time": time.time(),
            "slippage_pips": 0.0
        }
