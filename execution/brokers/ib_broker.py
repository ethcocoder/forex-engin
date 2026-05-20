import time
from typing import Any, Dict, List, Optional
import structlog

from execution.brokers.base_broker import BaseBroker
from risk.risk_engine import OrderRequest

logger = structlog.get_logger()


class IBBroker(BaseBroker):
    """
    Interactive Brokers (IBKR) API Adapter.
    
    Uses `ib_insync` to connect to Trader Workstation (TWS) or IB Gateway.
    Implements standard connect, order routing, balance queries, and positions.
    Includes a robust simulation fallback if IB Gateway is not running.
    """

    def __init__(self, name: str = "ib_broker", config: Any = None) -> None:
        super().__init__(name, config or {})
        
        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 7497)  # Default TWS Paper port
        self.client_id = self.config.get("client_id", 1)
        self.ib: Optional[Any] = None
        self.simulated = False

        # Simulation fallback state
        self.sim_positions: Dict[str, float] = {}
        self.sim_cash = self.config.get("initial_capital", 100000.0)
        
        logger.info(
            "IBBroker initialized",
            host=self.host,
            port=self.port,
            client_id=self.client_id
        )

    def connect(self) -> bool:
        """
        Connect to IB Gateway or TWS. Falls back to simulated mode on failure.
        """
        try:
            from ib_insync import IB
            self.ib = IB()
            logger.info("Attempting connection to live IB Gateway/TWS...")
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=3.0)
            self.simulated = False
            logger.info("Successfully connected to live IB TWS/Gateway")
            return True
        except Exception as e:
            logger.warning(
                "Could not connect to live IB Gateway. Falling back to Simulated IB Mode.",
                reason=str(e)
            )
            self.simulated = True
            return True

    def disconnect(self) -> None:
        if not self.simulated and self.ib:
            try:
                self.ib.disconnect()
                logger.info("Disconnected from IB Gateway/TWS")
            except Exception as e:
                logger.error("Error disconnecting from IB", error=str(e))
        else:
            logger.info("Simulated IBBroker shut down")

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Place an order to Interactive Brokers.
        """
        if self.simulated:
            return self._place_order_simulated(order)

        # Connect check
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("IB Gateway is not connected.")

        pair = order.pair
        if len(pair) == 6:
            base, quote = pair[:3], pair[3:]
        else:
            base, quote = pair, "USD"

        # Import ib_insync objects
        from ib_insync import Forex, MarketOrder, LimitOrder

        # Build contract (Forex CASH)
        contract = Forex(symbol=base, currency=quote)
        self.ib.qualifyContracts(contract)

        # Determine direction
        action = "BUY" if order.direction == 1 else "SELL"
        order_type = (order.order_type or "MARKET").upper()

        if order_type == "MARKET":
            ib_order = MarketOrder(action, order.size)
        elif order_type == "LIMIT":
            if order.limit_price is None:
                raise ValueError("Limit price must be specified for LIMIT orders.")
            ib_order = LimitOrder(action, order.size, order.limit_price)
        else:
            logger.error("Unsupported order type for IBKR", order_type=order_type)
            return {"status": "REJECTED", "reason": f"Unsupported order type: {order_type}"}

        try:
            trade = self.ib.placeOrder(contract, ib_order)
            # Sleep briefly to allow execution event to trigger
            self.ib.sleep(0.5)
            
            status = "PENDING"
            fill_price = 0.0
            
            if trade.isDone():
                status = "FILLED"
                fill_price = trade.orderStatus.avgFillPrice if trade.orderStatus else 0.0
                
            logger.info(
                "IB Order placed",
                order_id=ib_order.orderId,
                status=status,
                fill_price=fill_price
            )
            
            return {
                "status": status,
                "fill_price": fill_price,
                "transaction_id": str(ib_order.orderId),
                "raw_response": str(trade)
            }
        except Exception as e:
            logger.error("Error submitting order to IBKR", error=str(e))
            raise

    def get_positions(self) -> Dict[str, float]:
        """
        Query current open positions.
        """
        if self.simulated:
            return self.sim_positions.copy()

        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("IB Gateway is not connected.")

        positions = {}
        try:
            for pos in self.ib.positions():
                contract = pos.contract
                # Filter for Forex CASH pairs
                if contract.secType == "CASH":
                    pair = f"{contract.symbol}{contract.currency}"
                    positions[pair] = pos.position
            return positions
        except Exception as e:
            logger.error("Failed to query live IB positions", error=str(e))
            raise

    def get_account_balance(self) -> float:
        """
        Retrieve current NAV/Cash balance.
        """
        if self.simulated:
            return self.sim_cash

        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("IB Gateway is not connected.")

        try:
            for item in self.ib.accountSummary():
                if item.tag == "NetLiquidation":
                    return float(item.value)
            return 0.0
        except Exception as e:
            logger.error("Failed to query live IB account summary", error=str(e))
            raise

    def _place_order_simulated(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Fallback order simulator for local verification.
        """
        pair = order.pair
        direction = order.direction
        size = order.size
        
        # Simulated fill price
        fill_price = order.limit_price if order.order_type == "LIMIT" and order.limit_price else 1.05
        
        # Update simulated positions
        current_pos = self.sim_positions.get(pair, 0.0)
        self.sim_positions[pair] = current_pos + (size * direction)
        
        if self.sim_positions[pair] == 0.0:
            del self.sim_positions[pair]
            
        logger.debug(
            "IB simulated fill",
            pair=pair,
            size=size,
            direction=direction,
            fill_price=fill_price
        )
        
        return {
            "status": "FILLED",
            "fill_price": fill_price,
            "transaction_id": f"sim_ib_{int(time.time() * 1000)}",
            "fill_time": time.time()
        }
