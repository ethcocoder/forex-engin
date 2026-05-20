import time
import requests
from typing import Any, Dict, List
import structlog

from execution.brokers.base_broker import BaseBroker
from risk.risk_engine import OrderRequest

logger = structlog.get_logger()


class OandaBroker(BaseBroker):
    """
    OANDA v20 API Adapter using pure Python requests.
    Optimized for low latency and zero external library bloat.
    """

    def __init__(self, name: str = "oanda_live", config: Any = None) -> None:
        super().__init__(name, config or {})
        
        # Extract auth from config
        self.access_token = self.config.get("access_token", "")
        self.account_id = self.config.get("account_id", "")
        
        # Use practice domain by default for safety
        self.domain = self.config.get("domain", "api-fxpractice.oanda.com")
        self.base_url = f"https://{self.domain}/v3/accounts/{self.account_id}"
        
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "UNIX"
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info("OandaBroker initialized", domain=self.domain, account_id=self.account_id)

    def connect(self) -> bool:
        """
        Verify connection by hitting the account endpoint.
        """
        try:
            response = self.session.get(f"{self.base_url}/summary", timeout=5.0)
            if response.status_code == 200:
                logger.info("Successfully connected to OANDA API")
                return True
            else:
                logger.error("Failed to connect to OANDA", status_code=response.status_code, response=response.text)
                return False
        except Exception as e:
            logger.error("Network error connecting to OANDA", error=str(e))
            return False

    def disconnect(self) -> None:
        self.session.close()
        logger.info("OandaBroker session closed")

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Submits a Market Order to the OANDA v20 API.
        """
        # Format the instrument (e.g., EURUSD -> EUR_USD)
        instrument = f"{order.pair[:3]}_{order.pair[3:]}" if len(order.pair) == 6 else order.pair
        
        # OANDA represents direction implicitly in the units sign
        units = int(order.size * order.direction)
        
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }
        
        try:
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/orders", json=payload, timeout=2.0)
            latency = (time.time() - start_time) * 1000
            
            data = response.json()
            
            if response.status_code in [200, 201]:
                # OANDA returns an orderCreateTransaction and optionally an orderFillTransaction
                fill_txn = data.get("orderFillTransaction", {})
                
                if fill_txn:
                    fill_price = float(fill_txn.get("price", 0.0))
                    logger.info(
                        "OANDA order filled",
                        instrument=instrument,
                        units=units,
                        fill_price=fill_price,
                        latency_ms=f"{latency:.2f}"
                    )
                    return {
                        "status": "FILLED",
                        "fill_price": fill_price,
                        "transaction_id": fill_txn.get("id"),
                        "raw_response": data
                    }
                else:
                    cancel_txn = data.get("orderCancelTransaction", {})
                    logger.warning(
                        "OANDA order cancelled/rejected",
                        reason=cancel_txn.get("reason", "Unknown")
                    )
                    return {"status": "REJECTED", "reason": cancel_txn.get("reason")}
            else:
                logger.error(
                    "OANDA order creation failed",
                    status_code=response.status_code,
                    error=data.get("errorMessage")
                )
                return {"status": "FAILED", "error": data.get("errorMessage")}
                
        except requests.exceptions.RequestException as e:
            logger.error("HTTP Exception placing OANDA order", error=str(e))
            raise  # Let the ExecutionEngine retry logic handle this

    def get_positions(self) -> Dict[str, float]:
        """
        Query all open positions.
        Returns dict of pair -> net size (positive for long, negative for short)
        """
        try:
            response = self.session.get(f"{self.base_url}/openPositions", timeout=5.0)
            response.raise_for_status()
            
            data = response.json()
            positions = {}
            
            for p in data.get("positions", []):
                instrument = p.get("instrument", "").replace("_", "")
                
                long_units = float(p.get("long", {}).get("units", 0.0))
                short_units = float(p.get("short", {}).get("units", 0.0))
                
                net_units = long_units + short_units # short_units are natively negative in OANDA
                if net_units != 0:
                    positions[instrument] = net_units
                    
            return positions
            
        except Exception as e:
            logger.error("Failed to fetch OANDA positions", error=str(e))
            raise

    def get_account_balance(self) -> float:
        """
        Retrieve current NAV (Net Asset Value) of the account.
        """
        try:
            response = self.session.get(f"{self.base_url}/summary", timeout=5.0)
            response.raise_for_status()
            
            nav = float(response.json().get("account", {}).get("NAV", 0.0))
            return nav
            
        except Exception as e:
            logger.error("Failed to fetch OANDA balance", error=str(e))
            raise
