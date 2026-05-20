import socket
import time
import datetime
import threading
from typing import Any, Dict, List, Optional
import structlog

from execution.brokers.base_broker import BaseBroker
from risk.risk_engine import OrderRequest

logger = structlog.get_logger()


class LMAXBroker(BaseBroker):
    """
    LMAX Exchange FIX API Adapter.
    
    Implements a lightweight, pure Python FIX 4.4 protocol client over TCP.
    Provides session management (Logon, Heartbeat, Sequence numbers) and 
    order submission (NewOrderSingle) with execution report parsing.
    Gracefully falls back to simulated FIX execution loop if unreachable.
    """

    def __init__(self, name: str = "lmax_broker", config: Any = None) -> None:
        super().__init__(name, config or {})
        
        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 4001)  # LMAX Gateway FIX port
        self.sender_comp_id = self.config.get("sender_comp_id", "CLIENT_SENDER")
        self.target_comp_id = self.config.get("target_comp_id", "LMAX_TARGET")
        self.username = self.config.get("username", "lmax_user")
        self.password = self.config.get("password", "lmax_pass")

        self.seq_num = 1
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.simulated = False
        self.reader_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None

        # Lock for socket writes & sequence numbers
        self.lock = threading.Lock()
        
        # Local state for simulation fallback
        self.sim_positions: Dict[str, float] = {}
        self.sim_cash = self.config.get("initial_capital", 100000.0)

        # Track active fills
        self.pending_orders: Dict[str, Any] = {}
        self.filled_orders: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "LMAXBroker initialized",
            host=self.host,
            port=self.port,
            sender_comp_id=self.sender_comp_id
        )

    def connect(self) -> bool:
        """
        Establishes a TCP socket to the LMAX FIX Gateway and initiates Logon.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(3.0)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.simulated = False
            
            # Start background message reader
            self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.reader_thread.start()
            
            # Send FIX logon
            self._sendLogon()
            
            # Start heartbeats
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            
            logger.info("Connected to LMAX FIX Gateway")
            return True
        except Exception as e:
            logger.warning(
                "Could not connect to live LMAX FIX endpoint. Falling back to Simulated LMAX Session.",
                reason=str(e)
            )
            self.connected = True
            self.simulated = True
            return True

    def disconnect(self) -> None:
        """
        Logoff and close the TCP connection.
        """
        self.connected = False
        if not self.simulated and self.socket:
            try:
                # Send Logoff (35=5)
                self._send_fix_msg("5", [])
                self.socket.close()
                logger.info("Disconnected from LMAX FIX Gateway")
            except Exception as e:
                logger.error("Error disconnecting LMAX socket", error=str(e))
        else:
            logger.info("Simulated LMAX FIX Session ended")

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Submit a New Order Single (35=D) over FIX.
        """
        if self.simulated:
            return self._place_order_simulated(order)

        if not self.connected or self.simulated:
            raise ConnectionError("LMAX FIX Gateway is not connected.")

        cl_ord_id = f"cl_{int(time.time() * 1000)}"
        side = "1" if order.direction == 1 else "2"  # 1 = Buy, 2 = Sell
        ord_type = "1" if (order.order_type or "MARKET").upper() == "MARKET" else "2"  # 1 = Market, 2 = Limit
        
        # Format instrument for LMAX (e.g. EUR/USD)
        symbol = f"{order.pair[:3]}/{order.pair[3:]}" if len(order.pair) == 6 else order.pair

        # FIX 4.4 NewOrderSingle fields
        body_fields = [
            (11, cl_ord_id),        # ClOrdID
            (21, "1"),              # HandlInst (Automated private execution)
            (55, symbol),           # Symbol
            (54, side),             # Side
            (60, datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]),  # TransactTime
            (38, str(order.size)),  # OrderQty
            (40, ord_type),         # OrdType
            (59, "1")               # TimeInForce (1 = IOC - Immediate Or Cancel for market orders)
        ]

        if ord_type == "2":
            if order.limit_price is None:
                raise ValueError("Limit price must be specified for limit orders.")
            body_fields.append((44, f"{order.limit_price:.5f}"))  # Price

        event = threading.Event()
        self.pending_orders[cl_ord_id] = {"event": event, "result": None}

        # Send NewOrderSingle (35=D)
        self._send_fix_msg("D", body_fields)
        
        # Wait up to 5 seconds for execution report (35=8)
        success = event.wait(timeout=5.0)
        
        if success:
            res = self.pending_orders[cl_ord_id]["result"]
            del self.pending_orders[cl_ord_id]
            return res
        else:
            logger.error("Timeout waiting for LMAX ExecutionReport", cl_ord_id=cl_ord_id)
            del self.pending_orders[cl_ord_id]
            return {"status": "REJECTED", "reason": "Timeout"}

    def get_positions(self) -> Dict[str, float]:
        if self.simulated:
            return self.sim_positions.copy()
            
        # LMAX positions are typically tracked via secondary database/API or post-trade drop copy.
        # We maintain them locally in this adapter from ExecutionReports.
        return self.sim_positions.copy()

    def get_account_balance(self) -> float:
        return self.sim_cash

    def _sendLogon(self) -> None:
        body_fields = [
            (98, "0"),       # EncryptMethod
            (108, "30"),     # HeartBtInt (30s)
            (553, self.username),
            (554, self.password)
        ]
        self._send_fix_msg("A", body_fields)
        logger.info("FIX logon message sent")

    def _send_fix_msg(self, msg_type: str, body_fields: List[tuple]) -> None:
        with self.lock:
            # Format header
            header = [
                (35, msg_type),
                (49, self.sender_comp_id),
                (56, self.target_comp_id),
                (34, str(self.seq_num)),
                (52, datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            ]
            self.seq_num += 1
            
            # Combine fields and construct body
            all_fields = header + body_fields
            body_str = "".join(f"{tag}={val}\x01" for tag, val in all_fields)
            
            pre_checksum = f"8=FIX.4.4\x019={len(body_str)}\x01" + body_str
            checksum = sum(ord(c) for c in pre_checksum) % 256
            msg = pre_checksum + f"10={checksum:03d}\x01"
            
            if self.socket and not self.simulated:
                self.socket.sendall(msg.encode("ascii"))

    def _read_loop(self) -> None:
        """
        Read stream from socket, segment messages by SOH, parse them.
        """
        buffer = b""
        while self.connected and self.socket:
            try:
                data = self.socket.recv(4096)
                if not data:
                    logger.warning("LMAX FIX connection closed by remote peer")
                    break
                buffer += data
                
                # Parse messages delimited by SOH and trailer
                while b"10=" in buffer:
                    idx = buffer.find(b"10=")
                    end_idx = buffer.find(b"\x01", idx)
                    if end_idx == -1:
                        break
                    
                    msg_bytes = buffer[:end_idx + 1]
                    buffer = buffer[end_idx + 1:]
                    self._parse_msg(msg_bytes.decode("ascii", errors="ignore"))
            except Exception as e:
                logger.error("Exception in LMAX reader loop", error=str(e))
                break

    def _heartbeat_loop(self) -> None:
        while self.connected and not self.simulated:
            time.sleep(30)
            try:
                self._send_fix_msg("0", [])  # Heartbeat (35=0)
            except Exception as e:
                logger.error("Failed to send LMAX heartbeat", error=str(e))
                break

    def _parse_msg(self, msg_str: str) -> None:
        """
        Deconstruct FIX message tags and evaluate action.
        """
        fields = {}
        for token in msg_str.split("\x01"):
            if "=" in token:
                k, v = token.split("=", 1)
                fields[int(k)] = v

        msg_type = fields.get(35)
        
        if msg_type == "0":  # Heartbeat
            logger.debug("Received FIX Heartbeat")
        elif msg_type == "1":  # Test Request
            test_req_id = fields.get(112, "0")
            self._send_fix_msg("0", [(112, test_req_id)])  # Respond with heartbeat
        elif msg_type == "8":  # Execution Report
            cl_ord_id = fields.get(11)
            exec_type = fields.get(150)  # 150=2 for Fill, 150=8 for Rejected
            avg_price = float(fields.get(6, 0.0))  # 6=AvgPx
            
            if cl_ord_id in self.pending_orders:
                status = "FILLED" if exec_type == "2" else "REJECTED"
                result = {
                    "status": status,
                    "fill_price": avg_price,
                    "transaction_id": fields.get(37),  # OrderID
                    "fill_time": time.time()
                }
                
                # Apply fill locally
                if status == "FILLED":
                    symbol = fields.get(55, "").replace("/", "")
                    qty = float(fields.get(38, 0.0))
                    side = fields.get(54)
                    direction = 1 if side == "1" else -1
                    
                    self.sim_positions[symbol] = self.sim_positions.get(symbol, 0.0) + (qty * direction)
                    if self.sim_positions[symbol] == 0.0:
                        del self.sim_positions[symbol]
                
                self.pending_orders[cl_ord_id]["result"] = result
                self.pending_orders[cl_ord_id]["event"].set()

    def _place_order_simulated(self, order: OrderRequest) -> Dict[str, Any]:
        """
        Simulated FIX fill response generator for testing.
        """
        pair = order.pair
        direction = order.direction
        size = order.size
        
        fill_price = order.limit_price if order.order_type == "LIMIT" and order.limit_price else 1.05
        
        current_pos = self.sim_positions.get(pair, 0.0)
        self.sim_positions[pair] = current_pos + (size * direction)
        
        if self.sim_positions[pair] == 0.0:
            del self.sim_positions[pair]
            
        logger.debug(
            "LMAX simulated FIX fill",
            pair=pair,
            size=size,
            direction=direction,
            fill_price=fill_price
        )
        
        return {
            "status": "FILLED",
            "fill_price": fill_price,
            "transaction_id": f"sim_lmax_{int(time.time() * 1000)}",
            "fill_time": time.time()
        }
