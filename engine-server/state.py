"""Thread-safe live state store for the engine service.

The engine thread writes into this store; the REST handlers and the WebSocket
snapshot read from it. It is the single source of truth for everything the
desktop displays. Ring buffers keep the most recent N events.
"""

import threading
from collections import deque
from typing import Any, Dict, List, Optional

RING_LIMIT = 5000


class StateStore:
    """Holds the full live state of a simulation run."""

    def __init__(self, ring_limit: int = RING_LIMIT) -> None:
        self._lock = threading.RLock()
        self._ring_limit = ring_limit

        # Lifecycle
        self.status: str = "idle"  # idle | loading | running | done | error
        self.error: Optional[str] = None
        self.progress: Dict[str, Any] = {}

        # Portfolio
        self.account: Dict[str, Any] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}

        # End-of-run artifacts
        self.reports: Dict[str, Any] = {}

        # Data/model readiness (health)
        self.data_ready: bool = False
        self.models_loaded: bool = False

        # Bounded ring buffers
        self._signals: deque = deque(maxlen=ring_limit)
        self._trades: deque = deque(maxlen=ring_limit)
        self._orders: deque = deque(maxlen=ring_limit)
        self._alerts: deque = deque(maxlen=ring_limit)
        self._equity: deque = deque(maxlen=ring_limit)

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Clear per-run state before a new simulation starts."""
        with self._lock:
            self.status = "running"
            self.error = None
            self.progress = {}
            self.account = {}
            self.positions = {}
            self.reports = {}
            self._signals.clear()
            self._trades.clear()
            self._orders.clear()
            self._alerts.clear()
            self._equity.clear()

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def set_error(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.status = "error"

    def set_progress(self, progress: Dict[str, Any]) -> None:
        with self._lock:
            self.progress = dict(progress)

    def set_account(self, account: Dict[str, Any]) -> None:
        with self._lock:
            self.account = dict(account)

    def set_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            self.positions = {k: dict(v) for k, v in positions.items()}

    def set_reports(self, reports: Dict[str, Any]) -> None:
        with self._lock:
            self.reports = dict(reports)

    def set_data_ready(self, value: bool) -> None:
        with self._lock:
            self.data_ready = bool(value)

    def set_models_loaded(self, value: bool) -> None:
        with self._lock:
            self.models_loaded = bool(value)

    # -- event appenders -----------------------------------------------------
    def _append(self, buffer: deque, ts: float, data: Dict[str, Any]) -> None:
        entry = dict(data)
        entry.setdefault("ts", ts)
        buffer.append(entry)

    def push_signal(self, data: Dict[str, Any], ts: float) -> None:
        with self._lock:
            self._append(self._signals, ts, data)

    def push_trade(self, data: Dict[str, Any], ts: float) -> None:
        with self._lock:
            self._append(self._trades, ts, data)

    def push_order(self, data: Dict[str, Any], ts: float) -> None:
        with self._lock:
            self._append(self._orders, ts, data)

    def push_alert(self, data: Dict[str, Any], ts: float) -> None:
        with self._lock:
            self._append(self._alerts, ts, data)

    def push_equity(self, data: Dict[str, Any], ts: float) -> None:
        with self._lock:
            self._append(self._equity, ts, data)

    # -- queries --------------------------------------------------------------
    def history(self, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return ring-buffer entries for ``name`` (signals/trades/orders/alerts/equity)."""
        with self._lock:
            buffer = getattr(self, f"_{name}", None)
            if buffer is None:
                return []
            items = list(buffer)
        if limit is not None and limit > 0:
            items = items[-limit:]
        return items

    def snapshot(self) -> Dict[str, Any]:
        """Full state snapshot, sent to a WebSocket client on connect."""
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "progress": dict(self.progress),
                "account": dict(self.account),
                "positions": {k: dict(v) for k, v in self.positions.items()},
                "signals": list(self._signals),
                "trades": list(self._trades),
                "orders": list(self._orders),
                "alerts": list(self._alerts),
                "equity": list(self._equity),
                "reports": dict(self.reports),
                "data_ready": self.data_ready,
                "models_loaded": self.models_loaded,
            }
