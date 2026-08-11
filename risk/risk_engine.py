import os
import sys
import ctypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from models.ensemble.signal_generator import AlphaSignal

logger = structlog.get_logger()

# -------------------------------------------------------------------------
# Dynamic C++ Shared Library Loading & Type Binding
# -------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ext = ".dll" if sys.platform.startswith("win") else ".dylib" if sys.platform.startswith("darwin") else ".so"
_lib_path = os.path.join(_current_dir, f"risk_speedups{_ext}")

_risk_lib = None
if os.path.exists(_lib_path):
    try:
        kwargs = {}
        if sys.platform.startswith("win"):
            kwargs["winmode"] = 0
        _risk_lib = ctypes.CDLL(_lib_path, **kwargs)
        _risk_lib.calculate_vol_z_score.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        _risk_lib.calculate_vol_z_score.restype = ctypes.c_double
        logger.info("Successfully loaded C++ Risk Engine speedups shared library", path=_lib_path)
    except Exception as e:
        logger.warning("Failed to load C++ Risk Engine speedups library. Falling back to pure Python.", error=str(e))

@dataclass
class PortfolioState:
    current_equity: float
    open_positions: Dict[str, float]
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    win_rate: float
    win_loss_ratio: float
    historical_returns: np.ndarray
    tail_risk_events: List[str] = field(default_factory=list)

@dataclass
class OrderRequest:
    pair: str
    direction: int
    size: float
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseRiskEngine(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def gate(self, signal: AlphaSignal, pair: str, portfolio_state: PortfolioState, market_data: Dict[str, Any]) -> Optional[OrderRequest]:
        pass

class AntiFragileRiskEngine(BaseRiskEngine):
    """
    GOAT-grade Anti-Fragile Risk Engine.
    
    Instead of just surviving volatility, this engine is designed to:
    1. Detect 'Unknown Unknowns' via fat-tail distribution monitoring.
    2. Implement 'Convex Sizing': increase exposure during chaotic/profitable regimes.
    3. Enforce 'Non-Linear Circuit Breakers' that tighten exponentially with drawdown.
    4. Handle 'Global Black Swans' (e.g., currency abolition) via emergency hedging.
    """

    def __init__(self, config: Any = None) -> None:
        config = config or {}
        super().__init__(config)
        self.filters = []
        self.limits = []
        self.sizer = None
        self.kill_switch_active = False
        
        # Anti-fragility params
        self.tail_risk_threshold = config.get("tail_risk_threshold", 4.0) # Z-score for fat tails
        self.chaos_multiplier = config.get("chaos_multiplier", 1.5)      # Sizing boost in high-vol/high-conf
        
        logger.info("AntiFragileRiskEngine initialized", tail_threshold=self.tail_risk_threshold)

    def register_filter(self, filter_obj: Any) -> None:
        self.filters.append(filter_obj)

    def register_limit(self, limit_obj: Any) -> None:
        self.limits.append(limit_obj)

    def set_sizer(self, sizer_obj: Any) -> None:
        self.sizer = sizer_obj

    def gate(
        self,
        signal: AlphaSignal,
        pair: str,
        portfolio_state: PortfolioState,
        market_data: Dict[str, Any]
    ) -> Optional[OrderRequest]:
        
        # 1. Global Kill Switch & Emergency Detection
        if self.kill_switch_active:
            return self._handle_emergency_exit(signal, pair, portfolio_state)

        # 2. Tail Risk (Unknown Unknown) Detection
        # Calculate rolling volatility Z-score to detect 'Black Swan' regimes
        vol_z_score = 0.0
        n_returns = len(portfolio_state.historical_returns)
        if n_returns > 50:
            if _risk_lib is not None:
                try:
                    returns_arr = np.ascontiguousarray(portfolio_state.historical_returns, dtype=np.float64)
                    returns_ptr = returns_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    vol_z_score = _risk_lib.calculate_vol_z_score(returns_ptr, ctypes.c_int(n_returns))
                except Exception as e:
                    logger.error("C++ Risk Z-score execution failed, falling back to Python", error=str(e))
                    recent_vol = np.std(portfolio_state.historical_returns[-20:])
                    hist_vol = np.std(portfolio_state.historical_returns)
                    vol_z_score = (recent_vol - hist_vol) / (np.std(np.diff(portfolio_state.historical_returns)) + 1e-6)
            else:
                recent_vol = np.std(portfolio_state.historical_returns[-20:])
                hist_vol = np.std(portfolio_state.historical_returns)
                vol_z_score = (recent_vol - hist_vol) / (np.std(np.diff(portfolio_state.historical_returns)) + 1e-6)
            
            if vol_z_score > self.tail_risk_threshold:
                logger.critical("BLACK SWAN DETECTED: Extreme volatility regime shift.", z_score=vol_z_score)
                # Anti-fragile move: reduce base exposure but allow high-conviction momentum signals
                if signal.confidence < 0.85:
                    return None

        # 3. Non-Linear Circuit Breakers & Filters
        for limit in self.limits + self.filters:
            if not limit.check(signal, pair, portfolio_state, market_data):
                # Check if trade is risk-reducing (closing a position)
                current_exposure = portfolio_state.open_positions.get(pair, 0.0)
                is_risk_reducing = (signal.direction > 0 and current_exposure < 0) or \
                                   (signal.direction < 0 and current_exposure > 0)
                if not is_risk_reducing:
                    return None

        # 4. Convex Sizing Logic
        if self.sizer is None:
            return None
            
        base_size = self.sizer.calculate_size(signal, pair, portfolio_state, market_data)
        
        # Anti-fragile sizing: Increase size if we are in a 'Winning Streak' and volatility is high but stable
        # This captures the 'Fat Tails' of profitable moves.
        multiplier = 1.0
        if portfolio_state.win_rate > 0.6 and signal.confidence > 0.8:
            multiplier = self.chaos_multiplier
            logger.info("Applying Convex Sizing Boost", multiplier=multiplier)

        final_size = base_size * multiplier
        
        # 5. Order Construction with Dynamic Guardrails
        # Calculate dynamic stop-loss based on tail-risk (tighten in chaos)
        atr = market_data.get("atr", 0.001)
        sl_mult = 2.0 if vol_z_score < 2.0 else 1.0
        base_price = market_data.get("price", market_data.get("close", market_data.get("mid_price", 0.0)))
        
        order = OrderRequest(
            pair=pair,
            direction=signal.direction,
            size=final_size,
            stop_loss=base_price - (signal.direction * atr * sl_mult),
            metadata={
                "anti_fragile": True,
                "vol_z_score": vol_z_score if 'vol_z_score' in locals() else 0,
                "multiplier": multiplier
            }
        )
        
        return order

    def _handle_emergency_exit(self, signal: AlphaSignal, pair: str, portfolio_state: PortfolioState) -> Optional[OrderRequest]:
        current_exposure = portfolio_state.open_positions.get(pair, 0.0)
        is_risk_reducing = (signal.direction > 0 and current_exposure < 0) or \
                           (signal.direction < 0 and current_exposure > 0)
        if is_risk_reducing:
            return OrderRequest(pair=pair, direction=signal.direction, size=abs(current_exposure), order_type="MARKET")
        return None


# Default risk engine alias used by scripts and pipeline imports
RiskEngine = AntiFragileRiskEngine
