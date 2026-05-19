import os
import sys
import ctypes
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import structlog
from typing import List, Dict, Any, Tuple, Optional

from models.rl_agent.reward_functions import ForexRewardEngine

logger = structlog.get_logger()

# -------------------------------------------------------------------------
# Dynamic C++ Shared Library Loading & Type Binding
# -------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith("win"):
    _lib_name = "rl_speedups.dll"
elif sys.platform.startswith("darwin"):
    _lib_name = "rl_speedups.dylib"
else:
    _lib_name = "rl_speedups.so"
_lib_path = os.path.join(_current_dir, _lib_name)

_rl_lib = None
if os.path.exists(_lib_path):
    try:
        _rl_lib = ctypes.CDLL(_lib_path)
        
        # Bind calculate_portfolio_step
        _rl_lib.calculate_portfolio_step.argtypes = [
            ctypes.c_double,                  # target_pos
            ctypes.c_double,                  # mid_price
            ctypes.c_double,                  # prev_price
            ctypes.c_double,                  # spread
            ctypes.c_double,                  # balance
            ctypes.c_double,                  # entry_price
            ctypes.c_double,                  # current_pos
            ctypes.c_double,                  # leverage
            ctypes.c_double,                  # margin_pct
            ctypes.c_double,                  # multiplier
            ctypes.c_double,                  # kyle_lambda
            ctypes.c_double,                  # slippage
            ctypes.POINTER(ctypes.c_double),  # out_realized_pnl
            ctypes.POINTER(ctypes.c_double),  # out_unrealized_pnl
            ctypes.POINTER(ctypes.c_double),  # out_new_position
            ctypes.POINTER(ctypes.c_double),  # out_new_balance
            ctypes.POINTER(ctypes.c_double),  # out_new_entry_price
            ctypes.POINTER(ctypes.c_int)      # out_margin_called
        ]
        _rl_lib.calculate_portfolio_step.restype = None

        # Bind calculate_portfolio_loop
        _rl_lib.calculate_portfolio_loop.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # target_positions
            ctypes.POINTER(ctypes.c_double),  # mid_prices
            ctypes.POINTER(ctypes.c_double),  # spreads
            ctypes.POINTER(ctypes.c_double),  # kyle_lambdas
            ctypes.c_int,                     # n
            ctypes.c_double,                  # initial_balance
            ctypes.c_double,                  # leverage
            ctypes.c_double,                  # margin_pct
            ctypes.c_double,                  # multiplier
            ctypes.c_double,                  # slippage
            ctypes.POINTER(ctypes.c_double),  # out_balances
            ctypes.POINTER(ctypes.c_double),  # out_positions
            ctypes.POINTER(ctypes.c_double),  # out_entry_prices
            ctypes.POINTER(ctypes.c_double),  # out_realized_pnls
            ctypes.POINTER(ctypes.c_double),  # out_unrealized_pnls
            ctypes.POINTER(ctypes.c_int)      # out_margin_called_steps
        ]
        _rl_lib.calculate_portfolio_loop.restype = None

        logger.info("Successfully loaded C++ RL portfolio speedups shared library", path=_lib_path)
    except Exception as e:
        logger.warning("Failed to load C++ RL portfolio speedups library. Falling back to pure Python.", error=str(e))
else:
    logger.warning("C++ RL portfolio speedups library not found. Falling back to pure Python.", path=_lib_path)


class ForexTradingEnv(gym.Env):
    """
    Gymnasium compliant Forex Trading Environment with C++ accelerated hot paths.
    Fuses rolling features, current position, unrealized PnL, timing indicators,
    and HMM regime probabilities into its observation space.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features_cols: List[str],
        regime_cols: Optional[List[str]] = None,
        initial_balance: float = 10000.0,
        leverage: float = 30.0,
        multiplier: float = 100000.0,  # EURUSD Standard Lot
        slippage: float = 0.0001,      # Base slippage in price units
        kyle_lambda_multiplier: float = 1.0,
        reward_config: Optional[Dict[str, Any]] = None,
        force_python_fallback: bool = False,
        action_space_type: str = "discrete"
    ) -> None:
        super().__init__()
        
        self.df = df.copy()
        self.features_cols = features_cols
        self.regime_cols = regime_cols or []
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.margin_pct = 1.0 / leverage
        self.multiplier = multiplier
        self.slippage = slippage
        self.kyle_lambda_multiplier = kyle_lambda_multiplier
        self.force_python_fallback = force_python_fallback
        self.action_space_type = action_space_type.lower()
        
        # Validations
        required_cols = ["close"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")
                
        # Optional columns fallbacks
        if "spread" not in self.df.columns:
            self.df["spread"] = 0.0002  # 2 pips default
        if "kyle_lambda" not in self.df.columns:
            self.df["kyle_lambda"] = 0.0  # no market impact by default
        if "vpin" not in self.df.columns:
            self.df["vpin"] = 0.0  # no toxic flow default

        # Pre-extract numpy arrays for ultra-fast step execution
        self.close_arr = self.df["close"].values.astype(np.float64)
        self.spread_arr = self.df["spread"].values.astype(np.float64)
        self.kyle_lambda_arr = self.df["kyle_lambda"].values.astype(np.float64)
        self.vpin_arr = self.df["vpin"].values.astype(np.float64)
        
        self.features_arr = self.df[self.features_cols].values.astype(np.float32)
        if self.regime_cols:
            self.regime_arr = self.df[self.regime_cols].values.astype(np.float32)
        else:
            self.regime_arr = None

        if isinstance(self.df.index, pd.DatetimeIndex):
            self.hour_arr = (self.df.index.hour.values.astype(np.float32)) / 23.0
        else:
            self.hour_arr = (np.arange(len(self.df)) % 24).astype(np.float32) / 23.0
            
        # Action space configuration
        if self.action_space_type == "discrete":
            self.action_space = spaces.Discrete(5)
            self.action_to_position = {
                0: 0.0,
                1: 0.5,
                2: 1.0,
                3: -0.5,
                4: -1.0
            }
        elif self.action_space_type == "continuous":
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action_space_type: {action_space_type}. Must be 'discrete' or 'continuous'.")
        
        # Observation dimension calculation
        # [len(features_cols)] + [position] + [unrealized_pnl_normalized] + [time_indicator] + [len(regime_cols)]
        self.obs_dim = len(self.features_cols) + 1 + 1 + 1 + len(self.regime_cols)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        # Setup Reward Engine
        self.reward_engine = ForexRewardEngine(reward_config)
        
        # State variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.margin_called = False
        
        logger.info(
            "ForexTradingEnv initialized",
            obs_dim=self.obs_dim,
            features=len(self.features_cols),
            regimes=len(self.regime_cols),
            initial_balance=self.initial_balance,
            leverage=self.leverage
        )

    def _get_time_indicator(self) -> float:
        """Returns normalized time indicator from pre-extracted array."""
        return float(self.hour_arr[self.current_step])

    def _get_observation(self) -> np.ndarray:
        """Fuses features, portfolio, timing, and HMM regimes into a single vector."""
        feats = self.features_arr[self.current_step]
        
        # Fused components
        pos = np.array([self.position], dtype=np.float32)
        unrealized = np.array([self.unrealized_pnl / self.initial_balance], dtype=np.float32)
        time_ind = np.array([self._get_time_indicator()], dtype=np.float32)
        
        # Regimes if present
        if self.regime_arr is not None:
            regimes = self.regime_arr[self.current_step]
            obs = np.concatenate([feats, pos, unrealized, time_ind, regimes])
        else:
            obs = np.concatenate([feats, pos, unrealized, time_ind])
            
        # Handle potential NaNs or infinite values gracefully
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.margin_called = False
        
        self.reward_engine.reset()
        
        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "equity": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "margin_called": 0
        }
        
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.margin_called or self.current_step >= len(self.df) - 1:
            obs = self._get_observation()
            info = {
                "balance": self.balance,
                "equity": self.balance + self.unrealized_pnl,
                "position": self.position,
                "entry_price": self.entry_price,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "margin_called": int(self.margin_called)
            }
            return obs, 0.0, True, False, info

        if self.action_space_type == "discrete":
            target_pos = self.action_to_position[int(action)]
        else:
            if isinstance(action, (np.ndarray, list)):
                target_pos = float(action[0])
            else:
                target_pos = float(action)
            target_pos = max(-1.0, min(1.0, target_pos))
        
        # Get quotes from fast numpy arrays
        mid_price = self.close_arr[self.current_step + 1]
        prev_price = self.close_arr[self.current_step]
        spread = self.spread_arr[self.current_step + 1]
        kyle_lambda = self.kyle_lambda_arr[self.current_step + 1] * self.kyle_lambda_multiplier
        vpin = self.vpin_arr[self.current_step + 1]
        
        prev_position = self.position
        prev_equity = self.balance + self.unrealized_pnl
        
        out_realized_pnl = 0.0
        out_unrealized_pnl = 0.0
        out_new_position = self.position
        out_new_balance = self.balance
        out_new_entry_price = self.entry_price
        out_margin_called = 0
        
        # Try C++ speedup first unless forced to python fallback
        if _rl_lib is not None and not self.force_python_fallback:
            try:
                realized_pnl_c = ctypes.c_double(0.0)
                unrealized_pnl_c = ctypes.c_double(0.0)
                new_position_c = ctypes.c_double(0.0)
                new_balance_c = ctypes.c_double(0.0)
                new_entry_price_c = ctypes.c_double(0.0)
                margin_called_c = ctypes.c_int(0)
                
                # Execute C++ call
                _rl_lib.calculate_portfolio_step(
                    ctypes.c_double(target_pos),
                    ctypes.c_double(mid_price),
                    ctypes.c_double(prev_price),
                    ctypes.c_double(spread),
                    ctypes.c_double(self.balance),
                    ctypes.c_double(self.entry_price),
                    ctypes.c_double(self.position),
                    ctypes.c_double(self.leverage),
                    ctypes.c_double(self.margin_pct),
                    ctypes.c_double(self.multiplier),
                    ctypes.c_double(kyle_lambda),
                    ctypes.c_double(self.slippage),
                    ctypes.byref(realized_pnl_c),
                    ctypes.byref(unrealized_pnl_c),
                    ctypes.byref(new_position_c),
                    ctypes.byref(new_balance_c),
                    ctypes.byref(new_entry_price_c),
                    ctypes.byref(margin_called_c)
                )
                
                out_realized_pnl = realized_pnl_c.value
                out_unrealized_pnl = unrealized_pnl_c.value
                out_new_position = new_position_c.value
                out_new_balance = new_balance_c.value
                out_new_entry_price = new_entry_price_c.value
                out_margin_called = margin_called_c.value
                
            except Exception as e:
                logger.error("C++ RL portfolio step failed, falling back to Python", error=str(e))
                out_margin_called = -1  # flag to trigger python execution below
                
        # Pure Python fallback execution
        if _rl_lib is None or self.force_python_fallback or out_margin_called == -1:
            active_pos = self.position
            active_balance = self.balance
            active_entry = self.entry_price
            out_realized_pnl = 0.0
            
            price_delta = mid_price - prev_price
            delta_unrealized = 0.0
            if abs(active_pos) > 1e-8:
                delta_unrealized = active_pos * price_delta * self.multiplier
                
            trade_size = target_pos - active_pos
            if abs(trade_size) > 1e-8:
                tx_cost = abs(trade_size) * self.multiplier * ((spread / 2.0) + (kyle_lambda * self.slippage))
                active_balance -= tx_cost
                
                if abs(active_pos) < 1e-8:
                    active_pos = target_pos
                    active_entry = mid_price
                elif (active_pos > 0.0 and target_pos > 0.0) or (active_pos < 0.0 and target_pos < 0.0):
                    if abs(target_pos) > abs(active_pos):
                        active_entry = ((active_pos * active_entry) + (trade_size * mid_price)) / target_pos
                        active_pos = target_pos
                    else:
                        closed_size = abs(trade_size)
                        if active_pos > 0.0:
                            realized_trade_pnl = closed_size * (mid_price - active_entry) * self.multiplier
                        else:
                            realized_trade_pnl = closed_size * (active_entry - mid_price) * self.multiplier
                        out_realized_pnl = realized_trade_pnl
                        active_balance += realized_trade_pnl
                        active_pos = target_pos
                else:
                    closed_size = abs(active_pos)
                    if active_pos > 0.0:
                        realized_trade_pnl = closed_size * (mid_price - active_entry) * self.multiplier
                    else:
                        realized_trade_pnl = closed_size * (active_entry - mid_price) * self.multiplier
                    out_realized_pnl = realized_trade_pnl
                    active_balance += realized_trade_pnl
                    
                    active_pos = target_pos
                    if abs(target_pos) > 1e-8:
                        active_entry = mid_price
                    else:
                        active_entry = 0.0
                        
            active_unrealized = 0.0
            if abs(active_pos) > 1e-8:
                active_unrealized = active_pos * (mid_price - active_entry) * self.multiplier
                
            equity = active_balance + active_unrealized
            margin_required = 0.0
            if abs(active_pos) > 1e-8:
                margin_required = abs(active_pos) * mid_price * self.multiplier * self.margin_pct
                
            if margin_required > 0.0 and equity <= (margin_required * 0.5):
                out_margin_called = 1
                out_new_position = 0.0
                out_new_balance = max(0.0, equity)
                out_new_entry_price = 0.0
                out_unrealized_pnl = 0.0
                out_realized_pnl += (max(0.0, equity) - self.balance)
            else:
                out_margin_called = 0
                out_new_position = active_pos
                out_new_balance = active_balance
                out_new_entry_price = active_entry
                out_unrealized_pnl = active_unrealized

        # Save states
        self.realized_pnl = out_realized_pnl
        self.unrealized_pnl = out_unrealized_pnl
        self.position = out_new_position
        self.balance = out_new_balance
        self.entry_price = out_new_entry_price
        self.margin_called = (out_margin_called == 1)
        
        # Calculate step return
        current_equity = self.balance + self.unrealized_pnl
        step_return = 0.0
        if prev_equity > 0.0:
            step_return = (current_equity - prev_equity) / prev_equity
            
        # Step increment
        self.current_step += 1
        
        # Gymnasium state terminal indicators
        terminated = (self.current_step >= len(self.df) - 1)
        truncated = self.margin_called
        
        info = {
            "balance": self.balance,
            "equity": current_equity,
            "position": self.position,
            "entry_price": self.entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "margin_called": out_margin_called
        }
        
        # Calculate reward
        reward = self.reward_engine.calculate_reward(
            step_return=step_return,
            current_position=self.position,
            prev_position=prev_position,
            vpin=vpin,
            info=info
        )
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def simulate_trajectory(self, target_positions: np.ndarray, force_python: bool = False) -> Dict[str, np.ndarray]:
        """
        Runs a sequential portfolio simulation over the entire historical sequence.
        Routes to the highly optimized C++ loop function if available.
        """
        n = len(self.df)
        assert len(target_positions) >= n - 1, "target_positions must contain at least n-1 trade actions."

        # Try C++ Acceleration First
        if _rl_lib is not None and not force_python:
            try:
                # Contiguous input arrays
                target_positions_c = np.ascontiguousarray(target_positions[:n-1], dtype=np.float64)
                mid_prices_c = np.ascontiguousarray(self.close_arr, dtype=np.float64)
                spreads_c = np.ascontiguousarray(self.spread_arr, dtype=np.float64)
                kyle_lambdas_c = np.ascontiguousarray(self.kyle_lambda_arr * self.kyle_lambda_multiplier, dtype=np.float64)
                
                # Preallocate output arrays
                out_balances = np.zeros(n, dtype=np.float64)
                out_positions = np.zeros(n, dtype=np.float64)
                out_entry_prices = np.zeros(n, dtype=np.float64)
                out_realized_pnls = np.zeros(n, dtype=np.float64)
                out_unrealized_pnls = np.zeros(n, dtype=np.float64)
                out_margin_called_steps = np.zeros(n, dtype=np.int32)
                
                # Pointers
                target_positions_ptr = target_positions_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                mid_prices_ptr = mid_prices_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                spreads_ptr = spreads_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                kyle_lambdas_ptr = kyle_lambdas_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                
                out_balances_ptr = out_balances.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_positions_ptr = out_positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_entry_prices_ptr = out_entry_prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_realized_pnls_ptr = out_realized_pnls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_unrealized_pnls_ptr = out_unrealized_pnls.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                out_margin_called_ptr = out_margin_called_steps.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                
                _rl_lib.calculate_portfolio_loop(
                    target_positions_ptr,
                    mid_prices_ptr,
                    spreads_ptr,
                    kyle_lambdas_ptr,
                    ctypes.c_int(n),
                    ctypes.c_double(self.initial_balance),
                    ctypes.c_double(self.leverage),
                    ctypes.c_double(self.margin_pct),
                    ctypes.c_double(self.multiplier),
                    ctypes.c_double(self.slippage),
                    out_balances_ptr,
                    out_positions_ptr,
                    out_entry_prices_ptr,
                    out_realized_pnls_ptr,
                    out_unrealized_pnls_ptr,
                    out_margin_called_ptr
                )
                
                return {
                    "balance": out_balances,
                    "position": out_positions,
                    "entry_price": out_entry_prices,
                    "realized_pnl": out_realized_pnls,
                    "unrealized_pnl": out_unrealized_pnls,
                    "margin_called": out_margin_called_steps
                }
            except Exception as e:
                logger.error("C++ simulate_trajectory failed, falling back to Python loop", error=str(e))

        # Pure Python fallback loop
        out_balances = np.zeros(n, dtype=np.float64)
        out_positions = np.zeros(n, dtype=np.float64)
        out_entry_prices = np.zeros(n, dtype=np.float64)
        out_realized_pnls = np.zeros(n, dtype=np.float64)
        out_unrealized_pnls = np.zeros(n, dtype=np.float64)
        out_margin_called_steps = np.zeros(n, dtype=np.int32)
        
        out_balances[0] = self.initial_balance
        current_balance = self.initial_balance
        current_position = 0.0
        current_entry_price = 0.0
        margin_called = False
        
        for i in range(n - 1):
            if margin_called:
                out_balances[i + 1] = current_balance
                out_positions[i + 1] = 0.0
                out_entry_prices[i + 1] = 0.0
                out_realized_pnls[i + 1] = 0.0
                out_unrealized_pnls[i + 1] = 0.0
                out_margin_called_steps[i + 1] = 1
                continue
                
            target_pos = target_positions[i]
            mid_price = self.close_arr[i + 1]
            prev_price = self.close_arr[i]
            spread = self.spread_arr[i + 1]
            kyle_lambda = self.kyle_lambda_arr[i + 1] * self.kyle_lambda_multiplier
            
            # Inline sequential portfolio mathematics
            active_pos = current_position
            active_balance = current_balance
            active_entry = current_entry_price
            realized_pnl = 0.0
            
            trade_size = target_pos - active_pos
            if abs(trade_size) > 1e-8:
                tx_cost = abs(trade_size) * self.multiplier * ((spread / 2.0) + (kyle_lambda * self.slippage))
                active_balance -= tx_cost
                
                if abs(active_pos) < 1e-8:
                    active_pos = target_pos
                    active_entry = mid_price
                elif (active_pos > 0.0 and target_pos > 0.0) or (active_pos < 0.0 and target_pos < 0.0):
                    if abs(target_pos) > abs(active_pos):
                        active_entry = ((active_pos * active_entry) + (trade_size * mid_price)) / target_pos
                        active_pos = target_pos
                    else:
                        closed_size = abs(trade_size)
                        if active_pos > 0.0:
                            realized_trade_pnl = closed_size * (mid_price - active_entry) * self.multiplier
                        else:
                            realized_trade_pnl = closed_size * (active_entry - mid_price) * self.multiplier
                        realized_pnl = realized_trade_pnl
                        active_balance += realized_trade_pnl
                        active_pos = target_pos
                else:
                    closed_size = abs(active_pos)
                    if active_pos > 0.0:
                        realized_trade_pnl = closed_size * (mid_price - active_entry) * self.multiplier
                    else:
                        realized_trade_pnl = closed_size * (active_entry - mid_price) * self.multiplier
                    realized_pnl = realized_trade_pnl
                    active_balance += realized_trade_pnl
                    
                    active_pos = target_pos
                    if abs(target_pos) > 1e-8:
                        active_entry = mid_price
                    else:
                        active_entry = 0.0
                        
            active_unrealized = 0.0
            if abs(active_pos) > 1e-8:
                active_unrealized = active_pos * (mid_price - active_entry) * self.multiplier
                
            equity = active_balance + active_unrealized
            margin_required = 0.0
            if abs(active_pos) > 1e-8:
                margin_required = abs(active_pos) * mid_price * self.multiplier * self.margin_pct
                
            if margin_required > 0.0 and equity <= (margin_required * 0.5):
                margin_called = True
                current_position = 0.0
                current_balance = max(0.0, equity)
                current_entry_price = 0.0
                realized_pnl += (max(0.0, equity) - current_balance)
                active_unrealized = 0.0
                out_margin_called_steps[i + 1] = 1
            else:
                current_position = active_pos
                current_balance = active_balance
                current_entry_price = active_entry
                out_margin_called_steps[i + 1] = 0
                
            out_balances[i + 1] = current_balance
            out_positions[i + 1] = current_position
            out_entry_prices[i + 1] = current_entry_price
            out_realized_pnls[i + 1] = realized_pnl
            out_unrealized_pnls[i + 1] = active_unrealized

        return {
            "balance": out_balances,
            "position": out_positions,
            "entry_price": out_entry_prices,
            "realized_pnl": out_realized_pnls,
            "unrealized_pnl": out_unrealized_pnls,
            "margin_called": out_margin_called_steps
        }

    def render(self, mode: str = "human") -> None:
        equity = self.balance + self.unrealized_pnl
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Equity: {equity:.2f} | Position: {self.position} | Realized PnL: {self.realized_pnl:.4f} | Unrealized PnL: {self.unrealized_pnl:.4f}")
