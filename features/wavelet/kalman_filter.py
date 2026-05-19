import os
import sys
import ctypes
import structlog
import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature

logger = structlog.get_logger()

# -------------------------------------------------------------------------
# Dynamic C++ Shared Library Loading & Type Binding
# -------------------------------------------------------------------------
_current_dir = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith("win"):
    _lib_name = "kalman_speedups.dll"
elif sys.platform.startswith("darwin"):
    _lib_name = "kalman_speedups.dylib"
else:
    _lib_name = "kalman_speedups.so"
_lib_path = os.path.join(_current_dir, _lib_name)

_kalman_lib = None
if os.path.exists(_lib_path):
    try:
        _kalman_lib = ctypes.CDLL(_lib_path)
        # Bind argument types and return type
        # void kalman_filter_2d(const double* close, int n, double qp, double qv, double r, double* filtered_price, double* velocity_estimate)
        _kalman_lib.kalman_filter_2d.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # close
            ctypes.c_int,                     # n
            ctypes.c_double,                  # qp
            ctypes.c_double,                  # qv
            ctypes.c_double,                  # r
            ctypes.POINTER(ctypes.c_double),  # filtered_price
            ctypes.POINTER(ctypes.c_double),  # velocity_estimate
        ]
        _kalman_lib.kalman_filter_2d.restype = None
        logger.info("Successfully loaded C++ Kalman speedups shared library", path=_lib_path)
    except Exception as e:
        logger.warning("Failed to load C++ Kalman speedups library. Falling back to pure Python.", error=str(e))
else:
    logger.warning("C++ Kalman speedups library not found. Falling back to pure Python.", path=_lib_path)


class KalmanStateFilter(BaseFeature):
    """
    State-Space Kalman Filter.
    State vector: x = [price, velocity]^T
    Observation: z = raw mid/close price
    Enables removing high-frequency microstructural noise to obtain the true
    underlying price and a smoothed trend velocity estimate.
    """

    def __init__(self, name: str = "kalman", config: Any = None) -> None:
        super().__init__(name, config)
        # Default covariance parameters
        self.q_price = 1e-4 if not config else config.get("q_price", 1e-4)
        self.q_velocity = 1e-5 if not config else config.get("q_velocity", 1e-5)
        self.r_observation = 1e-2 if not config else config.get("r_observation", 1e-2)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required column 'close' exists.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column as raw price observation.")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes recursive 2D Kalman filter over price observations.
        Routes to fast C++ implementation if available; otherwise falls back to pure Python.
        """
        self.validate(df)
        
        close = df["close"].values
        n = len(df)
        
        # Hyperparameters
        qp = kwargs.get("q_price", self.q_price)
        qv = kwargs.get("q_velocity", self.q_velocity)
        r = kwargs.get("r_observation", self.r_observation)
        
        # Try C++ Acceleration first
        if _kalman_lib is not None:
            try:
                # Preallocate contiguous numpy arrays for C++ (no copies if possible)
                close_arr = np.ascontiguousarray(close, dtype=np.float64)
                filtered_price = np.ascontiguousarray(np.zeros(n), dtype=np.float64)
                velocity_estimate = np.ascontiguousarray(np.zeros(n), dtype=np.float64)
                
                # Get pointers to the arrays
                close_ptr = close_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                filtered_ptr = filtered_price.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                velocity_ptr = velocity_estimate.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                
                # Call compiled C++ function
                _kalman_lib.kalman_filter_2d(
                    close_ptr,
                    ctypes.c_int(n),
                    ctypes.c_double(qp),
                    ctypes.c_double(qv),
                    ctypes.c_double(r),
                    filtered_ptr,
                    velocity_ptr
                )
                
                return pd.DataFrame(
                    {
                        f"{self.name}_price": pd.Series(filtered_price, index=df.index),
                        f"{self.name}_velocity": pd.Series(velocity_estimate, index=df.index),
                    },
                    index=df.index
                )
            except Exception as e:
                logger.error("C++ Kalman execution failed, falling back to Python", error=str(e))

        # -------------------------------------------------------------------------
        # Fallback Pure Python Implementation
        # -------------------------------------------------------------------------
        # 1. Initialize State space matrices
        # Transition matrix A
        A = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
                      
        # Observation matrix H
        H = np.array([[1.0, 0.0]])
        
        # Process noise covariance Q
        Q = np.array([[qp, 0.0],
                      [0.0, qv]])
                      
        # Observation noise covariance R
        R = np.array([[r]])
        
        # 2. Preallocate arrays
        filtered_price = np.zeros(n)
        velocity_estimate = np.zeros(n)
        
        # 3. Kalman Filter Loop
        # Initial state estimate: [initial close, 0]^T
        x = np.array([[close[0]],
                      [0.0]])
                      
        # Initial covariance P
        P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
                      
        filtered_price[0] = x[0, 0]
        velocity_estimate[0] = x[1, 0]
        
        for k in range(1, n):
            z = close[k]
            
            # Predict
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            
            # Update
            # Innovation: y = z - Hx_pred
            y = z - (H @ x_pred)[0, 0]
            
            # Innovation Covariance: S = H P_pred H^T + R
            S = (H @ P_pred @ H.T + R)[0, 0]
            
            # Avoid division by zero
            if abs(S) < 1e-15:
                S = 1e-15 if S >= 0 else -1e-15

            # Kalman Gain: K = P_pred H^T S^-1
            K = (P_pred @ H.T) / S
            
            # Update State: x = x_pred + K y
            x = x_pred + K * y
            
            # Update Covariance: P = (I - KH) P_pred
            I = np.eye(2)
            P = (I - K @ H) @ P_pred
            
            # Store values
            filtered_price[k] = x[0, 0]
            velocity_estimate[k] = x[1, 0]
            
        result = pd.DataFrame(
            {
                f"{self.name}_price": pd.Series(filtered_price, index=df.index),
                f"{self.name}_velocity": pd.Series(velocity_estimate, index=df.index),
            },
            index=df.index
        )
        return result

