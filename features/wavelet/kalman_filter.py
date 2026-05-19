import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


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
        """
        self.validate(df)
        
        close = df["close"].values
        n = len(df)
        
        # Hyperparameters
        qp = kwargs.get("q_price", self.q_price)
        qv = kwargs.get("q_velocity", self.q_velocity)
        r = kwargs.get("r_observation", self.r_observation)
        
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
