import unittest
import time
import pandas as pd
import numpy as np
from features.wavelet.kalman_filter import KalmanStateFilter, _kalman_lib


class TestCPPKalmanFilter(unittest.TestCase):
    def setUp(self) -> None:
        # Create a mock dataset of 5,000 steps for speed and correctness verification
        np.random.seed(42)
        self.n = 5000
        time_index = pd.date_range(start="2026-01-01", periods=self.n, freq="s")
        
        # Generate random walk mid prices with microstructural noise
        steps = np.random.normal(0, 0.0002, self.n)
        self.close_clean = 1.1000 + np.cumsum(steps)
        noise = np.random.normal(0, 0.001, self.n)
        self.close_noisy = self.close_clean + noise
        
        self.df = pd.DataFrame(
            {"close": self.close_noisy},
            index=time_index
        )
        
        # Filter hyperparameters
        self.qp = 1e-4
        self.qv = 1e-5
        self.r = 1e-2

    def test_cpp_library_loaded(self) -> None:
        """Verify that the compiled C++ Kalman speedups library is loaded successfully."""
        self.assertIsNotNone(_kalman_lib, "C++ Kalman speedups library should be loaded and not None")

    def test_precision_parity(self) -> None:
        """
        Verify flawless precision parity (10^-12 decimal places) between
        the C++ implementation and the pure Python reference implementation.
        """
        # 1. Run Pure Python Reference Implementation
        close = self.df["close"].values
        n = len(close)
        filtered_price_py = np.zeros(n)
        velocity_estimate_py = np.zeros(n)
        
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[self.qp, 0.0], [0.0, self.qv]])
        R = np.array([[self.r]])
        
        x = np.array([[close[0]], [0.0]])
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        filtered_price_py[0] = x[0, 0]
        velocity_estimate_py[0] = x[1, 0]
        
        for k in range(1, n):
            z = close[k]
            x_pred = A @ x
            P_pred = A @ P @ A.T + Q
            y = z - (H @ x_pred)[0, 0]
            S = (H @ P_pred @ H.T + R)[0, 0]
            if abs(S) < 1e-15:
                S = 1e-15 if S >= 0 else -1e-15
            K = (P_pred @ H.T) / S
            x = x_pred + K * y
            I = np.eye(2)
            P = (I - K @ H) @ P_pred
            filtered_price_py[k] = x[0, 0]
            velocity_estimate_py[k] = x[1, 0]

        # 2. Run C++ Accelerated Implementation
        filter_instance = KalmanStateFilter(config={
            "q_price": self.qp,
            "q_velocity": self.qv,
            "r_observation": self.r
        })
        result_cpp = filter_instance.compute(self.df)
        
        # 3. Assert precision parity up to 12 decimal places
        np.testing.assert_array_almost_equal(
            result_cpp["kalman_price"].values,
            filtered_price_py,
            decimal=12,
            err_msg="Kalman filtered price does not match between Python and C++ at 10^-12 precision"
        )
        
        np.testing.assert_array_almost_equal(
            result_cpp["kalman_velocity"].values,
            velocity_estimate_py,
            decimal=12,
            err_msg="Kalman velocity estimate does not match between Python and C++ at 10^-12 precision"
        )

    def test_execution_speedup(self) -> None:
        """
        Verify that the compiled C++ Kalman filter executes at least 20x faster
        than the raw Python loop version.
        """
        # Run Pure Python Reference multiple times to get average duration
        close = self.df["close"].values
        n = len(close)
        
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.array([[self.qp, 0.0], [0.0, self.qv]])
        R = np.array([[self.r]])
        
        start_py = time.perf_counter()
        
        for _ in range(5):  # Run 5 times for a stable mean
            filtered_price_py = np.zeros(n)
            velocity_estimate_py = np.zeros(n)
            x = np.array([[close[0]], [0.0]])
            P = np.array([[1.0, 0.0], [0.0, 1.0]])
            filtered_price_py[0] = x[0, 0]
            velocity_estimate_py[0] = x[1, 0]
            
            for k in range(1, n):
                z = close[k]
                x_pred = A @ x
                P_pred = A @ P @ A.T + Q
                y = z - (H @ x_pred)[0, 0]
                S = (H @ P_pred @ H.T + R)[0, 0]
                if abs(S) < 1e-15:
                    S = 1e-15 if S >= 0 else -1e-15
                K = (P_pred @ H.T) / S
                x = x_pred + K * y
                I = np.eye(2)
                P = (I - K @ H) @ P_pred
                filtered_price_py[k] = x[0, 0]
                velocity_estimate_py[k] = x[1, 0]
                
        end_py = time.perf_counter()
        avg_py_duration = (end_py - start_py) / 5.0
        
        # Run C++ Accelerated multiple times
        filter_instance = KalmanStateFilter(config={
            "q_price": self.qp,
            "q_velocity": self.qv,
            "r_observation": self.r
        })
        
        # Warmup
        filter_instance.compute(self.df)
        
        start_cpp = time.perf_counter()
        for _ in range(5):
            filter_instance.compute(self.df)
        end_cpp = time.perf_counter()
        avg_cpp_duration = (end_cpp - start_cpp) / 5.0
        
        speedup = avg_py_duration / avg_cpp_duration
        print(f"\nExecution Profile (n={self.n}):")
        print(f"  - Pure Python Avg Duration: {avg_py_duration * 1000:.3f} ms")
        print(f"  - C++ Optimized Avg Duration: {avg_cpp_duration * 1000:.3f} ms")
        print(f"  - Achieved Speedup Factor: {speedup:.1f}x")
        
        # Assert minimum speedup of 20x
        self.assertGreaterEqual(
            speedup, 20.0,
            f"C++ implementation must run at least 20x faster than pure Python (got {speedup:.1f}x)"
        )


if __name__ == "__main__":
    unittest.main()
