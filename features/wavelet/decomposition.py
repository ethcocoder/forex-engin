import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature

try:
    import pywt
except ImportError:
    pywt = None


class WaveletDecomposition(BaseFeature):
    """
    Wavelet frequency decomposition feature using PyWavelets.
    Applies Daubechies 4 (db4) Discrete Wavelet Transform (DWT) at 5 decomposition levels.
    Separates market price into low-frequency trend, mid-frequency cycles, and high-frequency noise.
    Reconstructs denoised signals via inverse transform with thresholding.
    """

    def __init__(self, name: str = "wavelet", config: Any = None) -> None:
        super().__init__(name, config)
        self.wavelet = "db4" if not config else config.get("wavelet_name", "db4")
        self.level = 5 if not config else config.get("decomposition_level", 5)
        # We need a minimum lookback of 256 for a stable 5-level db4 transform
        self.rolling_window = 256 if not config else config.get("rolling_window", 256)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required column 'close' exists and PyWavelets is installed.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if "close" not in df.columns:
            raise ValueError("Missing required column: close")
        if pywt is None:
            raise ImportError("PyWavelets ('pywt') library is not installed. Please install it.")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes rolling wavelet-denoised trend, cycles, and noise signals.
        """
        self.validate(df)
        
        # Ensure we have a writable numpy array to avoid "buffer source array is read-only" error in pywt
        close = df["close"].to_numpy(copy=True)
        n = len(df)
        
        window = kwargs.get("rolling_window", self.rolling_window)
        wav = kwargs.get("wavelet_name", self.wavelet)
        lvl = kwargs.get("decomposition_level", self.level)
        
        trend_arr = np.zeros(n)
        cycle_arr = np.zeros(n)
        noise_arr = np.zeros(n)
        
        # Pre-fill initial window with close prices or zeroes
        trend_arr[:window] = close[:window]
        
        # For each tick, decompose the rolling window of close prices to avoid lookahead bias
        for i in range(window - 1, n):
            window_data = close[i - window + 1 : i + 1]
            
            # Perform Discrete Wavelet Transform
            # coeff = [cA_L, cD_L, cD_L-1, ..., cD_1]
            coeff = pywt.wavedec(window_data, wav, level=lvl, mode="symmetric")
            
            # cA_L is the lowest-frequency approximation (trend)
            # Detail coefficients capture different frequencies:
            # cD_L ... cD_3 are mid-frequency cycles
            # cD_2, cD_1 are high-frequency noise
            
            # 1. Denoised Trend: Set all detail coefficients to 0
            coeff_trend = [coeff[0]] + [np.zeros_like(c) for c in coeff[1:]]
            trend_reconstructed = pywt.waverec(coeff_trend, wav, mode="symmetric")
            # Account for potential length mismatch in reconstruction
            trend_arr[i] = trend_reconstructed[-1]
            
            # 2. Cycles: Keep only mid-frequency details (e.g. levels 3, 4, 5)
            coeff_cycle = [np.zeros_like(coeff[0])]
            for j in range(1, len(coeff)):
                # If level is 1 or 2 (which correspond to index -1 and -2), set to 0. Else keep.
                if j >= len(coeff) - 2:
                    coeff_cycle.append(np.zeros_like(coeff[j]))
                else:
                    coeff_cycle.append(coeff[j])
            cycle_reconstructed = pywt.waverec(coeff_cycle, wav, mode="symmetric")
            cycle_arr[i] = cycle_reconstructed[-1]
            
            # 3. Noise: Keep only high-frequency details (levels 1 and 2)
            coeff_noise = [np.zeros_like(coeff[0])]
            for j in range(1, len(coeff)):
                if j >= len(coeff) - 2:
                    coeff_noise.append(coeff[j])
                else:
                    coeff_noise.append(np.zeros_like(coeff[j]))
            noise_reconstructed = pywt.waverec(coeff_noise, wav, mode="symmetric")
            noise_arr[i] = noise_reconstructed[-1]
            
        result = pd.DataFrame(
            {
                f"{self.name}_trend": pd.Series(trend_arr, index=df.index),
                f"{self.name}_cycle": pd.Series(cycle_arr, index=df.index),
                f"{self.name}_noise": pd.Series(noise_arr, index=df.index),
            },
            index=df.index
        )
        return result
