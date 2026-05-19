import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class VolumeFeatures(BaseFeature):
    """
    Computes volume-based features:
    - Volume Delta: change in volume.
    - Volume-Weighted Average Price (VWAP) over a rolling window.
    - Rolling Volume Profile Point of Control (VPOC): price level with highest accumulated volume.
    """

    def __init__(self, name: str = "volume", config: Any = None) -> None:
        super().__init__(name, config)
        self.rolling_window = 50 if not config else config.get("rolling_window", 50)
        self.vpoc_bins = 10 if not config else config.get("vpoc_bins", 10)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required columns 'close' and 'volume' exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if "close" not in df.columns:
            raise ValueError("Missing required column: close")
        if "volume" not in df.columns:
            raise ValueError("Missing required column: volume")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes volume features including VWAP and rolling VPOC.
        """
        self.validate(df)
        
        close = df["close"]
        volume = df["volume"]
        window = kwargs.get("rolling_window", self.rolling_window)
        bins_k = kwargs.get("vpoc_bins", self.vpoc_bins)
        
        features_dict = {}
        
        # 1. Volume Delta
        features_dict[f"{self.name}_delta"] = volume.diff().fillna(0.0)
        
        # 2. Rolling VWAP
        # VWAP = sum(Price * Volume) / sum(Volume)
        pv = close * volume
        sum_pv = pv.rolling(window=window, min_periods=1).sum()
        sum_v = volume.rolling(window=window, min_periods=1).sum()
        
        sum_v_clean = np.where(sum_v <= 1e-8, 1e-8, sum_v)
        vwap = sum_pv / sum_v_clean
        
        features_dict[f"{self.name}_vwap"] = vwap.fillna(close)
        
        # 3. Rolling Volume Profile Point of Control (VPOC)
        # For each rolling window of size W, we bin the prices into K bins,
        # find the bin with the maximum volume, and return its center price.
        # To make it fast, we can write a row-by-row numpy loop for the rolling VPOC.
        prices_arr = close.values
        vols_arr = volume.values
        n = len(df)
        
        vpoc_arr = np.zeros(n)
        
        # Pre-fill initial values with close price
        vpoc_arr[:window] = prices_arr[:window]
        
        # Fast rolling window calculation
        for i in range(window, n):
            win_prices = prices_arr[i - window + 1 : i + 1]
            win_vols = vols_arr[i - window + 1 : i + 1]
            
            p_min = win_prices.min()
            p_max = win_prices.max()
            
            if p_max - p_min < 1e-8:
                vpoc_arr[i] = p_min
                continue
                
            # Create price bins
            bin_edges = np.linspace(p_min, p_max, bins_k + 1)
            
            # Digitized bin indexes for prices
            bin_indices = np.digitize(win_prices, bin_edges) - 1
            # Clip bin indices to [0, bins_k - 1] to handle the max boundary price
            bin_indices = np.clip(bin_indices, 0, bins_k - 1)
            
            # Accumulate volume per bin
            bin_vols = np.zeros(bins_k)
            np.add.at(bin_vols, bin_indices, win_vols)
            
            # Find the bin with the maximum volume
            max_bin_idx = np.argmax(bin_vols)
            
            # Compute center price of that bin
            bin_center = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0
            vpoc_arr[i] = bin_center
            
        features_dict[f"{self.name}_vpoc"] = pd.Series(vpoc_arr, index=df.index)
        
        return pd.DataFrame(features_dict, index=df.index)
