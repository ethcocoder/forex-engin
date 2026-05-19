import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class VPIN(BaseFeature):
    """
    Volume-Synchronized Probability of Toxicity (VPIN).
    Measures the imbalance of buy/sell volume in constant-volume buckets.
    High VPIN indicates high order flow toxicity/informed trading.
    """

    def __init__(self, name: str = "vpin", config: Any = None) -> None:
        super().__init__(name, config)
        self.bucket_volume = 1000.0 if not config else config.get("bucket_volume", 1000.0)
        self.rolling_buckets = 50 if not config else config.get("rolling_buckets", 50)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required columns exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if "volume" not in df.columns:
            raise ValueError("Missing required column: volume")
        if not ("bid" in df.columns and "ask" in df.columns) and "close" not in df.columns:
            raise ValueError("DataFrame must contain either ('bid', 'ask') or 'close' columns.")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes VPIN using a vectorized volume-bucketing algorithm.
        """
        self.validate(df)
        
        # 1. Determine prices and signs
        if "close" in df.columns:
            price = df["close"]
        else:
            price = (df["ask"] + df["bid"]) / 2.0
            
        price_diff = price.diff().fillna(0.0)
        signs = np.sign(price_diff)
        
        # Lee-Ready style sign fill
        signs_series = pd.Series(signs, index=df.index)
        signs_series = signs_series.replace(0.0, np.nan).ffill().fillna(1.0)
        
        # Calculate buy/sell volume per row
        volume = df["volume"].values
        signed_vol = (signs_series.values * volume)
        
        buy_vol = np.where(signed_vol > 0, volume, 0.0)
        sell_vol = np.where(signed_vol < 0, volume, 0.0)
        
        # 2. Vectorized Volume Bucketing
        # We compute the cumulative volume to align rows with buckets
        cum_vol = np.cumsum(volume)
        bucket_vol = kwargs.get("bucket_volume", self.bucket_volume)
        
        # bucket_idx indicates which bucket each row belongs to
        bucket_idx = (cum_vol // bucket_vol).astype(int)
        
        # Group buy and sell volumes by bucket index
        df_buckets = pd.DataFrame({
            "buy_v": buy_vol,
            "sell_v": sell_vol,
            "bucket": bucket_idx
        })
        
        bucket_totals = df_buckets.groupby("bucket").sum()
        
        # Compute absolute imbalance per bucket: |V_B - V_S|
        abs_imbalance = np.abs(bucket_totals["buy_v"] - bucket_totals["sell_v"])
        
        # Compute VPIN = rolling sum of absolute imbalances / (N * bucket_volume)
        roll_b = kwargs.get("rolling_buckets", self.rolling_buckets)
        rolling_imbalance_sum = abs_imbalance.rolling(window=roll_b, min_periods=1).sum()
        
        vpin_by_bucket = rolling_imbalance_sum / (roll_b * bucket_vol)
        
        # Map VPIN back to the original time index
        # To guarantee production-backtest equivalence and avoid lookahead bias,
        # we assign the VPIN of the last fully completed bucket (bucket_idx - 1).
        vpin_series = pd.Series(bucket_idx - 1, index=df.index).map(vpin_by_bucket).fillna(0.0)
        
        # Limit VPIN to [0, 1] range
        vpin_series = np.clip(vpin_series, 0.0, 1.0)
        
        return pd.DataFrame({f"{self.name}": vpin_series}, index=df.index)
