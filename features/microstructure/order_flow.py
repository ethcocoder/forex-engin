import pandas as pd
import numpy as np
from typing import Any, Union
from features.base_feature import BaseFeature


class OrderFlow(BaseFeature):
    """
    Computes order flow features from tick or bar data, including:
    - Signed trade flow using the Lee-Ready tick test algorithm
    - Signed volume (trade sign * volume)
    - Cumulative Volume Delta (CVD) over rolling windows
    """

    def __init__(self, name: str = "order_flow", config: Any = None) -> None:
        super().__init__(name, config)
        self.rolling_window = 100 if not config else config.get("rolling_window", 100)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required columns exist. Can operate on tick data ('bid', 'ask', 'volume')
        or transactional/ohlcv data ('close', 'volume').
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # We need volume
        if "volume" not in df.columns:
            raise ValueError("Missing required column: volume")
            
        # We need either ('bid', 'ask') or 'close'
        if not ("bid" in df.columns and "ask" in df.columns) and "close" not in df.columns:
            raise ValueError("DataFrame must contain either ('bid', 'ask') or 'close' columns.")
        
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes signed trade flow and rolling cumulative volume delta.
        """
        self.validate(df)
        
        # Determine the price series to use for the tick test
        if "close" in df.columns:
            price = df["close"]
        else:
            price = (df["ask"] + df["bid"]) / 2.0
            
        # Compute price differences
        price_diff = price.diff()
        
        # Compute signs (+1 for buy, -1 for sell, 0 for no change)
        signs = np.sign(price_diff.fillna(0.0))
        
        # Forward-fill zero signs (Lee-Ready algorithm: tick inherits previous tick's sign if price is unchanged)
        # Convert signs to series to use ffill()
        signs_series = pd.Series(signs, index=df.index)
        signs_series = signs_series.replace(0.0, np.nan).ffill().fillna(1.0) # start with +1 if first has no diff
        
        # Calculate signed volume
        signed_vol = signs_series * df["volume"]
        
        # Cumulative delta
        rolling_w = kwargs.get("rolling_window", self.rolling_window)
        cumulative_delta = signed_vol.rolling(window=rolling_w, min_periods=1).sum()
        
        result = pd.DataFrame(
            {
                f"{self.name}_sign": signs_series,
                f"{self.name}_signed_vol": signed_vol,
                f"{self.name}_cum_delta": cumulative_delta,
            },
            index=df.index
        )
        return result
