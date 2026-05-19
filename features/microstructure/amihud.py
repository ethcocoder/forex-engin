import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class AmihudIlliquidity(BaseFeature):
    """
    Amihud Illiquidity Ratio.
    Measures price impact of volume. Defined as the ratio of absolute price return to volume.
    Amihud = |Return| / Volume
    Higher values signify lower liquidity.
    """

    def __init__(self, name: str = "amihud", config: Any = None) -> None:
        super().__init__(name, config)
        self.rolling_window = 20 if not config else config.get("rolling_window", 20)

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
        Computes rolling Amihud illiquidity ratio.
        """
        self.validate(df)
        
        # 1. Determine price
        if "close" in df.columns:
            price = df["close"]
        else:
            price = (df["ask"] + df["bid"]) / 2.0
            
        # Compute return
        returns = price.pct_change().fillna(0.0)
        abs_returns = np.abs(returns)
        
        # Avoid division by zero
        volume_clean = np.where(df["volume"] <= 0.0, np.nan, df["volume"])
        
        # Amihud ratio
        ratio = abs_returns / volume_clean
        
        # 2. Smooth using rolling window
        rolling_w = kwargs.get("rolling_window", self.rolling_window)
        amihud_series = pd.Series(ratio, index=df.index).rolling(window=rolling_w, min_periods=1).mean().fillna(0.0)
        
        return pd.DataFrame({f"{self.name}": amihud_series}, index=df.index)
