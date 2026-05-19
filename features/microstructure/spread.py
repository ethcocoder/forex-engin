import pandas as pd
import numpy as np
from typing import Any, Union
from features.base_feature import BaseFeature


class BidAskSpread(BaseFeature):
    """
    Computes bid-ask spread features including:
    - Absolute spread: ask - bid
    - Percentage spread: (ask - bid) / mid
    - EWM spread: Exponentially Weighted Moving Average of absolute spread
    """

    def __init__(self, name: str = "bid_ask_spread", config: Any = None) -> None:
        super().__init__(name, config)
        self.span = 20 if not config else config.get("ewm_span", 20)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required columns 'bid' and 'ask' exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_cols = ["bid", "ask"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Compute absolute, percentage, and EWM spread features.
        """
        self.validate(df)
        
        # Calculate mid-price
        mid = (df["ask"] + df["bid"]) / 2.0
        
        # Avoid division by zero
        mid_clean = np.where(mid <= 0, np.nan, mid)
        
        spread = df["ask"] - df["bid"]
        spread_pct = spread / mid_clean
        
        # Exponential moving average of spread
        ewm_span = kwargs.get("ewm_span", self.span)
        spread_ewm = spread.ewm(span=ewm_span, min_periods=1).mean()
        
        result = pd.DataFrame(
            {
                f"{self.name}_abs": spread,
                f"{self.name}_pct": spread_pct,
                f"{self.name}_ewm": spread_ewm,
            },
            index=df.index
        )
        return result
