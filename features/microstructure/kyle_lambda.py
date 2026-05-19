import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class KylesLambda(BaseFeature):
    """
    Kyle's Lambda (market impact).
    Measures illiquidity via rolling OLS regression of price change on net order flow:
    ΔP_t = lambda * (Signed Volume_t) + epsilon
    Higher lambda indicates that trades have a larger impact on prices (less liquid market).
    """

    def __init__(self, name: str = "kyle_lambda", config: Any = None) -> None:
        super().__init__(name, config)
        self.rolling_window = 100 if not config else config.get("rolling_window", 100)

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
        Computes Kyle's Lambda using a rolling covariance/variance OLS estimation.
        """
        self.validate(df)
        
        # 1. Determine price
        if "close" in df.columns:
            price = df["close"]
        else:
            price = (df["ask"] + df["bid"]) / 2.0
            
        price_diff = price.diff().fillna(0.0)
        
        # Lee-Ready style sign fill for order direction
        signs = np.sign(price_diff)
        signs_series = pd.Series(signs, index=df.index)
        signs_series = signs_series.replace(0.0, np.nan).ffill().fillna(1.0)
        
        # Signed volume
        signed_vol = signs_series * df["volume"]
        
        # 2. Compute rolling covariance and variance
        rolling_w = kwargs.get("rolling_window", self.rolling_window)
        
        cov = price_diff.rolling(window=rolling_w, min_periods=3).cov(signed_vol)
        var = signed_vol.rolling(window=rolling_w, min_periods=3).var()
        
        # Clean up zero variance values to avoid division by zero
        var_clean = np.where(var <= 1e-12, np.nan, var)
        
        kyle_lambda = cov / var_clean
        
        # Fill missing values and handle extreme anomalies
        kyle_lambda_series = pd.Series(kyle_lambda, index=df.index).ffill().fillna(0.0)
        
        return pd.DataFrame({f"{self.name}": kyle_lambda_series}, index=df.index)
