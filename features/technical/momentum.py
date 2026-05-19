import pandas as pd
import numpy as np
from typing import Any, List
from features.base_feature import BaseFeature


class MomentumFeatures(BaseFeature):
    """
    Computes momentum features:
    - Relative Strength Index (RSI) with Wilder's EWMA smoothing.
    - Moving Average Convergence Divergence (MACD) including Signal and Histogram.
    - Rate of Change (ROC) across multiple configurable lookbacks.
    """

    def __init__(self, name: str = "momentum", config: Any = None) -> None:
        super().__init__(name, config)
        self.rsi_period = 14 if not config else config.get("rsi_period", 14)
        self.macd_fast = 12 if not config else config.get("macd_fast", 12)
        self.macd_slow = 26 if not config else config.get("macd_slow", 26)
        self.macd_signal = 9 if not config else config.get("macd_signal", 9)
        self.roc_periods = [5, 10, 20, 50] if not config else config.get("roc_periods", [5, 10, 20, 50])

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required column 'close' exists.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if "close" not in df.columns:
            raise ValueError("Missing required column: close")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes RSI, MACD, and ROC features.
        """
        self.validate(df)
        
        close_series = df["close"]
        features_dict = {}
        
        # 1. Relative Strength Index (RSI)
        rsi_p = kwargs.get("rsi_period", self.rsi_period)
        delta = close_series.diff()
        
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        # Wilder's smoothing corresponds to EWMA with alpha = 1 / period
        avg_gain = pd.Series(gain, index=df.index).ewm(alpha=1.0 / rsi_p, min_periods=rsi_p).mean()
        avg_loss = pd.Series(loss, index=df.index).ewm(alpha=1.0 / rsi_p, min_periods=rsi_p).mean()
        
        # Clean average losses to prevent division by zero
        avg_loss_clean = np.where(avg_loss <= 0, 1e-8, avg_loss)
        rs = avg_gain / avg_loss_clean
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        features_dict[f"{self.name}_rsi"] = pd.Series(rsi, index=df.index).fillna(50.0)
        
        # 2. MACD
        fast_p = kwargs.get("macd_fast", self.macd_fast)
        slow_p = kwargs.get("macd_slow", self.macd_slow)
        sig_p = kwargs.get("macd_signal", self.macd_signal)
        
        ema_fast = close_series.ewm(span=fast_p, min_periods=fast_p).mean()
        ema_slow = close_series.ewm(span=slow_p, min_periods=slow_p).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig_p, min_periods=sig_p).mean()
        macd_hist = macd_line - signal_line
        
        features_dict[f"{self.name}_macd"] = macd_line.fillna(0.0)
        features_dict[f"{self.name}_macd_signal"] = signal_line.fillna(0.0)
        features_dict[f"{self.name}_macd_hist"] = macd_hist.fillna(0.0)
        
        # 3. Rate of Change (ROC)
        roc_p_list = kwargs.get("roc_periods", self.roc_periods)
        for roc_p in roc_p_list:
            shift_series = close_series.shift(roc_p)
            # Avoid division by zero
            shift_series_clean = np.where(shift_series <= 0, np.nan, shift_series)
            roc = ((close_series - shift_series_clean) / shift_series_clean) * 100.0
            features_dict[f"{self.name}_roc_{roc_p}"] = pd.Series(roc, index=df.index).fillna(0.0)
            
        return pd.DataFrame(features_dict, index=df.index)
