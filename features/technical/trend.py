import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class TrendFeatures(BaseFeature):
    """
    Computes trend and trend-strength features:
    - Average Directional Index (ADX) with DI+ and DI- lines.
    - Rolling Linear Regression Slope: rate of price change.
    - Rolling R-squared (Coefficient of Determination): trend strength.
    """

    def __init__(self, name: str = "trend", config: Any = None) -> None:
        super().__init__(name, config)
        self.adx_period = 14 if not config else config.get("adx_period", 14)
        self.slope_window = 20 if not config else config.get("slope_window", 20)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required columns 'high', 'low', 'close' exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_cols = ["high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for trend features: {missing_cols}")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes ADX, trend slope, and R-squared values.
        """
        self.validate(df)
        
        high = df["high"]
        low = df["low"]
        close = df["close"]
        features_dict = {}
        
        adx_p = kwargs.get("adx_period", self.adx_period)
        slope_w = kwargs.get("slope_window", self.slope_window)
        
        # 1. Compute DMI & ADX
        high_diff = high.diff()
        low_diff = (-low.diff())
        
        # Directional Movement (+DM and -DM)
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        
        # True Range (TR)
        close_shift = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - close_shift)
        tr3 = np.abs(low - close_shift)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(1e-8)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1 / period)
        tr_smooth = tr.ewm(alpha=1.0 / adx_p, min_periods=adx_p).mean()
        pos_dm_smooth = pd.Series(pos_dm, index=df.index).ewm(alpha=1.0 / adx_p, min_periods=adx_p).mean()
        neg_dm_smooth = pd.Series(neg_dm, index=df.index).ewm(alpha=1.0 / adx_p, min_periods=adx_p).mean()
        
        # Avoid division by zero
        tr_smooth_clean = np.where(tr_smooth <= 1e-8, 1e-8, tr_smooth)
        
        plus_di = 100.0 * (pos_dm_smooth / tr_smooth_clean)
        minus_di = 100.0 * (neg_dm_smooth / tr_smooth_clean)
        
        # DX = 100 * |+DI - -DI| / (|+DI + -DI|)
        di_sum = plus_di + minus_di
        di_sum_clean = np.where(di_sum <= 1e-8, 1e-8, di_sum)
        dx = 100.0 * np.abs(plus_di - minus_di) / di_sum_clean
        
        # ADX is the smoothed version of DX
        adx = dx.ewm(alpha=1.0 / adx_p, min_periods=adx_p).mean()
        
        features_dict[f"{self.name}_adx"] = adx.fillna(0.0)
        features_dict[f"{self.name}_plus_di"] = plus_di.fillna(0.0)
        features_dict[f"{self.name}_minus_di"] = minus_di.fillna(0.0)
        
        # 2. Rolling Linear Regression Slope & R-squared
        # X is just a range from 1 to slope_w
        x = np.arange(1, slope_w + 1)
        x_mean = np.mean(x)
        x_var = np.var(x)
        
        # Rolling covariance of X and Y
        # For a vectorized rolling OLS in pandas, we can compute rolling covariance:
        # Cov(X, Y) = mean(X * Y) - mean(X) * mean(Y)
        # We can construct rolling terms:
        # roll_mean_y = close.rolling(window=slope_w).mean()
        # For covariance: sum_i (x_i * y_{t - slope_w + i}) / slope_w
        # We can do this efficiently using pandas rolling with a custom window apply,
        # or by combining rolling means of lagged close.
        # Custom rolling functions in pandas can be slow. A beautiful vectorized linear algebra trick:
        # Since X is fixed, the linear regression slope is a linear filter (convolution)!
        # Weights for the convolution filter are: w_i = (x_i - x_mean) / (slope_w * x_var)
        weights = (x - x_mean) / (slope_w * x_var)
        
        # Now, we simply apply rolling 1D convolution (convolution is just linear filter):
        slope = close.rolling(window=slope_w).apply(lambda y: np.dot(y, weights), raw=True)
        
        # Trend strength (R-squared) is the squared correlation coefficient
        # Corr(X, Y) = Cov(X, Y) / (Std(X) * Std(Y))
        # Std(X) = sqrt(x_var)
        # Std(Y) = close.rolling(window=slope_w).std()
        std_x = np.sqrt(x_var)
        std_y = close.rolling(window=slope_w).std()
        std_y_clean = np.where(std_y <= 1e-8, 1e-8, std_y)
        
        # Cov(X, Y) = slope * x_var
        cov_xy = slope * x_var
        corr = cov_xy / (std_x * std_y_clean)
        r_squared = corr ** 2
        
        features_dict[f"{self.name}_slope"] = slope.fillna(0.0)
        features_dict[f"{self.name}_strength"] = pd.Series(r_squared, index=df.index).fillna(0.0)
        
        return pd.DataFrame(features_dict, index=df.index)
