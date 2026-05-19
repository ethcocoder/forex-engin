import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class VolatilityEstimators(BaseFeature):
    """
    Computes rolling realized volatility estimators, including:
    - Close-to-Close Volatility
    - Parkinson Volatility (1980)
    - Garman-Klass Volatility (1980)
    - Rogers-Satchell Volatility (1991)
    - Yang-Zhang Volatility (2000)
    """

    def __init__(self, name: str = "volatility", config: Any = None) -> None:
        super().__init__(name, config)
        self.rolling_window = 20 if not config else config.get("rolling_window", 20)
        self.annualize_factor = 252 if not config else config.get("annualize_factor", 252)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates that required columns 'open', 'high', 'low', 'close' exist.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for OHLC volatility calculation: {missing_cols}")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes various volatility estimators over a rolling window.
        """
        self.validate(df)
        
        window = kwargs.get("rolling_window", self.rolling_window)
        ann = kwargs.get("annualize_factor", self.annualize_factor)
        
        open_p = df["open"].values
        high_p = df["high"].values
        low_p = df["low"].values
        close_p = df["close"].values
        
        # Avoid zero or negative values in prices
        open_p = np.where(open_p <= 0, 1e-8, open_p)
        high_p = np.where(high_p <= 0, 1e-8, high_p)
        low_p = np.where(low_p <= 0, 1e-8, low_p)
        close_p = np.where(close_p <= 0, 1e-8, close_p)
        
        # 1. Close-to-Close realized volatility
        # Note: log returns are ln(C_t / C_t-1)
        close_series = pd.Series(close_p, index=df.index)
        log_ret = np.log(close_series / close_series.shift(1)).fillna(0.0)
        vol_cc = log_ret.rolling(window=window).std() * np.sqrt(ann)
        
        # 2. Parkinson realized volatility
        # Parkinson = sqrt( N / (4 * ln(2) * n) * sum( ln(H / L)^2 ) )
        hl_ratio = np.log(high_p / low_p)
        park_element = (hl_ratio ** 2) / (4.0 * np.log(2.0))
        vol_park = pd.Series(park_element, index=df.index).rolling(window=window).mean()
        vol_park = np.sqrt(vol_park * ann)
        
        # 3. Garman-Klass realized volatility
        # GK = sqrt( N / n * sum( 0.5 * ln(H / L)^2 - (2 * ln(2) - 1) * ln(C / O)^2 ) )
        gk_element = 0.5 * (hl_ratio ** 2) - (2.0 * np.log(2.0) - 1.0) * (np.log(close_p / open_p) ** 2)
        vol_gk = pd.Series(gk_element, index=df.index).rolling(window=window).mean()
        vol_gk = np.sqrt(np.clip(vol_gk, 0.0, None) * ann)
        
        # 4. Rogers-Satchell realized volatility
        # RS = sqrt( N / n * sum( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) ) )
        rs_element = np.log(high_p / close_p) * np.log(high_p / open_p) + np.log(low_p / close_p) * np.log(low_p / open_p)
        vol_rs = pd.Series(rs_element, index=df.index).rolling(window=window).mean()
        vol_rs = np.sqrt(np.clip(vol_rs, 0.0, None) * ann)
        
        # 5. Yang-Zhang realized volatility
        # YZ = overnight_var + k * open_to_close_var + (1-k) * rs_var
        # overnight = ln(O_t / C_t-1)
        close_shift = close_series.shift(1).values
        close_shift = np.where(np.isnan(close_shift) | (close_shift <= 0), open_p, close_shift)
        
        log_overnight = np.log(open_p / close_shift)
        log_open_close = np.log(close_p / open_p)
        
        # Rolling variance of overnight returns and open_close returns
        var_overnight = pd.Series(log_overnight, index=df.index).rolling(window=window).var()
        var_open_close = pd.Series(log_open_close, index=df.index).rolling(window=window).var()
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1)) if window > 1 else 0.34
        
        # Rogers-Satchell variance (which is vol_rs^2 without annualization factor)
        rs_var = pd.Series(rs_element, index=df.index).rolling(window=window).mean()
        
        yz_var = var_overnight + k * var_open_close + (1.0 - k) * rs_var
        vol_yz = np.sqrt(np.clip(yz_var, 0.0, None) * ann)
        
        result = pd.DataFrame(
            {
                f"{self.name}_cc": vol_cc.fillna(0.0),
                f"{self.name}_parkinson": vol_park.fillna(0.0),
                f"{self.name}_garman_klass": vol_gk.fillna(0.0),
                f"{self.name}_rogers_satchell": vol_rs.fillna(0.0),
                f"{self.name}_yang_zhang": vol_yz.fillna(0.0),
            },
            index=df.index
        )
        return result
