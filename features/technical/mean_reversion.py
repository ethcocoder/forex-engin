import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class MeanReversionFeatures(BaseFeature):
    """
    Computes mean reversion and stationarity features:
    - Rolling Z-Score: distance of current close price from rolling mean in std units.
    - Fast Hurst Exponent: calculated via rolling scale-invariant variance of price differences.
    - Rolling Dickey-Fuller t-statistic: t-statistic of OLS regression Δy_t ~ y_{t-1} + c.
    """

    def __init__(self, name: str = "mean_reversion", config: Any = None) -> None:
        super().__init__(name, config)
        self.rolling_window = 50 if not config else config.get("rolling_window", 50)
        self.hurst_window = 100 if not config else config.get("hurst_window", 100)

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
        Computes rolling Z-Score, Hurst exponent, and Dickey-Fuller statistics.
        """
        self.validate(df)
        
        close_series = df["close"]
        features_dict = {}
        
        window = kwargs.get("rolling_window", self.rolling_window)
        
        # 1. Rolling Z-Score
        roll_mean = close_series.rolling(window=window, min_periods=window).mean()
        roll_std = close_series.rolling(window=window, min_periods=window).std()
        # Avoid division by zero
        roll_std_clean = np.where(roll_std <= 1e-8, 1e-8, roll_std)
        
        z_score = (close_series - roll_mean) / roll_std_clean
        features_dict[f"{self.name}_zscore"] = pd.Series(z_score, index=df.index).fillna(0.0)
        
        # 2. Fast Hurst Exponent
        # Hurst Exponent using scale-invariant variance of differences over multiple lags
        # Var(P(t + lag) - P(t)) ~ lag^(2H)
        # log(std(P(t+lag) - P(t))) = H * log(lag) + c
        h_win = kwargs.get("hurst_window", self.hurst_window)
        lags = [2, 4, 8, 16, 32]
        log_lags = np.log(lags)
        
        # We preallocate standard deviations for rolling window
        hurst_series = pd.Series(0.5, index=df.index)
        
        # To make it fast, we can approximate Hurst using pandas rolling operations
        # over the different lags.
        stds = []
        for lag in lags:
            diff = close_series.diff(lag)
            roll_std_lag = diff.rolling(window=h_win, min_periods=h_win).std()
            stds.append(roll_std_lag)
            
        # For each row, do linear regression log(std) ~ log(lag)
        # Vectorized linear regression of log(stds) against log_lags
        # slope = Cov(X, Y) / Var(X)
        log_lags_mean = np.mean(log_lags)
        log_lags_var = np.var(log_lags)
        
        stds_df = pd.concat(stds, axis=1) # (len(df), len(lags))
        log_stds = np.log(np.clip(stds_df.values, 1e-8, None))
        
        # E(XY) - E(X)E(Y)
        log_stds_mean = np.mean(log_stds, axis=1)
        # Broadcast log_lags to multiply each row
        xy_mean = np.mean(log_stds * log_lags, axis=1)
        cov_xy = xy_mean - log_stds_mean * log_lags_mean
        
        hurst_estimates = cov_xy / log_lags_var
        # Clip Hurst between 0 and 1
        hurst_estimates = np.clip(hurst_estimates, 0.0, 1.0)
        
        features_dict[f"{self.name}_hurst"] = pd.Series(hurst_estimates, index=df.index).fillna(0.5)
        
        # 3. Rolling Dickey-Fuller t-statistic (custom OLS: Δy_t ~ y_{t-1} + c)
        # y_lag = y_{t-1}
        # dy = y_t - y_{t-1}
        y = close_series.values
        dy = close_series.diff().fillna(0.0).values
        y_lag = close_series.shift(1).fillna(0.0).values
        
        df_stat = pd.Series(0.0, index=df.index)
        
        # We can implement a fast rolling linear regression for t-stat
        # Formula: beta = Cov(dy, y_lag) / Var(y_lag)
        # standard error of beta = s_e / sqrt( sum((y_lag - y_lag_bar)^2) )
        # where s_e^2 = sum(residual^2) / (n - 2)
        dy_series = pd.Series(dy, index=df.index)
        y_lag_series = pd.Series(y_lag, index=df.index)
        
        mean_y_lag = y_lag_series.rolling(window=window).mean()
        mean_dy = dy_series.rolling(window=window).mean()
        
        cov_y_dy = dy_series.rolling(window=window).cov(y_lag_series)
        var_y_lag = y_lag_series.rolling(window=window).var()
        
        beta = cov_y_dy / np.where(var_y_lag <= 1e-8, np.nan, var_y_lag)
        alpha = mean_dy - beta * mean_y_lag
        
        # Residual variance calculation
        # sum(res^2) = sum((dy - (beta * y_lag + alpha))^2)
        # We can compute this rolled over window
        # sum_res2 = sum(dy^2) - 2*beta*sum(dy*y_lag) - 2*alpha*sum(dy) + beta^2 * sum(y_lag^2) + 2*alpha*beta*sum(y_lag) + n*alpha^2
        # A simpler robust rolling regression using pandas is to compute the residual series:
        # But that has lookahead if we compute beta first, so we do it row-by-row or vectorized:
        # In a vectorized rolling framework:
        # residual(t) = dy(t) - (beta(t) * y_lag(t) + alpha(t))
        # Let's approximate the residual variance using the rolling variance of dy minus beta^2 * rolling variance of y_lag:
        # Var(residual) = Var(dy) - beta^2 * Var(y_lag)
        var_dy = dy_series.rolling(window=window).var()
        residual_var = np.clip(var_dy - (beta ** 2) * var_y_lag, 1e-8, None)
        
        # Standard error of beta: SE = sqrt( Var(residual) / ((N-2) * Var(y_lag) * (N-1)) )
        n_adj = window - 2 if window > 2 else 1
        se_beta = np.sqrt(residual_var / (n_adj * np.where(var_y_lag <= 1e-8, np.nan, var_y_lag)))
        
        df_t_stat = beta / np.where(se_beta <= 1e-8, 1e-8, se_beta)
        
        features_dict[f"{self.name}_df_stat"] = pd.Series(df_t_stat, index=df.index).fillna(0.0)
        
        return pd.DataFrame(features_dict, index=df.index)
