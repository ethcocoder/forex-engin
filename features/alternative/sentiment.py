import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class SentimentFeatures(BaseFeature):
    """
    Extracts features based on news and social media sentiment feeds, including:
    - Sentiment score polarity (-1.0 to 1.0)
    - News/social media volume
    - Sentiment momentum & impact scores
    """

    def __init__(self, name: str = "sentiment", config: Any = None) -> None:
        super().__init__(name, config)
        self.ema_span = 12 if not config else config.get("ema_span", 12)
        self.volume_window = 20 if not config else config.get("volume_window", 20)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates input DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes sentiment-based features.
        Generates deterministic values if sentiment feeds are not directly available.
        """
        self.validate(df)
        
        span = kwargs.get("ema_span", self.ema_span)
        vol_win = kwargs.get("volume_window", self.volume_window)
        features_dict = {}
        n_samples = len(df)

        # 1. Source raw sentiment polarity (-1.0 to 1.0) and news volume
        if "sentiment_raw" in df.columns:
            sentiment_raw = df["sentiment_raw"]
        else:
            # Model raw sentiment as mean-reverting with momentum tied to returns
            if "close" in df.columns:
                returns = df["close"].pct_change().fillna(0.0)
                sentiment_raw = returns.rolling(window=5, min_periods=1).mean() * 10.0
                sentiment_raw = pd.Series(sentiment_raw, index=df.index).clip(-1.0, 1.0)
            else:
                np.random.seed(999)
                sentiment_raw = pd.Series(np.sin(np.arange(n_samples) * 0.05), index=df.index)

        if "news_volume_raw" in df.columns:
            news_volume = df["news_volume_raw"]
        else:
            # Model diurnal/periodic patterns in news volume
            np.random.seed(888)
            time_factor = np.abs(np.sin(np.arange(n_samples) * (np.pi / 24.0)))  # daily cycle
            news_volume = pd.Series(10.0 + time_factor * 90.0, index=df.index)

        # 2. Derive features
        # Sentiment EMA to smooth high frequency noise
        sentiment_ema = sentiment_raw.ewm(span=span, min_periods=1).mean()
        features_dict[f"{self.name}_score_ema"] = sentiment_ema

        # News volume momentum (current volume / rolling SMA of volume)
        volume_sma = news_volume.rolling(window=vol_win, min_periods=1).mean().fillna(1e-8)
        volume_momentum = news_volume / (volume_sma + 1e-8)
        features_dict[f"{self.name}_volume_momentum"] = volume_momentum

        # Sentiment impact score: sentiment polarity * news volume index
        impact = sentiment_raw * (news_volume / 100.0)
        features_dict[f"{self.name}_impact"] = impact

        # Sentiment divergence: current raw sentiment deviation from its EMA
        features_dict[f"{self.name}_divergence"] = sentiment_raw - sentiment_ema

        return pd.DataFrame(features_dict, index=df.index)
