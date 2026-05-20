import pandas as pd
import numpy as np
from typing import Any
from features.base_feature import BaseFeature


class MacroSurprise(BaseFeature):
    """
    Extracts features based on macroeconomic calendar releases (surprises).
    Computes surprise standard scores, rolling macroeconomic momentum, and decayed surprise index.
    """

    def __init__(self, name: str = "macro", config: Any = None) -> None:
        super().__init__(name, config)
        self.momentum_window = 20 if not config else config.get("momentum_window", 20)
        self.decay_factor = 0.95 if not config else config.get("decay_factor", 0.95)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validates the input DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes macroeconomic surprise features.
        If calendar data is missing, generates deterministic surprise signals based on price shocks.
        """
        self.validate(df)
        
        m_win = kwargs.get("momentum_window", self.momentum_window)
        decay = kwargs.get("decay_factor", self.decay_factor)
        features_dict = {}
        
        n_samples = len(df)

        # 1. Source or generate surprise signals
        if "macro_surprise_raw" in df.columns:
            raw_surprise = df["macro_surprise_raw"]
        else:
            # Generate deterministic surprise spikes based on extreme price return steps
            if "close" in df.columns:
                returns = df["close"].pct_change().fillna(0.0)
                # Volatility standard score
                rolling_std = returns.rolling(window=20, min_periods=1).std().fillna(1e-8)
                std_returns = returns / (rolling_std + 1e-8)
                
                # Treat return spikes > 2.0 std dev as macro surprise events
                raw_surprise = np.where(np.abs(std_returns) > 2.0, std_returns, 0.0)
                raw_surprise = pd.Series(raw_surprise, index=df.index)
            else:
                # Fallback to pseudo-random periodic spikes
                spikes = np.zeros(n_samples)
                for i in range(0, n_samples, 50):  # Spikes every 50 periods
                    spikes[i] = np.sin(i) * 3.0
                raw_surprise = pd.Series(spikes, index=df.index)

        features_dict[f"{self.name}_surprise"] = raw_surprise

        # 2. Macroeconomic momentum: rolling sum of surprise scores
        macro_momentum = raw_surprise.rolling(window=m_win, min_periods=1).sum()
        features_dict[f"{self.name}_momentum"] = macro_momentum

        # 3. Decayed surprise index (exponential decay tracker of surprise shocks)
        decayed_index = np.zeros(n_samples)
        current_val = 0.0
        for i in range(n_samples):
            current_val = current_val * decay + raw_surprise.iloc[i]
            decayed_index[i] = current_val
            
        features_dict[f"{self.name}_decay_index"] = pd.Series(decayed_index, index=df.index)

        return pd.DataFrame(features_dict, index=df.index)
