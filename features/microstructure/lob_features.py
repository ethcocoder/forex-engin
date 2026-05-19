import pandas as pd
import numpy as np
from typing import Any, List
from features.base_feature import BaseFeature


class LOBFeatures(BaseFeature):
    """
    Computes Limit Order Book (LOB) features, including:
    - Weighted mid-price: volume-weighted average of bid and ask prices.
    - Depth Imbalance: imbalance between buy and sell volumes at various levels.
    Supports up to 5 book levels, falling back gracefully to L0/L1 if higher levels are absent.
    """

    def __init__(self, name: str = "lob", config: Any = None) -> None:
        super().__init__(name, config)
        self.levels = 5 if not config else config.get("levels", 5)

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Ensures at least the base level (bid and ask) exists in the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
            
        # Basic validation: must have at least bid/ask or level 0 bid_p0/ask_p0
        has_l0_basic = "bid" in df.columns and "ask" in df.columns
        has_l0_explicit = "bid_p0" in df.columns and "ask_p0" in df.columns
        
        if not (has_l0_basic or has_l0_explicit):
            raise ValueError("DataFrame must contain at least Level 0 bid and ask price columns.")
            
        return True

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Computes weighted mid-price and depth imbalance for available levels.
        """
        self.validate(df)
        
        features_dict = {}
        
        # 1. Map columns for bid/ask prices and volumes at level 0
        if "bid_p0" in df.columns and "ask_p0" in df.columns:
            bid_p = [f"bid_p{i}" for i in range(self.levels) if f"bid_p{i}" in df.columns]
            ask_p = [f"ask_p{i}" for i in range(self.levels) if f"ask_p{i}" in df.columns]
            bid_v = [f"bid_v{i}" for i in range(self.levels) if f"bid_v{i}" in df.columns]
            ask_v = [f"ask_v{i}" for i in range(self.levels) if f"ask_v{i}" in df.columns]
            
            actual_levels = min(len(bid_p), len(ask_p), len(bid_v), len(ask_v))
        else:
            # Fallback when only basic bid/ask are present. We assume bid_v0 and ask_v0 might be present,
            # or if absent we look for "volume" or default to equal weight (0.5).
            df_temp = df.copy()
            if "bid_v0" not in df_temp.columns:
                df_temp["bid_v0"] = df_temp.get("volume", pd.Series(1.0, index=df_temp.index)) / 2.0
            if "ask_v0" not in df_temp.columns:
                df_temp["ask_v0"] = df_temp.get("volume", pd.Series(1.0, index=df_temp.index)) / 2.0
                
            df_temp["bid_p0"] = df_temp["bid"]
            df_temp["ask_p0"] = df_temp["ask"]
            
            bid_p = ["bid_p0"]
            ask_p = ["ask_p0"]
            bid_v = ["bid_v0"]
            ask_v = ["ask_v0"]
            actual_levels = 1
            df = df_temp
            
        # 2. Compute Weighted Mid-Price
        # Weighted Mid = (BidPrice * AskVolume + AskPrice * BidVolume) / (BidVolume + AskVolume)
        total_vol_l0 = df[bid_v[0]] + df[ask_v[0]]
        total_vol_l0_clean = np.where(total_vol_l0 <= 0, 1e-8, total_vol_l0)
        
        weighted_mid = (df[bid_p[0]] * df[ask_v[0]] + df[ask_p[0]] * df[bid_v[0]]) / total_vol_l0_clean
        features_dict[f"{self.name}_weighted_mid"] = weighted_mid
        
        # 3. Compute Depth Imbalance for each level
        # Imbalance = (BidVolume - AskVolume) / (BidVolume + AskVolume)
        for i in range(actual_levels):
            bp = bid_p[i]
            ap = ask_p[i]
            bv = bid_v[i]
            av = ask_v[i]
            
            sum_vol = df[bv] + df[av]
            sum_vol_clean = np.where(sum_vol <= 0, 1e-8, sum_vol)
            
            imbalance = (df[bv] - df[av]) / sum_vol_clean
            features_dict[f"{self.name}_imbalance_l{i}"] = imbalance
            
        # 4. Cumulative Depth Imbalance across all available levels
        if actual_levels > 1:
            total_bv = sum(df[bid_v[i]] for i in range(actual_levels))
            total_av = sum(df[ask_v[i]] for i in range(actual_levels))
            total_sum = total_bv + total_av
            total_sum_clean = np.where(total_sum <= 0, 1e-8, total_sum)
            
            cum_imbalance = (total_bv - total_av) / total_sum_clean
            features_dict[f"{self.name}_cum_imbalance"] = cum_imbalance
            
        return pd.DataFrame(features_dict, index=df.index)
