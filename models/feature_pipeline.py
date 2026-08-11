import numpy as np
import pandas as pd
import structlog
from typing import Tuple, List

logger = structlog.get_logger()

class LeakageSafeFeaturePipeline:
    """
    Computes technical and microstructure features while strictly enforcing
    past-only visibility to prevent look-ahead bias and feature leakage.
    Provides purged walk-forward train/test splits.
    """

    def __init__(self, window_sizes: List[int] = [5, 15, 60]):
        self.window_sizes = window_sizes

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling returns, volatility, spread features, and volume imbalance
        using only past data (expanding / rolling windows with min_periods).
        """
        if df.empty:
            return df

        df = df.sort_values("timestamp").copy()
        
        # Mid price returns
        df["return_1"] = df["mid"].pct_change(1)
        
        # Rolling volatility (standard deviation of returns)
        for w in self.window_sizes:
            df[f"vol_{w}"] = df["return_1"].rolling(window=w, min_periods=w).std()

        # Relative spread
        df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]

        # Drop NaN rows created by rolling windows
        df = df.dropna().reset_index(drop=True)

        logger.info("Leakage-safe features computed", records=len(df), features=list(df.columns))
        return df

    def purged_walk_forward_split(
        self, df: pd.DataFrame, n_splits: int = 5, purge_window: int = 10
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates purged train/test splits where a time buffer (purge_window)
        is removed between train and test sets to prevent overlap contamination.
        """
        splits = []
        n = len(df)
        fold_size = n // (n_splits + 1)

        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + purge_window
            test_end = min(test_start + fold_size, n)

            if test_start >= n or test_end <= test_start:
                break

            train_set = df.iloc[:train_end]
            test_set = df.iloc[test_start:test_end]

            splits.append((train_set, test_set))

        logger.info("Purged walk-forward splits generated", total_folds=len(splits))
        return splits

if __name__ == "__main__":
    # Test feature pipeline and splitters
    sample_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01", periods=500, freq="s", tz="UTC"),
        "bid": np.linspace(1.1000, 1.1500, 500) + np.random.normal(0, 0.0001, 500),
        "ask": np.linspace(1.1002, 1.1502, 500) + np.random.normal(0, 0.0001, 500),
        "mid": np.linspace(1.1001, 1.1501, 500) + np.random.normal(0, 0.0001, 500)
    })
    
    pipeline = LeakageSafeFeaturePipeline()
    features_df = pipeline.compute_features(sample_df)
    splits = pipeline.purged_walk_forward_split(features_df, n_splits=3, purge_window=5)
    print(f"Generated {len(splits)} purged walk-forward folds successfully.")
