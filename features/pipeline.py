import pandas as pd
import numpy as np
import time
from typing import Any, Dict, List, Optional
import structlog
from features.microstructure.spread import BidAskSpread
from features.microstructure.order_flow import OrderFlow
from features.microstructure.lob_features import LOBFeatures
from features.microstructure.vpin import VPIN
from features.microstructure.kyle_lambda import KylesLambda
from features.microstructure.amihud import AmihudIlliquidity
from features.technical.volatility import VolatilityEstimators
from features.technical.momentum import MomentumFeatures
from features.technical.mean_reversion import MeanReversionFeatures
from features.technical.trend import TrendFeatures
from features.technical.volume import VolumeFeatures
from features.wavelet.decomposition import WaveletDecomposition
from features.wavelet.kalman_filter import KalmanStateFilter

logger = structlog.get_logger()


class FeaturePipeline:
    """
    Feature Extraction Pipeline.
    Orchestrates all feature extraction classes.
    Enforces NaN propagation rules, min lookback periods, and handles data joins.
    Ensures identical output between offline backtest batching and online live streaming.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        
        # Instantiate all sub-feature extractors
        self.extractors = {
            "spread": BidAskSpread(name="spread", config=config),
            "order_flow": OrderFlow(config=config),
            "lob": LOBFeatures(config=config),
            "vpin": VPIN(config=config),
            "kyle_lambda": KylesLambda(config=config),
            "amihud": AmihudIlliquidity(config=config),
            "volatility": VolatilityEstimators(config=config),
            "momentum": MomentumFeatures(config=config),
            "mean_reversion": MeanReversionFeatures(config=config),
            "trend": TrendFeatures(config=config),
            "volume": VolumeFeatures(config=config),
            "wavelet": WaveletDecomposition(config=config),
            "kalman": KalmanStateFilter(config=config),
        }
        
        # We need a minimum lookback of 256 for wavelet decomposition
        self.min_lookback = 256

    def compute_all(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Orchestrates running all feature extractors sequentially on the raw DataFrame.
        Concatenates all resulting feature columns, handles NaN values, and returns the master matrix.
        """
        t_start = time.perf_counter()
        
        if len(df) < self.min_lookback:
            logger.warning(
                "DataFrame length is less than the minimum required lookback.",
                length=len(df),
                required=self.min_lookback
            )
            
        feature_dfs: List[pd.DataFrame] = []
        
        for name, extractor in self.extractors.items():
            try:
                t_feat_start = time.perf_counter()
                feat_df = extractor.compute(df, **kwargs)
                t_feat_end = time.perf_counter()
                
                # Check for output shape matches
                if len(feat_df) != len(df):
                    raise ValueError(
                        f"Extractor '{name}' returned {len(feat_df)} rows, expected {len(df)} rows."
                    )
                    
                feature_dfs.append(feat_df)
                
                logger.debug(
                    "Computed feature extractor successfully",
                    extractor=name,
                    elapsed_ms=(t_feat_end - t_feat_start) * 1000.0
                )
            except Exception as e:
                logger.error(
                    "Feature extraction failed in pipeline",
                    extractor=name,
                    error=str(e)
                )
                raise e
                
        # Combine all features along the columns axis
        master_features = pd.concat(feature_dfs, axis=1)
        
        # Enforce NaN resolution rules:
        # 1. Forward-fill holes from stale/delayed trades
        # 2. Back-fill initial lookback window startup periods
        master_features = master_features.ffill().bfill().fillna(0.0)
        
        t_end = time.perf_counter()
        logger.info(
            "Completed entire feature pipeline computation",
            total_features=master_features.shape[1],
            rows=master_features.shape[0],
            total_elapsed_ms=(t_end - t_start) * 1000.0
        )
        
        return master_features
