"""Causal feature orchestration for offline research and live-compatible inference."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd
import structlog

from features.microstructure.amihud import AmihudIlliquidity
from features.microstructure.kyle_lambda import KylesLambda
from features.microstructure.lob_features import LOBFeatures
from features.microstructure.order_flow import OrderFlow
from features.microstructure.spread import BidAskSpread
from features.microstructure.vpin import VPIN
from features.technical.mean_reversion import MeanReversionFeatures
from features.technical.momentum import MomentumFeatures
from features.technical.trend import TrendFeatures
from features.technical.volatility import VolatilityEstimators
from features.technical.volume import VolumeFeatures
from features.wavelet.decomposition import WaveletDecomposition
from features.wavelet.kalman_filter import KalmanStateFilter

logger = structlog.get_logger()


class FeaturePipeline:
    """Compute an explicit causal feature set.

    The default pipeline includes only features calculated directly from OHLCV
    observations. Microstructure features are opt-in and require their actual
    source fields. Alternative-data features are deliberately excluded because
    they require separate provider contracts, release timestamps, and revision
    policies. Missing warm-up values are retained as NaN so downstream training
    can drop them causally; this class never backfills future observations.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        include_microstructure: bool = False,
        include_alternative: bool = False,
        forward_fill: bool = False,
    ) -> None:
        if include_alternative:
            raise ValueError(
                "Alternative-data features require a provider-specific, timestamped "
                "data contract and are not enabled by the causal core pipeline."
            )
        self.config = config
        self.include_microstructure = include_microstructure
        self.forward_fill = forward_fill
        self.extractors: Dict[str, Any] = {
            "volatility": VolatilityEstimators(config=config),
            "momentum": MomentumFeatures(config=config),
            "mean_reversion": MeanReversionFeatures(config=config),
            "trend": TrendFeatures(config=config),
            "volume": VolumeFeatures(config=config),
            "wavelet": WaveletDecomposition(config=config),
            "kalman": KalmanStateFilter(config=config),
        }
        if include_microstructure:
            self.extractors.update(
                {
                    "spread": BidAskSpread(name="spread", config=config),
                    "order_flow": OrderFlow(config=config),
                    "lob": LOBFeatures(config=config),
                    "vpin": VPIN(config=config),
                    "kyle_lambda": KylesLambda(config=config),
                    "amihud": AmihudIlliquidity(config=config),
                }
            )
        self.min_lookback = 256

    def _validate_input(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Feature input must be a pandas DataFrame.")
        required = {"open", "high", "low", "close", "volume"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing OHLCV input columns: {missing}.")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Feature input must use a timestamp DatetimeIndex.")
        if frame.index.tz is None:
            raise ValueError("Feature timestamps must be timezone-aware.")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("Feature timestamps must be unique and chronological.")
        if self.include_microstructure and not {"bid", "ask"}.issubset(frame.columns):
            raise ValueError(
                "Microstructure features require real bid and ask columns; synthetic quotes are forbidden."
            )

    def compute_all(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute all configured features without non-causal backfilling."""
        self._validate_input(frame)
        started = time.perf_counter()
        feature_frames: List[pd.DataFrame] = []
        for name, extractor in self.extractors.items():
            extractor_started = time.perf_counter()
            feature_frame = extractor.compute(frame, **kwargs)
            if not feature_frame.index.equals(frame.index):
                raise ValueError(
                    f"Extractor '{name}' returned a misaligned timestamp index."
                )
            if len(feature_frame) != len(frame):
                raise ValueError(
                    f"Extractor '{name}' returned {len(feature_frame)} rows; expected {len(frame)}."
                )
            if feature_frame.columns.duplicated().any():
                raise ValueError(f"Extractor '{name}' returned duplicate columns.")
            feature_frames.append(feature_frame)
            logger.debug(
                "Computed causal feature extractor",
                extractor=name,
                elapsed_ms=(time.perf_counter() - extractor_started) * 1000.0,
            )

        features = pd.concat(feature_frames, axis=1)
        if features.columns.duplicated().any():
            raise ValueError("Feature pipeline generated duplicate feature names.")
        if self.forward_fill:
            # Forward fill is causal but must be an explicit research decision.
            features = features.ffill()

        logger.info(
            "Completed causal feature pipeline",
            total_features=features.shape[1],
            rows=len(features),
            include_microstructure=self.include_microstructure,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return features
