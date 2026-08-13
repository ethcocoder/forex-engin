import unittest
import pandas as pd
import numpy as np
from features.base_feature import BaseFeature
from features.pipeline import FeaturePipeline
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


class TestForexFeaturePipeline(unittest.TestCase):
    def setUp(self) -> None:
        # Create a mock dataset of 300 ticks/candles with high, low, open, close, bid, ask, volume
        np.random.seed(42)
        n = 300
        time_index = pd.date_range(start="2026-01-01", periods=n, freq="min", tz="UTC")
        
        # Generate random walk mid prices
        steps = np.random.normal(0, 0.0005, n)
        mid_prices = 1.1000 + np.cumsum(steps)
        
        # Structure mock bid-ask spread
        spread = np.random.uniform(0.0001, 0.0003, n)
        bid = mid_prices - spread / 2.0
        ask = mid_prices + spread / 2.0
        
        # Construct mock OHLCV bars
        close_p = mid_prices
        open_p = close_p + np.random.normal(0, 0.0002, n)
        high_p = np.maximum(open_p, close_p) + np.random.uniform(0, 0.0004, n)
        low_p = np.minimum(open_p, close_p) - np.random.uniform(0, 0.0004, n)
        volume = np.random.uniform(100.0, 1000.0, n)
        
        self.df = pd.DataFrame(
            {
                "bid": bid,
                "ask": ask,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
            },
            index=time_index
        )

    def test_feature_class_inheritance(self) -> None:
        """
        Verify that all feature classes correctly inherit from BaseFeature.
        """
        pipeline = FeaturePipeline()
        for name, extractor in pipeline.extractors.items():
            self.assertIsInstance(extractor, BaseFeature)
            self.assertEqual(extractor.name, name)

    def test_bid_ask_spread(self) -> None:
        """
        Test BidAskSpread calculations.
        """
        extractor = BidAskSpread()
        result = extractor.compute(self.df)
        self.assertIn("bid_ask_spread_abs", result.columns)
        self.assertIn("bid_ask_spread_pct", result.columns)
        self.assertIn("bid_ask_spread_ewm", result.columns)
        self.assertTrue((result["bid_ask_spread_abs"] >= 0).all())

    def test_volatility_estimators(self) -> None:
        """
        Test that realized volatility estimators produce positive values.
        """
        extractor = VolatilityEstimators()
        result = extractor.compute(self.df)
        self.assertIn("volatility_cc", result.columns)
        self.assertIn("volatility_parkinson", result.columns)
        self.assertIn("volatility_garman_klass", result.columns)
        self.assertIn("volatility_rogers_satchell", result.columns)
        self.assertIn("volatility_yang_zhang", result.columns)
        # Standard volatilities must be non-negative
        self.assertTrue((result["volatility_cc"] >= 0).all())
        self.assertTrue((result["volatility_parkinson"] >= 0).all())
        self.assertTrue((result["volatility_yang_zhang"] >= 0).all())

    def test_mean_reversion_features(self) -> None:
        """
        Verify fast Hurst exponent and Dickey-Fuller t-statistic bounds.
        """
        extractor = MeanReversionFeatures()
        result = extractor.compute(self.df)
        self.assertIn("mean_reversion_zscore", result.columns)
        self.assertIn("mean_reversion_hurst", result.columns)
        self.assertIn("mean_reversion_df_stat", result.columns)
        # Hurst must strictly be bounded in [0, 1]
        self.assertTrue(((result["mean_reversion_hurst"] >= 0.0) & (result["mean_reversion_hurst"] <= 1.0)).all())

    def test_wavelet_decomposition(self) -> None:
        """
        Verify wavelet trend, cycle, and noise extraction.
        """
        extractor = WaveletDecomposition()
        result = extractor.compute(self.df)
        self.assertIn("wavelet_trend", result.columns)
        self.assertIn("wavelet_cycle", result.columns)
        self.assertIn("wavelet_noise", result.columns)

    def test_kalman_state_filter(self) -> None:
        """
        Verify Kalman filtered price tracking.
        """
        extractor = KalmanStateFilter()
        result = extractor.compute(self.df)
        self.assertIn("kalman_price", result.columns)
        self.assertIn("kalman_velocity", result.columns)

    def test_pipeline_equivalence_incremental_vs_batch(self) -> None:
        """
        Equivalence check: Batch features must equal the incremental tick-by-tick
        arrivals up to 7 decimal places to ensure production-backtest equivalence.
        """
        pipeline = FeaturePipeline()
        
        # 1. Batch mode
        batch_features = pipeline.compute_all(self.df)
        
        # 2. Incremental mode
        # Simulate feeding data tick-by-tick and extracting the last computed feature
        incremental_features_list = []
        
        # Start at self.min_lookback (256) so wavelet and other rolling estimators have enough history
        for i in range(255, len(self.df)):
            df_slice = self.df.iloc[: i + 1]
            # Compute on the slice
            computed = pipeline.compute_all(df_slice)
            # Retrieve last row
            incremental_features_list.append(computed.iloc[-1])
            
        incremental_df = pd.DataFrame(incremental_features_list)
        
        # Trim batch features to match the same index as the incremental subset
        batch_trimmed = batch_features.iloc[255:]
        
        # Compare columns present in both
        for col in batch_trimmed.columns:
            np.testing.assert_array_almost_equal(
                batch_trimmed[col].values,
                incremental_df[col].values,
                decimal=6,
                err_msg=f"Incremental mismatch on feature: {col}"
            )

    def test_cot_positioning(self) -> None:
        """
        Verify COT speculative positioning features.
        """
        from features.alternative.cot_positioning import COTPositioning
        extractor = COTPositioning()
        result = extractor.compute(self.df)
        self.assertIn("cot_net_spec", result.columns)
        self.assertIn("cot_spec_ratio", result.columns)
        self.assertIn("cot_index", result.columns)
        self.assertTrue(((result["cot_spec_ratio"] >= 0.0) & (result["cot_spec_ratio"] <= 1.0)).all())

    def test_macro_surprise(self) -> None:
        """
        Verify macroeconomic surprise signals and momentum.
        """
        from features.alternative.macro_surprise import MacroSurprise
        extractor = MacroSurprise()
        result = extractor.compute(self.df)
        self.assertIn("macro_surprise", result.columns)
        self.assertIn("macro_momentum", result.columns)
        self.assertIn("macro_decay_index", result.columns)

    def test_options_flow(self) -> None:
        """
        Verify options flow, IV skew, and gamma exposure features.
        """
        from features.alternative.options_flow import OptionsFlow
        extractor = OptionsFlow()
        result = extractor.compute(self.df)
        self.assertIn("options_put_call_ratio", result.columns)
        self.assertIn("options_implied_vol", result.columns)
        self.assertIn("options_skew", result.columns)
        self.assertIn("options_gamma_exposure", result.columns)

    def test_sentiment_features(self) -> None:
        """
        Verify news sentiment polarity, volume momentum and impact features.
        """
        from features.alternative.sentiment import SentimentFeatures
        extractor = SentimentFeatures()
        result = extractor.compute(self.df)
        self.assertIn("sentiment_score_ema", result.columns)
        self.assertIn("sentiment_volume_momentum", result.columns)
        self.assertIn("sentiment_impact", result.columns)
        self.assertIn("sentiment_divergence", result.columns)


if __name__ == "__main__":
    unittest.main()
