# FEATURE ANALYSIS
## Forex Neural Trading Engine — Feature Engineering & Selection

This document details the feature processing pipelines designed for the Forex Neural Trading Engine, explaining the reasoning behind the quantitative indicators and microstructure variables extracted.

---

## 1. Microstructure Features

Unlike traditional technical indicators (which lag price), microstructure features capture order book imbalances and liquidity conditions, providing leading indications of short-term price movements.

* **Volume-Synchronized Probability of Toxicity (VPIN)**: 
  Estimates the probability that liquidity providers are trading against informed investors (toxic flow). Calculated by bucketing ticks by volume rather than time, and measuring the imbalance between buy and sell pressure within these volume buckets. High VPIN strongly correlates with imminent volatility spikes and regime transitions.
* **Kyle's Lambda**:
  Measures market impact (price change per unit of volume). Used to dynamically adjust slippage estimates and detect illiquid market states where large orders would cause unacceptable drag.
* **Amihud Illiquidity**:
  Calculated as the absolute return divided by dollar trading volume. Identifies periods of low liquidity where small volumes create outsized price impacts.

---

## 2. Technical & Volatility Features

Standard technical indicators are transformed into stationary features to prevent neural networks from learning absolute price levels (which would fail on out-of-sample data).

* **Realized Volatility**: Rolling standard deviation of logarithmic returns over multiple time horizons (e.g., 5m, 1h, 4h, 1d). Forms the primary input for position sizing (Volatility-scaled Kelly).
* **Hurst Exponent**:
  Measures the long-term memory of a time series. 
  - $H < 0.5$: Mean-reverting regime.
  - $H \approx 0.5$: Geometric Brownian Motion (random walk).
  - $H > 0.5$: Trending regime.
  This is a critical input to the Hidden Markov Model (HMM) for regime classification.
* **Normalized Momentum & MACD**: Converted to Z-scores relative to rolling 100-period windows to enforce stationarity.

---

## 3. Signal Processing (Wavelet & Kalman)

To filter high-frequency noise from true directional trends:

* **Kalman Filter**:
  An iterative state estimator that dynamically tracks the "true" underlying price by modeling market noise as a Gaussian process. Crucial for smoothing price series without introducing the massive lag inherent in moving averages.
* **Wavelet Decomposition**:
  Uses Daubechies (`db4`) wavelets to decompose price series into frequency domains. By stripping out the highest frequency components (noise) and feeding the reconstructed mid-frequency signals into the model, the temporal network significantly improves its signal-to-noise ratio.

---

## 4. Alternative Data Integration

* **COT (Commitment of Traders) Index**: 
  Extracts institutional (non-commercial) net positioning from weekly CFTC reports. Provides a macro-trend bias for the Meta-Learner over long horizons (weeks to months).
* **FinBERT Sentiment**:
  Natural Language Processing (NLP) pipeline running on live economic news feeds. Categorizes news sentiment into $[-1, 1]$ float values, helping the temporal model interpret fundamental macro shocks (e.g., NFP payroll surprises, central bank rate decisions) before they are fully priced into technical indicators.
