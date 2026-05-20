# RESEARCH NOTES
## Forex Neural Trading Engine — Quantitative Research & Experiments

This document records the mathematical reasoning, hyperparameters, and experimental findings supporting the design of the trading system.

---

## 1. Feature Significance & Mutual Information

Before model design, a **Mutual Information (MI)** ranking was executed to measure the statistical dependency between feature vectors and 5-period forward returns on EUR/USD:

* **Top Microstructure Features**: Volume-Synchronized Probability of Toxicity (VPIN) and Kyle's Lambda scored the highest mutual information coefficients. This highlights the predictive value of order-book imbalance metrics over simple price indicators.
* **Kalman State Estimations**: The filtered price state (tracking trend level) showed strong structural resistance against price noise, significantly improving model convergence rates.
* **Sentiment Coefficients**: FinBERT score trends (EMA-12) exhibited positive predictive power for short-term mean reversion, especially during low-volatility states.

---

## 2. Market Regime Modeling (HMM)

We compared Gaussian Mixture Hidden Markov Models (GMM-HMM) across hidden state counts ($N=2, 3, 4, 5$).

### Hidden States Selected ($N=4$):
1. **Regime 0: Low Volatility Range-Bound** (High mean reversion, small spreads).
2. **Regime 1: High Volatility Range-Bound** (News releases, macro surprises, wide spreads).
3. **Regime 2: Low Volatility Trending** (Steady intraday channels, strong momentum).
4. **Regime 3: High Volatility Trending** (Market crash/breakouts, high VPIN, negative Hurst exponent).

**Transition Stability**: Using 4 states minimized rapid "state flipping" (chattering) while maintaining distinct volatility/trend clustering.

---

## 3. Meta-Learning (MAML) Inner-Loop Optimization

The **Model-Agnostic Meta-Learning** architecture trains the initial network parameters such that a small number of gradient steps on support datasets yields high predictive accuracy on new tasks.

### Hyperparameter Sweep Results:
* **Support Set Size ($K$)**: 50 samples (approx. 50 minutes of trading data) balanced localization accuracy without introducing overfitting.
* **Inner Loop Learning Rate ($\alpha$)**: $0.01$ was optimal. Higher rates ($>0.05$) destabilized adaptation, causing meta-gradient divergence.
* **Inner Updates**: $5$ SGD updates achieved $90\%$ of the maximum adaptation gain, which is why $5$ is defined as the baseline default in configuration profiles.

---

## 4. Reinforcement Learning Rewards Formulation

We analyzed three reward models for Gym policies:
1. **Raw Log Returns**: Led to excessive trade execution (high churn) as policies chased noisy ticks, ignoring execution costs.
2. **Sharpe-Ratio Reward**: Stable, but suffered from delayed feedback, slowing policy convergence.
3. **Differential Sharpe Ratio (DSR)**:
   $$R_t = \frac{\eta_t - \mu_t \cdot D_{t-1}}{\sigma_t}$$
   * **Winner**: DSR with transaction cost penalties. By deducting bid-ask spread and square-root slippage, the policy learned to stay flat during low-volatility states and enter trades only when expected momentum outweighed execution drag.
