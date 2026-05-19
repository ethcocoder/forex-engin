# ROADMAP
## Strategic Vision & Milestones

---

## Vision

Build the most sophisticated retail-accessible Forex trading engine — matching institutional-grade signal quality with production-grade reliability. The system should be self-improving: learning from live performance, adapting to regime changes, and continuously refining its edge.

The north star metric: **Sharpe > 2.5 sustained over 24 months of live trading.**

---

## Milestone Overview

```
Phase 0   Phase 1    Phase 2    Phase 3    Phase 4    Phase 5
  │          │          │          │          │          │
Foundation  Data &   Feature    Neural     Risk &    Live &
  Setup    Pipeline  Engine   Ensemble  Execution  Scale
  ████      ████      ████      ████      ████      ████
 Wk 1-2   Wk 3-6   Wk 7-10  Wk 11-18  Wk 19-22  Wk 23+
```

---

## Phase 0 — Foundation (Weeks 1–2)

**Goal:** Everything needed to build is in place.

### Infrastructure
- [ ] Repository setup (GitHub, branch strategy, PR templates)
- [ ] Development environment (Docker, Python virtualenv, pre-commit hooks)
- [ ] CI/CD pipeline (GitHub Actions — lint, test, build)
- [ ] TimescaleDB + Redis deployment (local Docker)
- [ ] MLflow tracking server setup
- [ ] Kafka cluster (3-broker local setup)
- [ ] Grafana + Prometheus stack

### Data Access
- [ ] Broker API accounts (OANDA paper account minimum)
- [ ] Historical data source confirmed (Dukascopy, HistData, or broker archive)
- [ ] Economic calendar data source (Investing.com API or ForexFactory scraper)

### Codebase
- [ ] All folder structure created
- [ ] Base classes defined (BaseBroker, BaseFeature, BaseModel)
- [ ] Config system working (YAML + env var override)
- [ ] Logging infrastructure
- [ ] Unit test framework (pytest)

**Exit criteria:** `python scripts/health_check.py` passes all checks.

---

## Phase 1 — Data Pipeline (Weeks 3–6)

**Goal:** Clean, reliable, multi-source data flowing continuously.

### Tick Data
- [ ] Historical tick data downloaded (5+ years, major pairs)
- [ ] Tick data ingestion pipeline (Kafka producer)
- [ ] TimescaleDB schema deployed and tested
- [ ] OHLCV resampling pipeline (1m, 5m, 1h, 4h, 1d)
- [ ] Data quality checks (gap detection, anomaly flagging)

### Order Book
- [ ] Level 2 order book streaming (OANDA or LMAX)
- [ ] Order book snapshot storage
- [ ] Order book feature extraction (imbalance, spread, depth)

### Alternative Data
- [ ] Economic calendar pipeline
- [ ] COT report parser (CFTC data)
- [ ] News NLP pipeline (financial BERT or FinBERT)

### Validation
- [ ] Data pipeline integration tests
- [ ] Backfill 5 years of all major pairs
- [ ] Data freshness monitoring alerts

**Exit criteria:** 5 years of tick data for 8 pairs stored and queryable. Live tick stream running for 48 hours without errors.

---

## Phase 2 — Feature Engineering (Weeks 7–10)

**Goal:** A rich, validated feature set. Every feature has proven predictive value.

### Microstructure Features
- [ ] Bid-ask spread (raw, EWM-smoothed)
- [ ] Order flow imbalance (1m, 5m windows)
- [ ] VPIN (Volume-Synchronized PIN)
- [ ] Kyle's lambda
- [ ] Amihud illiquidity ratio
- [ ] LOB queue imbalance (5 levels deep)

### Technical Features
- [ ] Realized volatility (Rogers-Satchell, Yang-Zhang estimators)
- [ ] GARCH(1,1) volatility forecast
- [ ] Multi-timeframe momentum (5m, 1h, 4h)
- [ ] Hurst exponent (trending vs. mean-reverting)
- [ ] ADF test rolling (mean-reversion strength)

### Frequency Domain
- [ ] Daubechies wavelet decomposition (5 levels)
- [ ] Kalman filter (trend + noise decomposition)
- [ ] FFT spectral features

### Validation
- [ ] Feature correlation matrix analysis
- [ ] Mutual information with forward returns
- [ ] Feature importance (permutation importance on simple model)
- [ ] Purged cross-validation framework

**Exit criteria:** Feature pipeline produces 150+ validated features. No feature leakage confirmed. Feature computation < 100ms for live inference.

---

## Phase 3 — Neural Ensemble (Weeks 11–18)

**Goal:** All four models trained, validated, and ensemble producing alpha signals.

### Temporal Model (Weeks 11–13)
- [ ] Transformer encoder (multi-head self-attention, positional encoding)
- [ ] Temporal Convolutional Network (TCN) with dilated convolutions
- [ ] Fusion architecture (cross-attention between Transformer and TCN)
- [ ] Training loop with purged cross-validation
- [ ] Hyperparameter search (Ray Tune)
- [ ] Model evaluation (IC, ICIR, signal decay curve)

### Regime Model (Weeks 12–13)
- [ ] HMM with Gaussian emissions (3–5 states)
- [ ] LSTM regime classifier (HMM state as auxiliary input)
- [ ] Regime transition probability outputs
- [ ] Regime-conditional strategy switching logic

### RL Agent (Weeks 14–16)
- [ ] Gym-compatible Forex environment
  - State space: features + portfolio state + regime
  - Action space: long/flat/short + position size
  - Reward: Sharpe-penalized PnL (risk-adjusted)
- [ ] PPO agent (Stable-Baselines3)
- [ ] SAC agent (entropy regularization for exploration)
- [ ] Curriculum learning (easy → hard market conditions)
- [ ] RL evaluation (Sharpe, drawdown, regime performance)

### Meta-Learner (Weeks 16–17)
- [ ] MAML implementation (second-order gradient updates)
- [ ] Online adaptation wrapper (few-shot regime adaptation)
- [ ] Meta-training across regime episodes

### Ensemble Aggregator (Week 18)
- [ ] Stacking layer (gradient boosted meta-model)
- [ ] Bayesian model averaging
- [ ] MC Dropout uncertainty quantification
- [ ] Dynamic weighting by recent model confidence
- [ ] Alpha signal object design (direction, magnitude, CI, decay)

**Exit criteria:** Ensemble achieves Sharpe > 1.5 in walk-forward out-of-sample test. Signal IC > 0.03. No obvious overfitting (live vs. backtest Sharpe ratio > 0.7).

---

## Phase 4 — Risk & Execution (Weeks 19–22)

**Goal:** Production-grade risk engine and execution layer.

### Risk Engine
- [ ] Kelly criterion (fractional Kelly, 0.25x cap)
- [ ] CVaR calculation (historical simulation, 95% confidence)
- [ ] Drawdown circuit breakers (daily 3%, weekly 7%, monthly 15%)
- [ ] Cross-pair correlation matrix (live, 20-day rolling)
- [ ] Correlation exposure cap enforcement
- [ ] Liquidity gating (spread threshold, session filter)
- [ ] Overnight/weekend exposure rules

### Execution
- [ ] Broker adapter (OANDA paper — full implementation)
- [ ] Paper broker (internal simulation)
- [ ] Slippage model (spread + market impact)
- [ ] Fill simulator
- [ ] TWAP execution algorithm
- [ ] Order state machine (pending → filled / rejected / cancelled)
- [ ] Retry logic with backoff

### Paper Trading
- [ ] Full system integration test (paper trading)
- [ ] 30-day paper trading validation
- [ ] Performance vs. backtest comparison
- [ ] Execution quality analysis

**Exit criteria:** Paper trading Sharpe > 1.5 over 30 days. Maximum drawdown < 15%. Order rejection rate < 1%.

---

## Phase 5 — Live Trading & Scale (Weeks 23+)

**Goal:** Real capital deployed. System monitors itself. Continuous improvement loop active.

### Live Deployment
- [ ] Live broker adapter tested with tiny position sizes
- [ ] Gradual capital scaling protocol (10% → 25% → 50% → 100% of allocation)
- [ ] Emergency stop procedures documented and tested
- [ ] Ops runbook complete

### Monitoring & Feedback
- [ ] Grafana dashboards live (PnL, risk, model, execution)
- [ ] Signal decay monitoring
- [ ] Live vs. backtest divergence alerts
- [ ] Automated retraining trigger (on performance degradation)
- [ ] Daily automated performance report

### Continuous Improvement
- [ ] Online learning pipeline (model updates from live fills)
- [ ] A/B testing framework for model improvements
- [ ] Research pipeline feeding production
- [ ] Quarterly model review process

---

## Long-Term Vision (Year 2+)

| Initiative | Description |
|---|---|
| Additional markets | Crypto pairs, precious metals FX |
| Higher frequency | Tick-level microstructure strategies |
| Portfolio optimization | Cross-strategy capital allocation |
| Alternative execution | FX options for delta hedging |
| Cloud scaling | Multi-region deployment, lower latency |
| Institutional grade | FIX protocol, prime broker integration |

---

## Risk Factors & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Overfitting | Model fails live | Purged CV, walk-forward, OOS holdout |
| Regime change | Sudden drawdown | Regime model + meta-learner adaptation |
| Broker outage | Missed fills | Multi-broker failover |
| Data quality issues | Bad signals | Data quality monitoring + alerts |
| Model degradation | Declining edge | Signal decay tracker + retrain trigger |
| Slippage underestimation | Eroded alpha | Conservative slippage model, live monitoring |
