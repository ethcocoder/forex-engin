# TASK TODO
## Granular Task Tracker

> Status: ⬜ Not started | 🔄 In progress | ✅ Done | ❌ Blocked | ⏸ Paused

---

## PHASE 0 — FOUNDATION

### Infrastructure
| # | Task | Status | Notes |
|---|------|--------|-------|
| 0.1 | Create GitHub repository + branch strategy | ⬜ | main, dev, feature/* |
| 0.2 | Write .gitignore (secrets, logs, model weights, data) | ⬜ | |
| 0.3 | Write docker-compose.yml (TimescaleDB, Redis, Kafka, MLflow, Grafana, Prometheus) | ⬜ | |
| 0.4 | Create requirements.txt | ⬜ | Pin all versions |
| 0.5 | Create requirements-dev.txt | ⬜ | pytest, black, mypy, etc. |
| 0.6 | Setup pre-commit hooks (black, isort, flake8, mypy) | ⬜ | |
| 0.7 | GitHub Actions CI (lint + test on push) | ⬜ | |
| 0.8 | Create .env.example with all variables documented | ⬜ | |
| 0.9 | TimescaleDB schema + migrations | ⬜ | See IMPLEMENTATION_PLAN.md |
| 0.10 | MLflow tracking server configured | ⬜ | |
| 0.11 | Grafana + Prometheus stack running | ⬜ | |
| 0.12 | Kafka 3-broker local cluster | ⬜ | |

### Base Classes
| # | Task | Status | Notes |
|---|------|--------|-------|
| 0.13 | configs/loader.py — YAML + env var merge | ⬜ | |
| 0.14 | models/base_model.py — abstract base | ⬜ | |
| 0.15 | features/base_feature.py — abstract base | ⬜ | |
| 0.16 | execution/brokers/base_broker.py — abstract base | ⬜ | |
| 0.17 | Unit tests for all base classes | ⬜ | |
| 0.18 | Logging infrastructure (structlog or loguru) | ⬜ | JSON logs |
| 0.19 | scripts/health_check.py | ⬜ | |

---

## PHASE 1 — DATA PIPELINE

### Historical Data
| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | scripts/download_data.py — Dukascopy tick downloader | ⬜ | |
| 1.2 | Download 5yr tick data — EUR/USD | ⬜ | ~20GB |
| 1.3 | Download 5yr tick data — GBP/USD, USD/JPY, USD/CHF | ⬜ | |
| 1.4 | Download 5yr tick data — EUR/GBP, EUR/JPY, AUD/USD, NZD/USD | ⬜ | |
| 1.5 | OHLCV resampler (tick → 1m/5m/1h/4h/1d) | ⬜ | |
| 1.6 | Data quality validator (gap detection, outlier flagging) | ⬜ | |
| 1.7 | Load all historical data into TimescaleDB | ⬜ | |
| 1.8 | Verify data completeness (% gaps per pair) | ⬜ | |

### Live Data Streaming
| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.9 | OANDA streaming WebSocket adapter | ⬜ | |
| 1.10 | Kafka producer for live tick stream | ⬜ | |
| 1.11 | Redis tick cache (latest 1000 ticks per pair) | ⬜ | |
| 1.12 | Order book streaming + storage | ⬜ | |
| 1.13 | Integration test: stream 48hr without errors | ⬜ | |
| 1.14 | Data freshness monitoring alert | ⬜ | Alert if no tick for > 60s |

### Alternative Data
| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.15 | Economic calendar pipeline (ForexFactory or Investing.com) | ⬜ | |
| 1.16 | COT report parser (CFTC weekly data) | ⬜ | |
| 1.17 | News NLP pipeline (FinBERT sentiment scoring) | ⬜ | |
| 1.18 | Kafka topics for all data streams | ⬜ | |

---

## PHASE 2 — FEATURE ENGINEERING

### Microstructure Features
| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | features/microstructure/spread.py | ⬜ | |
| 2.2 | features/microstructure/order_flow.py | ⬜ | |
| 2.3 | features/microstructure/lob_features.py | ⬜ | |
| 2.4 | features/microstructure/vpin.py | ⬜ | Easley et al. |
| 2.5 | features/microstructure/kyle_lambda.py | ⬜ | |
| 2.6 | features/microstructure/amihud.py | ⬜ | |
| 2.7 | Unit tests: all microstructure features | ⬜ | |

### Technical Features
| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.8 | features/technical/volatility.py (5 estimators) | ⬜ | |
| 2.9 | features/technical/momentum.py | ⬜ | |
| 2.10 | features/technical/mean_reversion.py | ⬜ | Hurst + ADF |
| 2.11 | features/technical/trend.py | ⬜ | |
| 2.12 | features/technical/volume.py | ⬜ | |
| 2.13 | Unit tests: all technical features | ⬜ | |

### Wavelet & Kalman
| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.14 | features/wavelet/decomposition.py (PyWavelets) | ⬜ | |
| 2.15 | features/wavelet/kalman_filter.py | ⬜ | |
| 2.16 | features/wavelet/spectral.py | ⬜ | |
| 2.17 | Unit tests: wavelet features | ⬜ | |

### Alternative Data Features
| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.18 | features/alternative/sentiment.py | ⬜ | |
| 2.19 | features/alternative/cot_positioning.py | ⬜ | |
| 2.20 | features/alternative/macro_surprise.py | ⬜ | |

### Pipeline Integration
| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.21 | features/pipeline.py — master orchestrator | ⬜ | |
| 2.22 | Feature correlation matrix analysis (notebook) | ⬜ | |
| 2.23 | Mutual information analysis (feature → forward return) | ⬜ | |
| 2.24 | Feature importance ranking | ⬜ | |
| 2.25 | Performance test: < 50ms per live update | ⬜ | |
| 2.26 | Integration test: live vs. batch output match | ⬜ | |

---

## PHASE 3 — NEURAL ENSEMBLE

### Temporal Model
| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | models/temporal/transformer.py | ⬜ | |
| 3.2 | models/temporal/tcn.py | ⬜ | |
| 3.3 | models/temporal/combined.py (cross-attention fusion) | ⬜ | |
| 3.4 | models/temporal/trainer.py | ⬜ | Purged CV |
| 3.5 | Hyperparameter search (Ray Tune) | ⬜ | |
| 3.6 | Evaluate: IC, ICIR, signal decay | ⬜ | Target: IC > 0.03 |
| 3.7 | Register model in MLflow | ⬜ | |
| 3.8 | Unit tests: temporal model | ⬜ | |

### Regime Model
| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.9 | models/regime/hmm.py (hmmlearn, 4 states) | ⬜ | |
| 3.10 | models/regime/lstm_classifier.py | ⬜ | |
| 3.11 | models/regime/combined.py | ⬜ | |
| 3.12 | models/regime/trainer.py | ⬜ | |
| 3.13 | Evaluate: regime accuracy, transition quality | ⬜ | |
| 3.14 | Register model in MLflow | ⬜ | |

### RL Agent
| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.15 | models/rl_agent/environment.py (Gym env) | ⬜ | |
| 3.16 | models/rl_agent/reward_functions.py | ⬜ | Sharpe-adjusted reward |
| 3.17 | models/rl_agent/ppo_agent.py | ⬜ | Stable-Baselines3 |
| 3.18 | models/rl_agent/sac_agent.py | ⬜ | |
| 3.19 | models/rl_agent/trainer.py | ⬜ | Curriculum learning |
| 3.20 | Phase 1 training: trending regimes only (1M steps) | ⬜ | |
| 3.21 | Phase 2 training: all regimes (1M steps) | ⬜ | |
| 3.22 | Phase 3 training: fine-tune on last 1yr (500K steps) | ⬜ | |
| 3.23 | Evaluate OOS: Sharpe, drawdown, win rate | ⬜ | |
| 3.24 | Register best agent in MLflow | ⬜ | |

### Meta-Learner
| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.25 | models/meta_learner/maml.py | ⬜ | Second-order gradients |
| 3.26 | models/meta_learner/online_adapter.py | ⬜ | |
| 3.27 | models/meta_learner/trainer.py | ⬜ | |
| 3.28 | Evaluate: adaptation speed across regime changes | ⬜ | |

### Ensemble
| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.29 | models/ensemble/aggregator.py (stacking + BMA) | ⬜ | |
| 3.30 | models/ensemble/uncertainty.py (MC Dropout) | ⬜ | |
| 3.31 | models/ensemble/weighting.py | ⬜ | |
| 3.32 | models/ensemble/signal_generator.py | ⬜ | AlphaSignal dataclass |
| 3.33 | Ensemble OOS Sharpe target: > 1.5 | ⬜ | |

---

## PHASE 4 — RISK ENGINE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | risk/sizing/kelly.py | ⬜ | Fractional Kelly |
| 4.2 | risk/sizing/volatility_scaled.py | ⬜ | |
| 4.3 | risk/limits/cvar_limits.py | ⬜ | 95% CVaR |
| 4.4 | risk/limits/drawdown_limits.py | ⬜ | 3-tier circuit breaker |
| 4.5 | risk/limits/correlation_limits.py | ⬜ | |
| 4.6 | risk/limits/liquidity_filter.py | ⬜ | |
| 4.7 | risk/limits/session_filter.py | ⬜ | |
| 4.8 | risk/risk_engine.py — gate + size master | ⬜ | |
| 4.9 | risk/monitoring/portfolio_monitor.py | ⬜ | |
| 4.10 | risk/monitoring/pnl_attribution.py | ⬜ | |
| 4.11 | risk/monitoring/alert_manager.py | ⬜ | |
| 4.12 | Unit tests: all circuit breakers | ⬜ | CRITICAL |
| 4.13 | Integration test: risk engine gates bad orders | ⬜ | CRITICAL |

---

## PHASE 5 — EXECUTION

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | execution/brokers/paper_broker.py | ⬜ | |
| 5.2 | execution/brokers/oanda_broker.py | ⬜ | |
| 5.3 | execution/simulation/slippage_model.py | ⬜ | |
| 5.4 | execution/simulation/fill_simulator.py | ⬜ | |
| 5.5 | execution/simulation/market_impact.py | ⬜ | |
| 5.6 | execution/routing/twap.py | ⬜ | |
| 5.7 | execution/routing/vwap.py | ⬜ | |
| 5.8 | execution/execution_engine.py | ⬜ | |
| 5.9 | Integration tests: all broker adapters | ⬜ | |

---

## PHASE 6 — BACKTESTING

| # | Task | Status | Notes |
|---|------|--------|-------|
| 6.1 | backtesting/engines/event_driven.py | ⬜ | Core engine |
| 6.2 | backtesting/engines/vectorized.py | ⬜ | Fast research version |
| 6.3 | backtesting/performance.py | ⬜ | All metrics |
| 6.4 | backtesting/scenarios/walk_forward.py | ⬜ | |
| 6.5 | backtesting/scenarios/monte_carlo.py | ⬜ | |
| 6.6 | backtesting/scenarios/regime_stress.py | ⬜ | |
| 6.7 | backtesting/scenarios/historical_stress.py | ⬜ | 2008, 2015 CHF, 2020 |
| 6.8 | Full system backtest (5yr, all pairs) | ⬜ | |
| 6.9 | Walk-forward validation (18m train / 3m OOS) | ⬜ | |
| 6.10 | Target: Mean OOS Sharpe > 1.5 | ⬜ | |

---

## PHASE 7 — MONITORING & LIVE

| # | Task | Status | Notes |
|---|------|--------|-------|
| 7.1 | monitoring/metrics_collector.py (Prometheus) | ⬜ | |
| 7.2 | monitoring/signal_monitor.py | ⬜ | |
| 7.3 | monitoring/model_monitor.py | ⬜ | |
| 7.4 | Grafana: PnL dashboard | ⬜ | |
| 7.5 | Grafana: Risk dashboard | ⬜ | |
| 7.6 | Grafana: Model performance dashboard | ⬜ | |
| 7.7 | Grafana: Execution quality dashboard | ⬜ | |
| 7.8 | Alert rules (risk breach, model degradation, system) | ⬜ | |
| 7.9 | monitoring/reporting/daily_report.py | ⬜ | |
| 7.10 | 30-day paper trading run | ⬜ | Pass: Sharpe > 1.5 |
| 7.11 | Live deployment checklist complete | ⬜ | See IMPLEMENTATION_PLAN.md |
| 7.12 | Capital scaling protocol: 10% → 25% → 50% → 100% | ⬜ | |

---

## DOCUMENTATION TASKS

| # | Task | Status | Notes |
|---|------|--------|-------|
| D.1 | docs/architecture/ARCHITECTURE.md | ⬜ | |
| D.2 | docs/architecture/DATA_FLOW.md | ⬜ | |
| D.3 | docs/architecture/MODEL_DESIGN.md | ⬜ | |
| D.4 | docs/api/API_REFERENCE.md | ⬜ | Auto-generate with pdoc |
| D.5 | docs/research/RESEARCH_NOTES.md | ⬜ | Ongoing |
| D.6 | research/papers/README.md — paper index | ⬜ | |
| D.7 | Ops runbook (incident response, emergency stop) | ⬜ | |

---

## Bugs & Issues

_Track issues here as they arise._

| # | Issue | Status | Priority |
|---|-------|--------|----------|
| — | — | — | — |

---

## Decisions Log

_Record key architectural decisions and why you made them._

| Date | Decision | Reason |
|------|----------|--------|
| — | PPO as primary RL algorithm | Stable training, well-documented, proven in finance |
| — | TimescaleDB over InfluxDB | SQL interface, joins with relational data |
| — | Fractional Kelly at 0.25x | Kelly assumes perfect probability estimates; 0.25x accounts for estimation error |
| — | 4 regime states (HMM) | More states → unstable transitions; fewer → insufficient granularity. 4 validated empirically |
