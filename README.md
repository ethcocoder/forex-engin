# FOREX NEURAL TRADING ENGINE
### Research-Stage Quantitative FX System

> This project uses common quantitative-research patterns. It has **no affiliation with Renaissance Technologies or its funds**, and it is not authorised for live trading.

---

## Overview

A research and validation platform for causal FX data preparation, model training, out-of-sample diagnostics, cost-scenario backtesting, risk gates, and audit artifacts. Broker integration is intentionally deferred until independently defined research gates are satisfied.

---

## Architecture Summary

```
┌─────────────────────────────────────────────┐
│           Layer 0: Data Ingestion           │
│  Tick · Order Book · Macro · Sentiment      │
├─────────────────────────────────────────────┤
│         Layer 1: Feature Engineering       │
│  Microstructure · Wavelet · Kalman · LOB   │
├─────────────────────────────────────────────┤
│       Layer 2: Neural Ensemble Core        │
│  Temporal | Regime | RL Agent | Meta       │
│         └── Ensemble Aggregator ──┘        │
├─────────────────────────────────────────────┤
│           Layer 3: Alpha Signal            │
├─────────────────────────────────────────────┤
│           Layer 4: Risk Engine             │
│  Kelly · CVaR · Drawdown · Correlation     │
├─────────────────────────────────────────────┤
│           Layer 5: Execution               │
│  Smart Routing · TWAP/VWAP · Slippage      │
├─────────────────────────────────────────────┤
│       Layer 6: Monitoring & Feedback       │
│  PnL · Sharpe · Decay · Retrain Trigger    │
└─────────────────────────────────────────────┘
```

---

## Core Philosophy

1. **Data over instinct** — Every decision is model-driven. No manual overrides in live trading.
2. **Uncertainty is a signal** — Confidence intervals are first-class outputs, not afterthoughts.
3. **Regime awareness** — The market is not stationary. The system must know what game it is playing.
4. **Risk first** — Sizing and risk management are as important as signal generation.
5. **Feedback loops** — The system must learn from its own live performance continuously.

---

## Target Markets

- Major pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF
- Minor pairs: EUR/GBP, EUR/JPY, GBP/JPY
- Commodities FX: AUD/USD, NZD/USD, USD/CAD

---

## Current Research Status

| Area | Status |
|--------|--------|
| Causal data contract, labels, core features, and experiment artifacts | Implemented and tested |
| Baseline and temporal out-of-sample experiments | Implemented; current repository-data experiments do **not** pass promotion gates |
| Cost-scenario backtest and drawdown gate | Implemented and tested |
| Broker/paper adapter | Deferred pending positive, independently reviewed research evidence |

No return, Sharpe, win-rate, or drawdown target is a promise or a deployment criterion by itself. Every candidate must pass the documented out-of-sample and cost-aware readiness gates.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| ML Framework | PyTorch 2.x |
| RL Framework | Stable-Baselines3 / RLlib |
| Feature Store | Feast |
| Streaming | Apache Kafka |
| Tick Storage | Redis (hot) + TimescaleDB (cold) |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow |
| Deployment | Docker + Kubernetes |
| Monitoring | Grafana + Prometheus |
| Backtesting | Custom event-driven engine |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/forex-neural-engine
cd forex-neural-engine

# Install dependencies
pip install -r requirements.txt

# Generate a strict OHLCV-derived feature matrix (no synthetic bid/ask or alternative data)
python scripts/generate_features.py \
  --input data/EUR_USD_ticks.csv \
  --output artifacts/research_data/EUR_USD_core_features.csv \
  --pair EUR_USD --provider repository_csv

# Train a causal baseline and save OOS diagnostics
python scripts/run_baseline_experiment.py \
  --raw data/EUR_USD_ticks.csv \
  --features artifacts/research_data/EUR_USD_core_features.csv

# Evaluate only OOS predictions under explicitly declared scenario costs
python scripts/evaluate_oos_backtest.py artifacts/experiments/<run-id> \
  --half-spread-bps 0.5 --slippage-bps 0.5

# Record the formal promotion-gate result
python scripts/run_readiness_gates.py artifacts/experiments/<run-id>

# Broker/paper execution remains disabled until a candidate passes review.
```

---

## Documentation

- [Architecture Deep Dive](docs/architecture/ARCHITECTURE.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Research Notes](docs/research/RESEARCH_NOTES.md)
- [Roadmap](ROADMAP.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)
- [Task Tracker](TASK_TODO.md)

---

## Project Structure

See [DOCUMENTATION.md](DOCUMENTATION.md) for full folder structure reference.

---

## Warning

> **This system is for research and educational purposes. Trading Forex involves substantial risk of loss. Past performance of any model does not guarantee future results. Never trade with capital you cannot afford to lose.**

---

## License

MIT License — see LICENSE file.
