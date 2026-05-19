# FOREX NEURAL TRADING ENGINE
### Production-Grade Quantitative Trading System

> Inspired by Renaissance Technologies' Medallion Fund philosophy: **the model decides, not humans.**

---

## Overview

A full production-grade Forex trading engine combining deep learning, reinforcement learning, and classical quantitative methods into a unified neural ensemble. Designed to identify and exploit statistical inefficiencies across Forex markets with institutional-grade risk management and execution.

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

## Performance Targets

| Metric | Target |
|--------|--------|
| Annualized Sharpe | > 2.5 |
| Max Drawdown | < 15% |
| Win Rate | > 52% |
| Avg R:R Ratio | > 1.5 |
| Calmar Ratio | > 1.8 |

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

# Configure environment
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your broker credentials and settings

# Start infrastructure
docker-compose up -d

# Run backtests
python scripts/run_backtest.py --config configs/backtest.yaml

# Start live paper trading
python scripts/run_paper_trading.py --config configs/paper.yaml
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
