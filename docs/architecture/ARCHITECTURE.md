# ARCHITECTURE
## System Design Deep Dive

---

## Design Philosophy

This system is designed around three core constraints:

1. **No future leakage** — Every feature, model, and backtest must use only data available at decision time. This is the most common source of false alpha in quant research.

2. **Production parity** — Backtest and live trading share the same code paths. The only difference is the data source (historical vs. live) and the broker adapter (simulator vs. real).

3. **Explicit uncertainty** — Every prediction carries a confidence estimate. The risk engine uses uncertainty to size positions smaller when the model is unsure.

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA SOURCES                         │
│  Broker WebSocket  │  Economic Calendar  │  News API   │
└────────┬───────────┴──────────┬──────────┴──────┬───────┘
         │                      │                  │
         ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                  KAFKA (Message Bus)                    │
│  topic: ticks.{pair}  │  topic: events  │  topic: news │
└────────┬───────────────────────────────────────────────-┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│               FEATURE PIPELINE (Consumer)               │
│  Microstructure │ Technical │ Wavelet │ Alternative     │
└────────┬────────────────────────────────────────────────┘
         │ features vector (150 dims)
         ▼
┌─────────────────────────────────────────────────────────┐
│                  NEURAL ENSEMBLE                        │
│  Temporal Model │ Regime Model │ RL Agent │ Meta-Learner│
│                 └──── Aggregator ────┘                  │
└────────┬────────────────────────────────────────────────┘
         │ AlphaSignal(direction, magnitude, confidence, uncertainty)
         ▼
┌─────────────────────────────────────────────────────────┐
│                   RISK ENGINE                           │
│  Kelly Sizing → CVaR Check → Drawdown Check → Gate      │
└────────┬────────────────────────────────────────────────┘
         │ sized, gated order
         ▼
┌─────────────────────────────────────────────────────────┐
│                 EXECUTION ENGINE                        │
│  Router → TWAP/VWAP → Broker API → Fill Handler        │
└────────┬────────────────────────────────────────────────┘
         │ fills
         ▼
┌─────────────────────────────────────────────────────────┐
│              PORTFOLIO & MONITORING                     │
│  PnL Attribution │ Risk Metrics │ Model Monitor        │
└──────────────────────────────────────────────────────-──┘
         │
         ▼ (feedback)
   Model Retraining Trigger → MLflow → Updated Weights
```

---

## Key Architectural Decisions

### Event-Driven Core
The system is event-driven throughout. A `MarketEvent` triggers the feature pipeline, which produces a `FeatureEvent`, which triggers the model ensemble, which produces a `SignalEvent`, which the risk engine converts to an `OrderEvent` or discards.

This architecture enables:
- Identical behavior in backtest and live (same event handlers)
- Easy testing of each component in isolation
- Auditability (every event is logged)

### Stateless Models, Stateful System
Each model call is stateless — given the same input, it produces the same output. All state (portfolio, regime history, feature cache) lives in dedicated stateful components (Redis, TimescaleDB, portfolio object).

This makes models easy to test, version, and replace without touching the rest of the system.

### Single Point of Risk Control
All order flow passes through `RiskEngine.gate(signal, portfolio_state)`. There is no way to place an order that bypasses the risk engine. This is enforced architecturally — the execution engine has no direct access to the broker; it only receives pre-approved order objects from the risk engine.

---

## Model Versioning

All models are versioned in MLflow. The system loads models by version tag, not by file path. This allows:
- Rollback to previous model version in seconds
- A/B testing between model versions
- Audit trail of which model was live when

Production promotion flow:
```
Research notebook → MLflow experiment → 
  Backtest validation → MLflow model registry (Staging) →
  Paper trading validation → MLflow model registry (Production) →
  Live deployment
```

---

## Failure Modes & Mitigations

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Broker API timeout | Connection watchdog | Retry with backoff, then halt |
| Feature computation error | Feature validation | Use last valid features, log alert |
| Model inference error | Try/except + logging | Fall back to conservative sizing |
| Data gap | Data freshness monitor | Halt trading, alert |
| Circuit breaker triggered | Portfolio monitor | Auto-halt, alert on-call |
| Regime shift | Regime model confidence drop | Reduce all position sizes 50% |
