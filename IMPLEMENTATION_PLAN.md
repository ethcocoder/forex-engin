# IMPLEMENTATION PLAN
## Technical Build Sequence & Specifications

---

## Principles

1. **Build bottom-up.** Infrastructure before features, features before models, models before risk, risk before execution.
2. **Test at every layer.** No layer depends on untested code from the layer below.
3. **Backtest before live.** Every component is backtested before paper trading. Paper trading before live.
4. **Fail loudly.** Every component validates its inputs. Silent failures are unacceptable in a trading system.
5. **No hardcoded values.** Every parameter lives in config files.

---

## Sprint 1 — Core Infrastructure (Days 1–7)

### Day 1–2: Environment
```
Goal: A clean, reproducible dev environment.

Tasks:
  - Create virtualenv, install core dependencies
  - Docker Compose for: TimescaleDB, Redis, Kafka, MLflow, Grafana, Prometheus
  - .env.example with all required variables documented
  - Pre-commit hooks: black, isort, flake8, mypy
  - GitHub Actions CI: lint + test on every push

Deliverable: `docker-compose up -d` starts full stack. CI passes.
```

### Day 3–4: Base Classes & Config System
```
Goal: Shared abstractions that all modules build on.

Classes to implement:
  - configs/loader.py         → YAML + env var merge, validation
  - models/base_model.py      → Abstract: fit(), predict(), save(), load()
  - features/base_feature.py  → Abstract: compute(), validate()
  - execution/brokers/base_broker.py → Abstract: connect(), place_order(), get_positions()
  - risk/risk_engine.py       → Abstract gate interface

Tests:
  - Config loading with env var override
  - Base class contracts (ABC compliance)
```

### Day 5–7: Data Layer
```
Goal: Historical data stored and queryable.

Tasks:
  - TimescaleDB hypertable schema (ticks, ohlcv, orderbook, features)
  - Data downloader script (Dukascopy or HistData)
  - OHLCV resampler (tick → 1m → 5m → 1h → 4h → 1d)
  - Data quality validator (gap detection, outlier flagging)
  - Redis tick cache (latest 1000 ticks per pair, fast inference access)

Schema (TimescaleDB):
  ticks(time, pair, bid, ask, volume)           → hypertable on time
  ohlcv(time, pair, tf, open, high, low, close, volume)
  features(time, pair, tf, feature_name, value)
  orders(id, time, pair, side, size, price, status, fill_price, slippage)
  trades(id, open_time, close_time, pair, pnl, sharpe_contribution)

Tests:
  - Round-trip: insert tick → query by time range
  - OHLCV resampling correctness (compare against pandas resample)
  - Data quality checks on known-bad data
```

---

## Sprint 2 — Feature Engineering (Days 8–21)

### Microstructure Features (Days 8–11)
```
Implementation order:
  1. spread.py           → bid-ask spread, % spread, EWM spread
  2. order_flow.py       → signed trade flow, cumulative delta
  3. lob_features.py     → depth imbalance (5 levels), weighted mid
  4. vpin.py             → VPIN bucket computation (50-bucket default)
  5. kyle_lambda.py      → OLS regression: Δprice ~ sign(volume)
  6. amihud.py           → |return| / volume ratio

Each feature class:
  - Input: pd.DataFrame with required columns
  - Output: pd.Series or pd.DataFrame
  - compute(df, **params) → validated output
  - validate(df) → raises FeatureValidationError on bad input

Tests: Against known values from academic papers.
```

### Technical Features (Days 12–15)
```
Implementation order:
  1. volatility.py       → Realized vol (5 estimators), GARCH(1,1)
  2. momentum.py         → RSI, MACD, Rate of Change (multi-lookback)
  3. mean_reversion.py   → Z-score, Hurst exponent, rolling ADF
  4. trend.py            → ADX, linear regression slope, trend strength
  5. volume.py           → Tick volume, volume delta, VPOC estimation

Note on Hurst exponent:
  Use R/S analysis (Hurst, 1951). Hurst < 0.5 = mean-reverting,
  Hurst ≈ 0.5 = random walk, Hurst > 0.5 = trending.
  This drives regime-conditional strategy selection.
```

### Wavelet & Kalman (Days 16–18)
```
wavelet/decomposition.py:
  - PyWavelets db4 wavelet, 5 decomposition levels
  - Separate: trend (low-freq) + noise (high-freq) + cycles (mid-freq)
  - Output: one signal per frequency band

wavelet/kalman_filter.py:
  - State space: [price, velocity]
  - Observation: raw tick mid-price
  - Q (process noise) and R (observation noise) from config
  - Output: filtered price + trend velocity estimate

Key insight: Kalman-filtered price is the input to regime detection,
not raw price. This removes microstructure noise before regime classification.
```

### Feature Pipeline Integration (Days 19–21)
```
features/pipeline.py:
  - Orchestrates all feature classes
  - Handles NaN propagation (minimum lookback enforcement)
  - Caches computed features to TimescaleDB
  - Live mode: incremental computation on new tick
  - Backtest mode: batch computation on full history

Performance target: < 50ms per feature update in live mode
Test: Compare live incremental vs. backtest batch output (must be identical)
```

---

## Sprint 3 — Neural Models (Days 22–50)

### Temporal Model (Days 22–32)

#### Architecture
```python
# models/temporal/transformer.py
class ForexTransformer(nn.Module):
    """
    Multi-head self-attention over feature sequences.
    
    Input:  (batch, seq_len=60, n_features=150)
    Output: (batch, d_model=256)  ← contextualized representation
    
    Architecture:
      - Input projection: n_features → d_model
      - Positional encoding (learned, not sinusoidal)
      - 4x TransformerEncoderLayer (d_model=256, nhead=8, dim_ff=512)
      - Output: CLS token representation
    
    Why learned positional encoding?
      Financial time series has non-uniform temporal patterns.
      Learned encodings adapt to the specific patterns in Forex data.
    """

# models/temporal/tcn.py  
class TemporalConvNet(nn.Module):
    """
    Dilated causal convolutions for multi-scale pattern extraction.
    
    Input:  (batch, n_features=150, seq_len=60)
    Output: (batch, n_channels=128)
    
    Architecture:
      - 6 residual blocks with exponentially increasing dilation
      - Dilation factors: [1, 2, 4, 8, 16, 32]
      - Receptive field: covers full 60-step sequence
      - Causal padding: no future leakage
    
    Why TCN alongside Transformer?
      Transformer captures global dependencies (attends across full sequence).
      TCN captures local patterns at multiple temporal scales.
      Together: global context + local detail.
    """

# models/temporal/combined.py
class TemporalModel(nn.Module):
    """
    Fuses Transformer and TCN representations.
    
    Fusion: Cross-attention (Transformer query, TCN key/value)
    Output: (batch, 2) → [direction_logit, magnitude_estimate]
    """
```

#### Training Protocol
```
Data preparation:
  - Purged k-fold cross-validation (gap = signal decay period)
  - Label: forward return over [1h, 4h, 8h, 24h] (multi-task)
  - Normalization: rolling z-score (no future data leakage)

Training:
  - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
  - Scheduler: CosineAnnealingWarmRestarts
  - Loss: Sharpe ratio loss (maximize risk-adjusted signal)
  - Batch size: 256
  - Max epochs: 100 with early stopping (patience=10)

Evaluation:
  - Information Coefficient (IC) = Spearman correlation(signal, forward_return)
  - ICIR = mean(IC) / std(IC)  →  target > 0.5
  - Signal decay: IC at lag 1h, 4h, 8h, 24h
```

### Regime Model (Days 29–34)

#### Architecture
```python
# models/regime/hmm.py
class RegimeHMM:
    """
    Gaussian HMM for latent market state detection.
    
    States: 4 (trending_up, trending_down, ranging, volatile)
    Features: [realized_vol, hurst_exponent, trend_strength, vpin]
    
    Outputs:
      - Current regime: one-hot (4,)
      - Transition probabilities: (4, 4)
      - Regime confidence: scalar [0, 1]
    
    Training: Baum-Welch algorithm (hmmlearn)
    Inference: Viterbi algorithm for most-likely state sequence
    """

# models/regime/lstm_classifier.py
class RegimeLSTM(nn.Module):
    """
    LSTM that uses HMM output as auxiliary input for regime classification.
    
    Input:  features (seq_len, n_features) + HMM state (seq_len, 4)
    Output: regime logits (4,) + transition probability matrix (4, 4)
    
    Why LSTM on top of HMM?
      HMM is a generative model — limited in capturing complex nonlinear
      transitions. LSTM adds discriminative power using the HMM state as
      a strong prior.
    """
```

### RL Agent (Days 35–45)

#### Environment Design
```python
# models/rl_agent/environment.py
class ForexTradingEnv(gym.Env):
    """
    OpenAI Gym-compatible Forex trading environment.
    
    STATE SPACE (continuous, ~165 dimensions):
      - Feature vector (150 features, normalized)
      - Portfolio state: [position (-1/0/1), unrealized_pnl, time_in_trade]
      - Regime state: one-hot (4,)
      - Session: one-hot (4,) [Asian, London, NY, overlap]
    
    ACTION SPACE (discrete, 5 actions):
      - 0: Flat (no position)
      - 1: Long 0.5 lot
      - 2: Long 1.0 lot
      - 3: Short 0.5 lot
      - 4: Short 1.0 lot
    
    REWARD FUNCTION (critical design):
      reward = pnl_t - λ * max_drawdown_penalty - γ * transaction_cost
      
      λ = 0.1  (penalize drawdown, not just raw PnL)
      γ = realistic spread + slippage
      
      Shaped reward during training:
        + bonus for correct direction in strong-signal regime
        + penalty for holding through news events (high VPIN)
    
    EPISODE DESIGN:
      - Length: 5000 steps (≈ 5 trading days at 1-min bars)
      - Reset: random start within training period
      - Curriculum: start on easy regimes (trending), add hard regimes
    """
```

#### Training
```
Algorithm: PPO (primary), SAC (secondary for comparison)

PPO hyperparameters:
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  ent_coef: 0.01  (encourages exploration)

Training schedule:
  Phase 1: 1M steps on trending regimes (curriculum start)
  Phase 2: 1M steps on all regimes (full difficulty)
  Phase 3: 500K steps fine-tuning on recent 1 year data

Evaluation: Every 50K steps, run on 3-month OOS data. Track:
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Profit factor
```

### Meta-Learner (Days 44–48)

```python
# models/meta_learner/maml.py
class MAMLAdapter:
    """
    Model-Agnostic Meta-Learning for rapid regime adaptation.
    
    Goal: Given 50 steps of new regime data, adapt in 5 gradient steps.
    
    Meta-training:
      - Sample task = random 2-week regime episode
      - Inner loop: 5 gradient steps on support set (first week)
      - Outer loop: meta-gradient on query set (second week)
    
    Live usage:
      - Every hour, run 5 adaptation steps on last 50 steps
      - Adapted weights override base model weights temporarily
      - Fallback to base weights if adapted performance < threshold
    
    This solves the biggest failure mode of quant models:
    they work until the regime changes, then they fail.
    MAML makes the model "expect to adapt."
    """
```

### Ensemble Aggregator (Days 48–50)

```python
# models/ensemble/aggregator.py
class EnsembleAggregator:
    """
    Combines signals from all 4 models into final alpha signal.
    
    Method 1: Stacking (primary)
      - Meta-model: LightGBM trained on model outputs → forward return
      - Features: [temporal_signal, regime_signal, rl_action_prob, 
                   meta_confidence, regime_state, session]
      - Trained with purged CV to prevent leakage
    
    Method 2: Bayesian Model Averaging (fallback when meta-model uncertain)
      - Prior: equal weights
      - Update: model weights ∝ recent IC (30-day rolling)
      - Posterior: normalized weights
    
    Uncertainty Quantification:
      - MC Dropout: 50 forward passes, compute predictive std
      - High uncertainty → reduce position size (via risk engine)
      - Uncertainty is passed as a first-class field in AlphaSignal
    
    Output: AlphaSignal dataclass
      direction:    Literal[-1, 0, 1]
      magnitude:    float [0, 1]       # signal strength
      confidence:   float [0, 1]       # model agreement
      uncertainty:  float              # MC Dropout std
      decay_steps:  int                # expected signal validity
      regime:       str                # current regime label
      timestamp:    datetime
    """
```

---

## Sprint 4 — Risk Engine (Days 51–60)

### Position Sizing
```python
# risk/sizing/kelly.py
"""
Kelly Criterion: f* = (p*b - q) / b
  p = win probability
  b = win/loss ratio
  q = 1 - p

In practice:
  - Use fractional Kelly: f_actual = 0.25 * f*
    (Reason: Kelly assumes known probabilities; we have estimates)
  - Hard cap: max 2% account risk per trade
  - Scaling: multiply by confidence from ensemble

Continuous Kelly (for size optimization):
  f* = μ / σ²  where μ = expected return, σ² = return variance
  
  Estimate μ and σ from rolling 60-bar window of signal returns.
"""
```

### Circuit Breakers
```python
# risk/limits/drawdown_limits.py
"""
Three-tier circuit breaker system:

Tier 1 (Yellow): Daily drawdown > 2%
  Action: Reduce all new position sizes by 50%
  Reset: Next trading day

Tier 2 (Orange): Daily drawdown > 3% OR weekly drawdown > 6%
  Action: Close all new entries. Manage existing positions only.
  Reset: Next trading day (daily), next Monday (weekly)

Tier 3 (Red): Monthly drawdown > 10%
  Action: Emergency stop. All positions closed. System halted.
  Reset: Manual review required. Retrain evaluation before restart.

Implementation: All order placement routes through CircuitBreaker.check()
before broker submission. Atomic — cannot be bypassed.
"""
```

---

## Sprint 5 — Backtesting & Validation (Days 61–70)

### Event-Driven Engine
```
backtesting/engines/event_driven.py:

Event types:
  - MarketEvent: new bar/tick received
  - SignalEvent: model generates signal
  - OrderEvent: risk engine approves order
  - FillEvent: simulated execution

Data handler → MarketEvent → Feature pipeline → 
  Signal → Risk engine → Order → Fill simulator → Portfolio update

This exact sequence mirrors the live system.
Backtest and live share the same signal, risk, and execution code.
Only the data source and broker differ.

Slippage model:
  slippage = base_spread + market_impact
  market_impact = kyle_lambda * sqrt(order_size / avg_volume)
  
  Add random noise: slippage += N(0, 0.1 * spread)
  This prevents backtest curves that are unrealistically smooth.
```

### Walk-Forward Validation
```
Protocol:
  Training window: 18 months
  Validation window: 3 months
  Step: 1 month
  
  Total walk-forward folds: depends on data length
  
  For each fold:
    1. Train all models on training window
    2. Evaluate ensemble on validation window (OOS)
    3. Record: Sharpe, Calmar, max_drawdown, IC, ICIR
  
  Pass criteria:
    - Mean OOS Sharpe > 1.0 across all folds
    - No fold with Sharpe < 0 (rules out lucky periods)
    - Mean(OOS_Sharpe) / Mean(IS_Sharpe) > 0.6 (not overfit)
```

---

## Sprint 6 — Paper Trading (Days 71–100)

### 30-Day Paper Trading Protocol
```
Week 1: Monitoring only
  - System running, signals generated, but no orders placed
  - Compare signal quality to backtest expectations
  - Monitor latency, data quality, error rates

Week 2-3: Paper trading active
  - All orders simulated (paper broker)
  - Full risk management active
  - Nightly reconciliation: positions, PnL, risk metrics

Week 4: Live comparison
  - Compare paper PnL to walk-forward OOS period
  - Analyze execution quality vs. backtest assumptions
  - Identify any live/backtest divergence

Pass criteria for live deployment:
  - Paper Sharpe > 1.5 over 30 days
  - No circuit breaker triggered
  - System uptime > 99.5%
  - Mean latency (tick → order) < 500ms
```

---

## Live Deployment Checklist

```
Infrastructure:
  [ ] All secrets in environment variables or secrets manager
  [ ] Database backups automated (hourly snapshots)
  [ ] Log rotation configured
  [ ] Alert channels configured (Slack, PagerDuty)
  [ ] Emergency stop procedure documented and tested

Models:
  [ ] All models loaded from versioned MLflow artifacts
  [ ] Model checksums verified on startup
  [ ] Fallback to conservative sizing if model load fails

Risk:
  [ ] All circuit breakers tested (unit + integration)
  [ ] Manual kill switch tested
  [ ] Position limits confirmed with broker

Execution:
  [ ] Broker API credentials tested
  [ ] Order rejection handling tested
  [ ] Partial fill handling tested
  [ ] Network timeout handling tested

Monitoring:
  [ ] All Grafana dashboards showing live data
  [ ] PagerDuty alerts verified
  [ ] Daily report email working
```
