# DOCUMENTATION
## Full Project Structure & Component Reference

---

## Folder Structure

```
forex-neural-engine/
│
├── README.md                        # Project overview & quick start
├── DOCUMENTATION.md                 # This file — full structure reference
├── ROADMAP.md                       # Strategic milestones & vision
├── IMPLEMENTATION_PLAN.md           # Phase-by-phase build plan
├── TASK_TODO.md                     # Granular task tracker
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Dev/test dependencies
├── pyproject.toml                   # Project metadata & build config
├── .env.example                     # Environment variable template
├── .gitignore
├── LICENSE
│
├── configs/                         # All configuration files
│   ├── config.example.yaml          # Main config template (copy to config.yaml)
│   ├── config.yaml                  # [gitignored] Your live config
│   ├── backtest.yaml                # Backtesting parameters
│   ├── paper.yaml                   # Paper trading parameters
│   ├── live.yaml                    # Live trading parameters [gitignored]
│   ├── models/
│   │   ├── temporal_model.yaml      # Transformer+TCN hyperparameters
│   │   ├── regime_model.yaml        # HMM+LSTM hyperparameters
│   │   ├── rl_agent.yaml            # PPO/SAC hyperparameters
│   │   └── meta_learner.yaml        # MAML hyperparameters
│   ├── risk/
│   │   ├── risk_limits.yaml         # CVaR, drawdown, exposure limits
│   │   └── position_sizing.yaml     # Kelly fraction, max position sizes
│   └── brokers/
│       ├── oanda.yaml               # OANDA API config template
│       ├── lmax.yaml                # LMAX config template
│       └── interactive_brokers.yaml # IB config template
│
├── data/                            # All data (mostly gitignored)
│   ├── raw/                         # Raw tick data as received
│   │   ├── ticks/                   # Per-pair tick CSV/Parquet files
│   │   ├── orderbook/               # Level 2 snapshots
│   │   └── economic_calendar/       # Macro event data
│   ├── processed/                   # Cleaned, normalized OHLCV
│   │   ├── 1m/
│   │   ├── 5m/
│   │   ├── 1h/
│   │   ├── 4h/
│   │   └── 1d/
│   ├── features/                    # Computed feature matrices
│   │   ├── microstructure/
│   │   ├── technical/
│   │   └── alternative/
│   ├── orderbook/                   # Processed order book features
│   ├── sentiment/                   # NLP-processed sentiment scores
│   └── macro/                       # Macro event surprise factors
│
├── models/                          # Model definitions & saved weights
│   ├── __init__.py
│   ├── base_model.py                # Abstract base class all models inherit
│   │
│   ├── temporal/                    # Temporal signal model
│   │   ├── __init__.py
│   │   ├── transformer.py           # Multi-head self-attention encoder
│   │   ├── tcn.py                   # Temporal Convolutional Network
│   │   ├── combined.py              # Transformer + TCN fusion
│   │   ├── trainer.py               # Training loop & callbacks
│   │   └── saved/                   # Saved .pt model weights [gitignored]
│   │
│   ├── regime/                      # Market regime detection
│   │   ├── __init__.py
│   │   ├── hmm.py                   # Hidden Markov Model (hmmlearn)
│   │   ├── lstm_classifier.py       # LSTM regime classifier
│   │   ├── combined.py              # HMM-informed LSTM
│   │   ├── trainer.py
│   │   └── saved/
│   │
│   ├── rl_agent/                    # Reinforcement learning agent
│   │   ├── __init__.py
│   │   ├── environment.py           # Gym-compatible Forex trading env
│   │   ├── ppo_agent.py             # Proximal Policy Optimization
│   │   ├── sac_agent.py             # Soft Actor-Critic
│   │   ├── reward_functions.py      # Risk-adjusted reward definitions
│   │   ├── trainer.py               # RL training orchestrator
│   │   └── saved/
│   │
│   ├── meta_learner/                # Fast adaptation / meta-learning
│   │   ├── __init__.py
│   │   ├── maml.py                  # Model-Agnostic Meta-Learning
│   │   ├── online_adapter.py        # Online learning wrapper
│   │   ├── trainer.py
│   │   └── saved/
│   │
│   └── ensemble/                    # Ensemble aggregation
│       ├── __init__.py
│       ├── aggregator.py            # Stacking + Bayesian model averaging
│       ├── uncertainty.py           # Uncertainty quantification (MC Dropout)
│       ├── weighting.py             # Dynamic confidence-based weighting
│       └── signal_generator.py      # Final alpha signal output
│
├── features/                        # Feature engineering pipeline
│   ├── __init__.py
│   ├── pipeline.py                  # Master feature pipeline orchestrator
│   ├── base_feature.py              # Abstract base feature class
│   │
│   ├── microstructure/              # Market microstructure features
│   │   ├── __init__.py
│   │   ├── spread.py                # Bid-ask spread, effective spread
│   │   ├── order_flow.py            # Order flow imbalance, trade direction
│   │   ├── vpin.py                  # Volume-synchronized PIN (toxicity)
│   │   ├── kyle_lambda.py           # Kyle's lambda (price impact)
│   │   ├── amihud.py                # Amihud illiquidity ratio
│   │   └── lob_features.py          # Full LOB depth, queue imbalance
│   │
│   ├── technical/                   # Technical & statistical features
│   │   ├── __init__.py
│   │   ├── volatility.py            # Realized vol, GARCH, Parkinson, Yang-Zhang
│   │   ├── momentum.py              # RSI, MACD, ROC, various lookbacks
│   │   ├── mean_reversion.py        # Z-score, Hurst exponent, ADF test
│   │   ├── trend.py                 # ADX, trend strength, slope features
│   │   └── volume.py                # Tick volume, volume delta, VPOC
│   │
│   ├── alternative/                 # Alternative data features
│   │   ├── __init__.py
│   │   ├── sentiment.py             # News NLP scores, social media
│   │   ├── cot_positioning.py       # COT report net positioning
│   │   ├── options_flow.py          # Put/call ratio, skew, term structure
│   │   └── macro_surprise.py        # Economic data surprise scoring
│   │
│   └── wavelet/                     # Frequency-domain features
│       ├── __init__.py
│       ├── decomposition.py         # Wavelet decomposition (PyWavelets)
│       ├── kalman_filter.py         # Kalman filter for noise reduction
│       └── spectral.py              # FFT-based spectral features
│
├── risk/                            # Risk management engine
│   ├── __init__.py
│   ├── risk_engine.py               # Master risk engine (gates all orders)
│   │
│   ├── sizing/                      # Position sizing models
│   │   ├── __init__.py
│   │   ├── kelly.py                 # Kelly criterion & fractional Kelly
│   │   ├── fixed_fractional.py      # Fixed % risk per trade
│   │   └── volatility_scaled.py     # Vol-adjusted position sizing
│   │
│   ├── limits/                      # Hard limit enforcement
│   │   ├── __init__.py
│   │   ├── cvar_limits.py           # Conditional Value-at-Risk limits
│   │   ├── drawdown_limits.py       # Max drawdown circuit breakers
│   │   ├── correlation_limits.py    # Cross-pair correlation exposure caps
│   │   ├── liquidity_filter.py      # Low-liquidity session gating
│   │   └── session_filter.py        # Trading session rules (Asian/London/NY)
│   │
│   └── monitoring/                  # Real-time risk monitoring
│       ├── __init__.py
│       ├── portfolio_monitor.py     # Live portfolio risk metrics
│       ├── pnl_attribution.py       # PnL breakdown by signal/pair/session
│       └── alert_manager.py         # Risk breach alerts & notifications
│
├── execution/                       # Order execution layer
│   ├── __init__.py
│   ├── execution_engine.py          # Master execution orchestrator
│   │
│   ├── brokers/                     # Broker API adapters
│   │   ├── __init__.py
│   │   ├── base_broker.py           # Abstract broker interface
│   │   ├── oanda_broker.py          # OANDA REST/Stream adapter
│   │   ├── lmax_broker.py           # LMAX FIX protocol adapter
│   │   ├── ib_broker.py             # Interactive Brokers adapter
│   │   └── paper_broker.py          # Paper trading simulator
│   │
│   ├── routing/                     # Smart order routing
│   │   ├── __init__.py
│   │   ├── smart_router.py          # Best execution router
│   │   ├── twap.py                  # Time-Weighted Average Price algo
│   │   ├── vwap.py                  # Volume-Weighted Average Price algo
│   │   └── iceberg.py               # Iceberg order logic
│   │
│   └── simulation/                  # Execution simulation
│       ├── __init__.py
│       ├── slippage_model.py         # Realistic slippage estimation
│       ├── fill_simulator.py         # Order fill simulation
│       └── market_impact.py          # Market impact estimation
│
├── backtesting/                     # Backtesting framework
│   ├── __init__.py
│   ├── engine.py                    # Core event-driven backtesting engine
│   ├── data_handler.py              # Historical data feed handler
│   ├── portfolio.py                 # Portfolio state tracking
│   ├── performance.py               # Performance analytics & reporting
│   │
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── vectorized.py            # Fast vectorized backtester (research)
│   │   └── event_driven.py          # Realistic event-driven backtester
│   │
│   ├── scenarios/                   # Stress test scenarios
│   │   ├── __init__.py
│   │   ├── monte_carlo.py           # Monte Carlo simulation
│   │   ├── walk_forward.py          # Walk-forward validation
│   │   ├── regime_stress.py         # Regime-specific stress tests
│   │   └── historical_stress.py     # Historical crisis replay (2008, 2020)
│   │
│   └── results/                     # [gitignored] Backtest output files
│
├── monitoring/                      # Live system monitoring
│   ├── __init__.py
│   ├── metrics_collector.py         # Prometheus metrics exporter
│   ├── signal_monitor.py            # Signal quality & decay tracking
│   ├── model_monitor.py             # Model performance vs. backtest
│   │
│   ├── dashboards/                  # Grafana dashboard JSON configs
│   │   ├── pnl_dashboard.json
│   │   ├── risk_dashboard.json
│   │   ├── model_dashboard.json
│   │   └── execution_dashboard.json
│   │
│   ├── alerts/                      # Alert rule definitions
│   │   ├── risk_alerts.yaml
│   │   ├── model_alerts.yaml
│   │   └── execution_alerts.yaml
│   │
│   └── reporting/                   # Automated report generation
│       ├── __init__.py
│       ├── daily_report.py          # End-of-day summary report
│       └── performance_report.py    # Weekly/monthly performance report
│
├── research/                        # Research & experimentation
│   ├── notebooks/                   # Jupyter notebooks (exploration only)
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_feature_research.ipynb
│   │   ├── 03_model_experiments.ipynb
│   │   ├── 04_regime_analysis.ipynb
│   │   ├── 05_signal_decay_analysis.ipynb
│   │   └── 06_execution_analysis.ipynb
│   │
│   ├── experiments/                 # MLflow experiment results
│   │   └── .gitkeep
│   │
│   └── papers/                      # Reference papers & notes
│       ├── README.md                # Index of relevant papers
│       └── notes/                   # Your research notes
│
├── infrastructure/                  # Infrastructure as code
│   ├── docker/
│   │   ├── Dockerfile               # Main application Dockerfile
│   │   ├── Dockerfile.backtest      # Backtesting image
│   │   └── docker-compose.yml       # Full stack compose file
│   │
│   ├── kubernetes/                  # K8s deployment manifests
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── hpa.yaml                 # Horizontal pod autoscaler
│   │
│   ├── kafka/                       # Kafka topic configurations
│   │   ├── topics.yaml
│   │   └── consumer_groups.yaml
│   │
│   └── database/                    # Database schemas & migrations
│       ├── timescaledb/
│       │   ├── schema.sql           # TimescaleDB hypertable schema
│       │   └── migrations/
│       └── redis/
│           └── redis.conf
│
├── tests/                           # Test suite
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   ├── test_risk_engine.py
│   │   └── test_execution.py
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   ├── test_model_pipeline.py
│   │   └── test_broker_adapters.py
│   └── e2e/
│       ├── test_paper_trading.py
│       └── test_backtest_pipeline.py
│
├── scripts/                         # CLI entry points
│   ├── run_backtest.py              # Run a full backtest
│   ├── run_paper_trading.py         # Start paper trading
│   ├── run_live_trading.py          # Start live trading [careful!]
│   ├── train_models.py              # Train all models
│   ├── generate_features.py         # Build feature matrices
│   ├── download_data.py             # Pull historical data
│   └── health_check.py             # System health check
│
├── docs/                            # Extended documentation
│   ├── api/
│   │   └── API_REFERENCE.md         # Module & class API reference
│   ├── architecture/
│   │   ├── ARCHITECTURE.md          # Deep dive into system design
│   │   ├── DATA_FLOW.md             # Data flow diagrams
│   │   └── MODEL_DESIGN.md          # Neural architecture decisions
│   └── research/
│       ├── RESEARCH_NOTES.md        # Findings from research phase
│       ├── FEATURE_ANALYSIS.md      # Feature importance & correlation analysis
│       └── SIGNAL_ANALYSIS.md       # Signal quality & decay analysis
│
└── logs/                            # [gitignored] Runtime logs
    ├── trading/
    ├── models/
    └── system/
```

---

## Key Design Decisions

### Why PyTorch over TensorFlow?
Dynamic computation graphs are essential for research flexibility. The meta-learner (MAML) requires second-order gradients that are trivial in PyTorch but complex in TF.

### Why TimescaleDB over InfluxDB?
TimescaleDB is built on PostgreSQL — standard SQL queries, ACID compliance, and easy joins with relational data (positions, trades, events). InfluxDB is faster for pure time-series but harder to integrate with relational models.

### Why Kafka over RabbitMQ?
Kafka's log-based architecture allows replaying tick streams for backtesting without separate infrastructure. The same pipeline that processes live ticks can replay historical ticks.

### Why event-driven backtesting?
Vectorized backtesting is fast but hides execution realism (slippage, partial fills, latency, spread). Event-driven backtesting mirrors the live system's actual code path, reducing the gap between backtest and live performance.

### Why a separate Regime model?
Markets alternate between trending, mean-reverting, and high-volatility regimes. A single model trained on all regimes learns average behavior across regimes — which is suboptimal in all of them. Regime-conditional models specialize.

---

## Data Flow Summary

```
Broker WebSocket
      │
      ▼
  Kafka Topic (raw ticks)
      │
      ▼
  Feature Pipeline (Redis pub/sub)
      │
      ▼
  Neural Ensemble (inference server)
      │
      ▼
  Alpha Signal Object
      │
      ▼
  Risk Engine (gate + size)
      │
      ▼
  Execution Engine
      │
      ▼
  Broker Order API
      │
      ▼
  Fill Callback → Portfolio State → Monitoring
```

---

## Configuration Hierarchy

```
configs/config.yaml          ← Master config (all defaults here)
    ↓ overridden by
configs/live.yaml            ← Live trading overrides
    ↓ overridden by
Environment variables        ← Secrets (API keys, passwords)
```

Never put secrets in YAML files. Use environment variables or a secrets manager (HashiCorp Vault, AWS Secrets Manager).
