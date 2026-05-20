# API REFERENCE
## Forex Neural Trading Engine — Component API Reference

This document provides python API references and signatures for core components, interfaces, and configurations.

---

## 1. Configuration Module (`configs/loader.py`)

### `load_config(config_path: str) -> AppConfig`
Loads, merges, and validates configuration settings from `config.yaml`, merges modular files from subdirectories, applies environment variable overrides, and returns an `AppConfig` instance.

### `AppConfig`
Pydantic schema representing the full configuration state:
- **`environment`**: `Literal["development", "paper", "live"]`
- **`pairs`**: `List[str]`
- **`database`**: `DatabaseConfig` (TimescaleDB + Redis details)
- **`kafka`**: `KafkaConfig` (bootstrap servers, consumer group, topics)
- **`features`**: `FeaturesConfig`
- **`models`**: `ModelsConfig`
- **`risk`**: `RiskConfig`
- **`execution`**: `ExecutionConfig`

---

## 2. Feature Engineering Module (`features/`)

### `BaseFeature`
Abstract base class for all feature calculations.
```python
class BaseFeature(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized batch calculation for training."""
        pass

    @abstractmethod
    def calculate_incremental(self, tick: dict) -> float:
        """Stateful real-time calculation for live trading."""
        pass
```

### `FeaturePipeline`
Orchestrator for feature calculation.
- **`calculate_batch(df: pd.DataFrame) -> pd.DataFrame`**: Returns a DataFrame of all features aligned to the index.
- **`process_tick(tick: dict) -> dict`**: Evaluates new ticks and updates the live feature cache.

---

## 3. Ensemble Module (`models/ensemble/`)

### `EnsembleAggregator`
Pydantic-wrapped model managing stacking models and BMA averages.
- **`register_model(name: str, model: BaseModel)`**: Registers a neural/regime sub-model.
- **`fit(X: pd.DataFrame, y: pd.Series)`**: Optimizes the LightGBM meta-learner using purged walk-forward cross-validation.
- **`predict(X: np.ndarray, return_signal: bool = False) -> Union[float, AlphaSignal]`**: Aggregates predictions. Generates an `AlphaSignal` if requested.

### `AlphaSignal`
Immutable data structure defining prediction properties:
- **`direction`**: `int` ($-1$ for Short, $0$ for Flat, $+1$ for Long)
- **`magnitude`**: `float` (strength bound $[0.0, 1.0]$)
- **`confidence`**: `float` (MC Dropout derived confidence score)
- **`uncertainty`**: `float` (Predictive standard deviation)
- **`expected_decay_steps`**: `int` (Predicted decay half-life)
- **`regime`**: `int` (Current Hidden Markov Model state index)
- **`timestamp`**: `float` (Generation time in epoch seconds)
- **`metadata`**: `dict` (Raw predictions from all sub-models)

---

## 4. Risk Engine Module (`risk/`)

### `RiskEngine`
Executes pre-trade validation checks and scales positions.
- **`gate(signal: AlphaSignal, portfolio: PortfolioState) -> Optional[OrderRequest]`**: Gates a signal. Returns an `OrderRequest` if it passes all filters, or `None` if rejected.

#### Internal Components:
- **`KellyPositionSizer`**: Computes optimal leverage using Kelly criterion scaled by signal uncertainty.
- **`DrawdownLimits`**: Drawdown tracking circuit breakers (daily, weekly, monthly thresholds).
- **`CVaRFilter`**: Limits portfolio Exposure if Value-at-Risk exceeds configured risk budgets.
- **`LiquidityFilter`**: Rejects orders if trade size exceeds a percentage of market volume.

---

## 5. Execution Engine Module (`execution/`)

### `ExecutionEngine`
Manages order routing, order tracking, and broker communication.
- **`submit_order(request: OrderRequest) -> OrderFill`**: Routes orders to execution routers (Iceberg/VWAP) and executes them.

#### Broker Adapters (`execution/brokers/`):
- **`OandaBroker`**: REST-based client for OANDA broker operations.
- **`IBBroker`**: Asynchronous adapter communicating via `ib_insync` with Interactive Brokers.
- **`LMAXBroker`**: FIXED-FIX 4.4 protocol client communicating via TCP sockets.

---

## 6. Risk Monitoring Module (`monitoring/`)

### `PortfolioMonitor`
Real-time tracking of portfolio risk health.
- **`update(portfolio: PortfolioState)`**: Re-evaluates leverage, single-asset concentration, and Parametric VaR. Dispatches alert triggers to `AlertManager`.

### `AlertManager`
Thread-safe notification throttler.
- **`trigger_alert(metric: str, level: AlertLevel, message: str)`**: Evaluates alerts and logs/notifies if the metric has cooled down since its last notification.

### `PnLAttribution`
Attributes trade results to execution factors.
- **`attribute_trade(trade: TradeRecord) -> PnLDecomposition`**: Decomposes trade result into Gross Alpha (ideal entry/exit), Slippage Drag, and Spread cost.
