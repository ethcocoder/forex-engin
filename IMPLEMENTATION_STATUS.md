# Forex Neural Trading Engine: Research-Core Implementation Status

**Status date:** 13 August 2026  
**Repository commit:** `690969e` — `Build causal forex research and validation core`  
**Scope:** Research, validation, and risk-control core only. Broker and paper-order integration remain deferred.

## Executive status

The repository now contains a functioning **research-stage FX system** focused on causal data handling, reproducible feature generation, leakage-resistant model training, cost-scenario backtesting, and hard readiness gates. The implementation is deliberately more conservative than the original marketing language: it treats the repository data as research input, not as evidence that a strategy is executable or profitable.

> **Current decision:** Neither the Ridge baseline nor the one-epoch temporal model passed the promotion gates. The correct next state is **research iteration required**, not broker-demo integration.

| Area | Delivered status | Evidence |
|---|---|---|
| Runtime and native acceleration baseline | Complete | C++ acceleration modules build from source; the full test suite passes. |
| Data contract | Complete | UTC, chronological, unique timestamps; positive/valid OHLCV; optional real bid/ask validation; immutable dataset manifests. |
| Labels | Complete | Explicit forward horizon plus one-bar decision-to-entry lag; unavailable tail labels remain null. |
| Features | Complete | Strict default uses OHLCV-derived technical, wavelet, and Kalman features only; no synthetic bid/ask, sentiment, macro, COT, options, or order-book fallbacks. |
| Baseline training | Complete | Ridge baseline uses approved features, fold-local scaling, expanding purged splits, OOS predictions, and saved lineage. |
| Temporal training | Complete | Sequence model uses fold-local scaling, causal sequences, expanding purged validation, saved model/scaler, and OOS diagnostics. |
| Backtest and risk evidence | Complete | Turnover-cost scenario, drawdown stop, equity/event audit, and persisted gate reports. |
| Broker/paper adapter | Deliberately deferred | Promotion gates block advancement because current OOS evidence is insufficient. |

## Delivered components

The new `research/` package establishes a dedicated research boundary. `contracts.py` validates real market-data schemas and creates hashes for the consumed data. `labels.py` builds log or simple forward-return labels with an explicit entry lag. `splits.py` provides expanding walk-forward splits that purge labels overlapping the validation interval. `training.py` trains a transparent Ridge baseline with approved, finite feature columns and persists the model, OOS predictions, metrics, dataset manifest, feature schema hash, random seed, and configuration.

The temporal workflow in `temporal.py` consumes the same data contract, builds causal feature windows, refits a scaler in each historical fold, saves final model/scaler artifacts, and records OOS metrics separately. This design prevents the validation period from influencing either normalisation or model fitting.

The cost-aware research backtest in `backtest.py` uses OOS predictions only. It records position changes, turnover, scenario transaction costs, gross/net returns, equity, drawdown, and circuit-breaker state. The model-readiness gate in `readiness.py` requires minimum out-of-sample prediction quality, positive cost-aware Sharpe, explicit nonzero cost assumptions, bounded drawdown, and correct non-execution labels before a model can become a **paper-candidate review** item. It never authorises live trading.

| New entry point | Purpose |
|---|---|
| `scripts/generate_features.py` | Generates strict, causal OHLCV-derived features and a data manifest. |
| `scripts/run_baseline_experiment.py` | Runs the leakage-resistant Ridge baseline and writes auditable artifacts. |
| `scripts/run_temporal_experiment.py` | Runs the leakage-safe temporal experiment and writes model/scaler artifacts. |
| `scripts/evaluate_oos_backtest.py` | Applies declared scenario costs to only saved OOS predictions. |
| `scripts/run_readiness_gates.py` | Generates the formal promotion-gate decision. |

## Validation evidence

The local test suite completed successfully after the implementation changes.

| Validation | Result |
|---|---:|
| Full test suite | **93 passed** |
| Runtime | 26.25 seconds |
| Native modules | Kalman, RL, and MAML C++ speedups compiled successfully from their C++ source files. |
| New coverage | Data contracts, labels, purged splits, causal sequences, experiment artifacts, cost/turnover accounting, drawdown halt, and promotion gates. |

## Model and backtest evidence

The repository’s EUR/USD CSV data were used to generate a fresh core-only OHLCV feature matrix. It contains 29 technical/wavelet/Kalman features and excludes all previously fabricated alternative or microstructure inputs. The label was one bar of forward return after an explicit one-bar entry lag. These are **research diagnostics**, not trade recommendations.

| Experiment | OOS rows | Directional accuracy | Information coefficient | Cost-scenario total return | Annualised Sharpe | Promotion result |
|---|---:|---:|---:|---:|---:|---|
| Ridge baseline | 6,168 prepared rows; 5,140 OOS rows | 46.87% | 0.0228 | -0.80% | -0.302 | Fail — research iteration required |
| Temporal model, 1 epoch | 6,168 prepared rows; 5,094 OOS rows | 45.56% | 0.0111 | -0.05% | -0.020 | Fail — research iteration required |

The cost scenario used a 0.5-basis-point half-spread plus 0.5-basis-point slippage assumption, 10% fractional position, and a 10% drawdown stop. These costs are stated assumptions because the source data do not contain executable broker bid/ask fills, commissions, financing, margin, or fill records. They must not be mistaken for actual broker costs.

## Why broker work is deferred

The readiness policy requires, at minimum, OOS information coefficient of 0.03, directional accuracy of 50%, cost-aware annualised Sharpe of 0.50, maximum drawdown no greater than 10%, nonzero explicit cost assumptions, and no incorrect `execution_ready` labels. Both current experiments failed the IC, directional-accuracy, and cost-aware Sharpe conditions. Progressing these models to a demo broker would be an engineering demonstration only and would not be professionally justified as a trading-system evaluation.

## Required next research cycle

The appropriate next iteration is to improve the **evidence**, not to add broker complexity. The research plan should first obtain a documented data source with bid/ask or transaction-level semantics, confirm the bar interval and availability time for every feature, and use a fixed holdout period not inspected during development. Next, compare simple no-trade, trend, mean-reversion, and volatility-targeting baselines against the temporal model using the same costs and turnover constraints. Every new feature family should be introduced through an ablation test with stable OOS improvement before it enters the production candidate schema.

If a future candidate passes the stated gates, the next engineering phase should add a paper-only adapter with broker-specific order-state reconciliation, idempotent client order IDs, partial-fill/reject handling, executable bid/ask accounting, margin/financing treatment, a kill switch, and an independent review of the live shadow-trading log. Live trading remains outside the scope of this implementation status.

## Reproduction commands

```bash
cd forex-engin
python3 scripts/compile_speedups.py
pytest -q

python3 scripts/generate_features.py \
  --input data/EUR_USD_ticks.csv \
  --output artifacts/research_data/EUR_USD_core_features.csv \
  --pair EUR_USD --provider repository_csv

python3 scripts/run_baseline_experiment.py \
  --raw data/EUR_USD_ticks.csv \
  --features artifacts/research_data/EUR_USD_core_features.csv

python3 scripts/evaluate_oos_backtest.py artifacts/experiments/<run-id> \
  --half-spread-bps 0.5 --slippage-bps 0.5

python3 scripts/run_readiness_gates.py artifacts/experiments/<run-id>
```

## Implementation boundary

This work is software and quantitative-research infrastructure. It is not investment advice, does not guarantee performance, and does not authorise live trading.
