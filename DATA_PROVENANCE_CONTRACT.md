# Data Provenance Contract

**Status:** Required for every model-training, backtest, and paper-trading candidate produced after this document was introduced.

> A file is eligible for modelling only when it has a machine-verifiable manifest that identifies its real market-data source, time range, content hash, and quality controls. Synthetic, manually edited, unprovenanced, or hash-mismatched data are ineligible.

## Purpose

Forex Engin previously contained a helper that generated random, regime-styled price series and labelled them as industrial data. That is not a valid basis for training, performance claims, or trading-system readiness. The generator now stops deliberately; it cannot create a file accepted by the new provenance gate.

The repository now uses a reproducible path beginning with Dukascopy historical bid/ask tick archives. Dukascopy states that its Historical Data Export provides Forex price data in formats from tick-by-tick to monthly, and its API documentation distinguishes retrieval by time interval from shift-based retrieval.[1] [2] OANDA is an alternative licensed source when authenticated data access is required; it advertises historical and tick-level FX data, but its data service is not silently substituted here.[3]

## Required Evidence

| Stage | Required file | Immutable evidence | Rejection conditions |
|---|---|---|---|
| Raw ingestion | `data/raw/<PAIR>/*.bi5` | Source URL, compressed-archive SHA-256, requested UTC hour | HTTP failure, malformed BI5 record length, impossible timestamp |
| Validated tick chunk | `data/<PAIR>_ticks_*.csv` and `data/manifests/*.manifest.json` | Dataset SHA-256, source type, source reference, row count, interval coverage, per-hour validation | Crossed/negative/zero quotes, missing provenance, content hash mismatch |
| Derived bars | `data/processed/*.csv` and adjacent manifest | Parent tick dataset hash, output hash, bar frequency, instrument | Input manifest does not identify `real_historical_tick_data`; stale or modified source chunk |
| Training candidate | experiment report and signed model manifest | Derived-bar manifest hash, feature configuration, label horizon, split specification, OOS metrics | Synthetic source, missing purge/embargo, in-sample-only results, missing execution-cost assumptions |

## Reproducible Workflow

The downloader intentionally limits each execution to a bounded period. Chunking protects the source, supports retries, and lets a long historical ingestion remain resumable rather than pretending a multi-year archive was successfully fetched.

```bash
# Real tick-data sample: January 2, 2024, 00:00–06:00 UTC.
python3 scripts/download_dukascopy_ticks.py \
  --instrument EURUSD \
  --start 2024-01-02T00:00:00Z \
  --end 2024-01-02T06:00:00Z \
  --output data

# Derive one-minute bars only after the dataset hash has been verified.
python3 scripts/prepare_tick_bars.py \
  --ticks data/EURUSD_ticks_20240102T0000Z_20240102T0600Z.csv \
  --manifest data/manifests/EURUSD_ticks_20240102T0000Z_20240102T0600Z.manifest.json \
  --frequency 1min \
  --output data/processed/EURUSD_20240102_1min.csv
```

The full five-year requirement must be fetched in bounded chunks across all selected pairs and then independently checked for missing hours, duplicates, spread anomalies, time-zone consistency, and source-specific sessions. A two-hour real-data smoke test has been executed in this repository; it is an ingestion verification only, **not** evidence of alpha, liquidity coverage, or model readiness.

## Non-Negotiable Research Constraints

| Constraint | Implementation requirement |
|---|---|
| Time handling | Store and validate timestamps in UTC. Align all resampling boundaries explicitly. |
| Executability | Train and evaluate on bid/ask-aware data. Mid-price labels alone are insufficient for executable P&L. |
| Reproducibility | Preserve raw archive SHA-256 values and every derived-file hash. Never overwrite manifests to fit changed data. |
| Leakage control | Features use information available at or before the decision timestamp; labels begin only after the chosen horizon. Purge and embargo periods scale with feature lookback and holding horizon. |
| Cost realism | Backtests must charge observed spreads plus conservative slippage and carry/financing where relevant. Randomised spreads are prohibited in readiness evidence. |
| Deployment posture | Historical data and simulated evaluation do not authorise live orders. Broker-demo validation remains a separate mandatory gate. |

## Reference Sources

[1]: https://www.dukascopy.com/swiss/english/marketwatch/historical/ "Dukascopy Historical Data Export"
[2]: https://www.dukascopy.com/wiki/en/development/strategy-api/historical-data/history-ticks/ "Dukascopy Strategy API: History ticks"
[3]: https://www.oanda.com/foreign-exchange-data-services/en/exchange-rates-api/ "OANDA Exchange Rates API"
