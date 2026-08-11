# Forex Engin — Safety Hardening and Research-Readiness Report

**Branch:** `elite10x-pr`  
**Commit:** `03a16e6` — *Harden research data and execution safety gates*  
**Status:** **Research-only; live trading and broker-demo execution remain locked.**

> **Financial disclaimer:** This is engineering and research analysis, not guaranteed financial advice. Trading involves material risk of loss, and no historical or simulated result predicts future performance.

## Executive Summary

This implementation corrects a critical readiness problem: the repository had paths that could label generated price series as industrial data, create placeholder model artifacts, and describe a local historical replay with randomized execution assumptions as real-time paper trading. Those paths are now blocked or replaced with provenance-gated research tooling. The desktop app now shows a fail-closed research state rather than treating stored preferences or API fields as authenticated broker connectivity.

The new workflow downloads actual bid/ask tick archives from a declared source, stores immutable source and derived-data hashes in manifests, rejects invalid quotes, and permits downstream bars and research evaluation only when the manifest chain is intact. Dukascopy states that its Historical Data Export covers data from tick-by-tick to monthly intervals, while its historical tick documentation supports retrieval by time interval.[1] [2]

## Delivered Controls

| Control area | Implemented change | Result |
|---|---|---|
| Data provenance | Added `scripts/download_dukascopy_ticks.py` with BI5 decoding, UTC controls, source URLs, SHA-256 hashing, invalid-quote rejection, per-hour manifest evidence, retry logic, and resumable archive reuse. | A model input can now be traced to a real source archive rather than a generated CSV. |
| Training-input gate | Added `scripts/prepare_tick_bars.py`, which validates the tick manifest and checksum before generating bid/ask-aware bars and a derived manifest. | Modified or unprovenanced data are rejected before feature construction. |
| Synthetic-data prevention | Replaced `generate_industrial_data.py` with a deliberate failure; replaced the placeholder retraining helper with a provenance-only gate. | Neither script can generate or save a model artifact that falsely implies real-data training. |
| Time-series validation | Rebuilt `models/feature_pipeline.py` and `models/train_harness.py` around executable bid/ask labels, label-end timestamps, expanding purged walk-forward folds, and explicit non-deployable results. | Evaluation now prevents a training label from crossing into the future test period. |
| Historical research runner | Added `scripts/run_research_validation.py`, which verifies the full tick-to-bar manifest chain and writes a report with `deployment_authorization: DENIED`. | Historical validation cannot be confused with paper or live trading. |
| Execution controls | `run_live_trading.py` fails before loading credentials or a broker. The legacy “real paper trading” entry point fails before creating a paper broker or filling an order. | No local command path now claims or initiates live execution. |
| Desktop interface | Electron status now remains `RESEARCH_ONLY`; saved preferences are non-operative; credential inputs are disabled; live provider choices are removed; fabricated 92%/100% win-rate displays were replaced. | The UI no longer conveys unsupported performance or authenticated connection status. |
| Regression coverage | Added tests for real-data decoding, crossed-quote rejection, checksum gates, no forward-filled bars, purged label boundaries, fail-closed execution, and desktop-compatible safety logic. | The focused suite completed with **20 passing tests**. |

## Real-Data Smoke Test

A bounded local smoke test was executed against EUR/USD data for **2024-01-02 00:00–03:00 UTC**, using the new downloader and transformation path. It retrieved **6,503 validated ticks** and produced **180 one-minute bars**. The generated local manifest recorded each source URL and archive hash; the raw data and local reports were deliberately excluded from source control to avoid committing downloaded market data.

The non-deployable walk-forward smoke evaluation used two chronological folds, feature windows of 5 and 15 bars, a one-bar executable label horizon, and a 15-bar purge. It produced an aggregate historical executable return of **-0.00013594845183884097**, mean balanced accuracy of **0.3469516594516595**, and mean macro-F1 of **0.28328282828282825**. This small test is **negative and inadequate by design**: it demonstrates that the new pipeline reports weak evidence honestly rather than manufacturing a 90%+ outcome.

| Smoke-test property | Observed value | Readiness interpretation |
|---|---:|---|
| Source | Dukascopy EUR/USD historical tick archive | Source provenance recorded locally; not a broker-demo feed. |
| Validated ticks | 6,503 | Insufficient for a five-year multi-regime programme. |
| Derived one-minute bars | 180 | Insufficient for production model training. |
| Out-of-sample folds | 2 | Smoke-test only; not a research validation programme. |
| Aggregate executable return | -0.00013594845183884097 | No demonstrated alpha. |
| Deployment authorization | Denied | Correct outcome; no broker-demo or live action is permitted. |

## Verification Completed

The following checks completed successfully before commit and push:

| Check | Result |
|---|---|
| Tick-data provenance unit tests | 5 passed |
| Purged walk-forward validation and provenance tests | 8 passed |
| Execution lock tests | 2 passed |
| Focused total, including existing execution test | **20 passed** |
| Python syntax compilation of modified runtime modules | Passed |
| Electron main-process syntax check | Passed |
| TypeScript/Vite production build | Passed using local build binaries |
| Git whitespace verification | Passed |
| GitHub push | `elite10x-pr` updated to `03a16e6` |

The package-manager wrapper reported an ignored optional dependency build policy for `electron-winstaller`; direct local TypeScript/Vite compilation succeeded. This is an environment/package-policy issue rather than an application compilation failure.

## Readiness Decision

**Do not enable broker authentication, paper-order submission, or live execution.** The system is safer and more auditable after this work, but it has not met the empirical gates for a trading deployment. The source downloader and evaluation harness are infrastructure for evidence gathering, not evidence of an investable strategy.

The next evidence-producing sequence is therefore: ingest five or more years of manifest-verified tick data across the approved currency universe; run feature, label, and data-quality audits; train candidates only with chronological purge/embargo controls; evaluate post-cost performance across independently held-out regimes; then conduct a separate authenticated broker-demo trial with reconciliation and monitoring. The existing master readiness plan remains the governing release contract.

## Sources

[1]: https://www.dukascopy.com/swiss/english/marketwatch/historical/ "Dukascopy Historical Data Export"
[2]: https://www.dukascopy.com/wiki/en/development/strategy-api/historical-data/history-ticks/ "Dukascopy Strategy API — History ticks"
[3]: https://www.oanda.com/foreign-exchange-data-services/en/exchange-rates-api/ "OANDA Exchange Rates API"
