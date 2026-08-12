# Free-Data Exploratory Smoke Evidence

**Run date:** 12 August 2026  
**Dataset status:** `EXPLORATORY_RESEARCH_ONLY`  
**Deployment status:** `DENIED`

## Dataset Lineage

The exploratory dataset was built from eight contiguous, manifest-verified one-hour EUR/USD tick chunks from Dukascopy’s Historical Data Export. The consolidated source contains **19,607 validated ticks** spanning **2024-01-02 00:00:01.340 UTC** through **2024-01-02 07:59:59.947 UTC**. Its consolidated SHA-256 is `aff13a3cfcb89b9c0b72fb419cb94bb14dfee32c4740803f13e4218d157811a3`.

The raw-source and derived-bar manifests each record `EXPLORATORY_RESEARCH_ONLY`, with institutional execution validation, broker-demo authorization, and live-trading authorization all set to `DENIED`. The one-minute bar output contains **480** complete bars and is traceable to the full eight-file manifest chain.

| Control | Recorded value |
|---|---|
| Provider | Dukascopy Historical Data Export |
| Instrument | EUR/USD |
| Free-source class | `free_public_broker_historical_export` |
| Tick count | 19,607 |
| Bar frequency | One minute |
| Bar count | 480 |
| Institutional execution validation | Denied |
| Broker-demo authorization | Denied |
| Live-trading authorization | Denied |

## Purged Walk-Forward Smoke Result

A deliberately constrained baseline was evaluated using three chronological folds, a 60-bar purge, a 10-bar embargo, executable bid/ask returns, and no synthetic data. It is a **diagnostic smoke test**, not model-selection evidence: the sample is shorter than one trading day and cannot establish stability across regimes.

| Metric | Result |
|---|---:|
| Successful folds | 3 |
| Mean balanced accuracy | 36.79% |
| Mean macro F1 | 31.25% |
| Mean trade coverage | 78.53% |
| Aggregate executable cumulative return | -0.7582% |
| Mean executable return per bar | -0.002439% |
| Research status | `RESEARCH_ONLY` |

The result is **negative** and therefore rejects any claim of deployable alpha or a 90% win rate. It demonstrates that the provenance, feature, purge, embargo, executable-return, and report-lock controls operate on genuine free-source historical ticks without manufacturing an attractive outcome.

## Next Research Requirement

The immediate requirement is materially broader free coverage across dates, volatility regimes, and the initial major-pair universe. Even if a future free-data study produces positive historical findings, free-source results cannot authorize execution; licensed multi-venue data and a separate authenticated broker-demo forward trial remain mandatory.

## Reference

[1]: https://www.dukascopy.com/swiss/english/marketwatch/historical/ "Dukascopy Historical Data Export"

## Expanded 23-Hour Check

The same locked workflow was subsequently run on a contiguous 23-hour EUR/USD span ending at 23:00 UTC, with the provider-unavailable final hour preserved as an explicit gap rather than imputed. The consolidated sample contained **100,431 validated ticks** and **1,380** complete one-minute bars. Its SHA-256 was `df5246da1215c629962bdf4aa8303099582adac40418274e2962e6f186c70bd7`.

The four-fold purged evaluation remained negative: mean balanced accuracy was **34.96%**, mean macro F1 was **31.83%**, mean executable return per bar was **-0.002402%**, and aggregate executable cumulative return was **-2.4963%**. The expanded result corroborates the original rejection of any deployable-alpha claim. It remains `RESEARCH_ONLY` with broker-demo and live trading denied.

## Coverage Gate Status

A manifest-driven coverage audit enforces a **90-day local exploratory-research minimum** before candidate training may be considered. The present EUR/USD collection contains one contiguous **23-hour** span across 23 individually hash-verified manifests. It therefore reports `INSUFFICIENT_COVERAGE` and `training_authorization: DENIED`. The 90-day threshold is a local data-sufficiency gate only; satisfying it would still not validate execution quality or permit broker-demo or live orders.
