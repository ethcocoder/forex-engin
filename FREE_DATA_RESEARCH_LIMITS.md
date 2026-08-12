# Free FX Data Research Limits

**Reference date:** 12 August 2026  
**Selected source:** Dukascopy Historical Data Export  
**Scope:** Exploratory research only  
**Live-execution status:** Denied

## Decision

Forex Engin uses Dukascopy’s free Historical Data Export as the default no-cost source for real-data exploratory research. Dukascopy states that its historical export provides CSV data across timeframes from tick-by-tick to monthly and is available without charge.[1] This source is materially preferable to generated prices because every downloaded hour is retained as a raw archive, checksummed, decoded, quality-checked, and linked to a manifest.

> Free, broker-originated historical quotes are useful for exploratory model research. They are not evidence that an execution algorithm will receive equivalent latency, liquidity, fill, venue, or spread conditions from a future broker.

## Permitted and Prohibited Uses

| Activity | Status | Rationale |
|---|---|---|
| Feature engineering and leak detection | Permitted | The source provides actual historical bid/ask quotes after manifest verification. |
| Exploratory candidate training | Permitted | Every result must remain labelled `EXPLORATORY_RESEARCH_ONLY`. |
| Purged historical walk-forward analysis | Permitted | It can measure historical robustness only under disclosed data limitations. |
| Institutional execution-cost validation | Denied | The free source does not establish the execution environment, queue position, venue-specific latency, or fill behaviour. |
| Broker-demo authorization | Denied | A separate authenticated broker-demo feed and monitored trial are required. |
| Live trading authorization | Denied | Historical research cannot authorize future capital deployment. |

## Selected Alternatives

| Source | Access model | Role in this project |
|---|---|---|
| Dukascopy Historical Data Export | Free direct export | **Primary free exploratory research source.** |
| TrueFX | Free registration; top-of-book tick data | Optional independent cross-source robustness check. TrueFX describes millisecond-detail top-of-book tick data and free access after registration.[2] |
| HistData | Public archive pages, with paid automated access | Fallback only; its documented formats include tick data and, in some interfaces, one-second bid/ask data.[3] |
| LSEG Tick History | Licensed commercial data | Required later for institutional-grade quote/depth and execution research. |

## Dataset Controls

Each Dukascopy raw dataset manifest now carries the following mandatory authorization controls:

| Manifest field | Required value |
|---|---|
| `source_class` | `free_public_broker_historical_export` |
| `research_authorization` | `EXPLORATORY_RESEARCH_ONLY` |
| `institutional_execution_validation` | `DENIED` |
| `broker_demo_authorization` | `DENIED` |
| `live_trading_authorization` | `DENIED` |

Derived bar manifests preserve the same limits. The preparation pipeline refuses a source dataset that does not explicitly deny live-trading authorization. This removes any pathway by which a free source could be silently relabelled as institutional validation evidence.

## Evidence Still Required

The free path can advance research but cannot complete production readiness. Before broker-demo or live execution, the programme still requires a licensed data entitlement appropriate to the target venues, independent cross-source checks, explicit transaction-cost and slippage assumptions, an authenticated broker-demo feed, and a monitored forward trial.

## References

[1]: https://www.dukascopy.com/swiss/english/marketwatch/historical/ "Dukascopy Historical Data Export"
[2]: https://www.truefx.com/truefx-historical-downloads-2/ "TrueFX Historical Downloads"
[3]: https://www.histdata.com/download-free-forex-data/ "HistData Free Forex Historical Data"
