# Professional FX Tick-Data Acquisition Specification

**Programme:** Forex Engin research-to-broker-demo readiness  
**Reference date:** 11 August 2026  
**Selected professional standard:** **LSEG Tick History**, using its normalised historical quote data and, where commercially available for the required FX venues, Tick History – PCAP for high-precision execution and microstructure research.  
**Deployment status:** **Research-only. No model, broker-demo, or live-trading approval is granted by this document.**

> This specification deliberately rejects generated prices, unverified public archives, and one-second aggregated data as substitutes for professional full-resolution bid/ask data in high-frequency validation.

## 1. Selection Rationale

LSEG Tick History is selected as the professional primary path because its official product documentation describes historical trades, quotes, and market-depth data; normalised or raw delivery; Web/API and cloud-delivery options; and custom extracts by instrument, field, and time period.[1] LSEG also describes its PCAP offering as direct exchange-source packet-capture data with GPS-synchronised nanosecond timestamps, a relevant capability for latency-sensitive microstructure research.[2]

Tick Data LLC is an acceptable independently sourced validation dataset, not a silent substitute. Its official FX product page describes millisecond-stamped tick-by-tick bid/ask quotes, more than 2,000 spot-FX pairs, and data from multiple contributors.[3] Where budget and licensing permit, use it as a held-out source for cross-source robustness checks.

| Role | Required source | Reason |
|---|---|---|
| Primary research data | LSEG Tick History normalised quotes, plus raw/PCAP only where venue-appropriate | Supports source-controlled timestamps, quote fields, venue metadata, and reproducible extracts. |
| Execution/microstructure research | LSEG Tick History – PCAP for available FX venues | Provides the high-precision venue feed required to test latency, message ordering, and queue assumptions. |
| Independent robustness check | Tick Data LLC spot-FX bid/ask quote history | Reduces the risk of treating one provider’s quote construction as universal FX truth. |
| Retail/public data | Prohibited from HFT readiness evidence | May only be used as clearly labelled exploratory research and can never satisfy this contract. |

## 2. Required Coverage

The initial licensed extract must cover **six complete calendar years: 1 January 2020 00:00:00 UTC through 31 December 2025 23:59:59.999999999 UTC**. This period exceeds the five-year minimum and includes distinct volatility, liquidity, and policy regimes. The initial currency universe is the seven G10 USD-major pairs, chosen for liquidity and practical coverage: `EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `USDCAD`, `AUDUSD`, and `NZDUSD`.

No claim that a signal generalises to every currency pair is permitted. Expansion to crosses and emerging-market pairs requires a separate coverage, liquidity, spread, and execution-cost acceptance review.

| Dataset partition | Purpose | Permitted use |
|---|---|---|
| 2020–2023 | Development and nested time-series model selection | Training only inside purged/embargoed folds. |
| 2024 | Locked validation regime | Calibration, threshold selection, and robustness checks only. |
| 2025 | Final holdout regime | One final out-of-sample evaluation after the development protocol is frozen. |
| Broker-demo feed | Forward live-data reconciliation | Required separately; never replaced by historical data. |

## 3. Minimum Record-Level Fields

The delivery must include the following fields or provider-specific equivalents, with a provider field dictionary. A missing mandatory field blocks the HFT/execution track; it may still be considered for non-HFT research only after an explicit documented exception.

| Field group | Mandatory fields | Acceptance rule |
|---|---|---|
| Identity | instrument/symbol, venue or contributor, source record identifier where available | Must retain provider symbology and an explicit Forex Engin mapping table. |
| Time | UTC event timestamp at native precision; ordering/sequence indicator where provided; source timezone declaration | No local-time assumptions. The raw timestamp is preserved without rounding. |
| Bid/ask quote | bid price, ask price, bid size, ask size, quote/update action, quote condition | `bid <= ask`; non-positive quotes/sizes, stale indicators, and crossed markets are audited and not silently repaired. |
| Trade/book context | last trade fields and Level 2/depth fields where licensed; trade/quote qualifiers | Required for PCAP/depth execution research; optional for baseline Level 1 signal research. |
| Provenance | vendor product/version, extract request ID, extract time, file-level checksum, licence scope | Each delivered file must be traceable to its exact licensed extract. |

## 4. Required Delivery Format and Access

Preferred delivery is **Parquet or Delta Parquet** with explicit schema and UTC timestamps. Compressed CSV may be accepted only when it includes an immutable schema document, RFC-compliant escaping, and file-level checksums. The provider must deliver through a user-controlled, read-only location such as an approved cloud bucket or signed export archive. Do **not** paste data-provider passwords, broker tokens, or access keys into messages.

| Delivery item | Requirement | Outcome if missing |
|---|---|---|
| Data files | One or more checksummed files with deterministic partition paths | Ingestion rejected. |
| Field dictionary | Provider definitions, units, timestamp semantics, condition codes | Data quarantined pending review. |
| Entitlement statement | Written confirmation that research, model training, and retained derived features are permitted | No ingestion or model training. |
| Coverage inventory | Pair, venue/contributor, start/end timestamps, rows, files | Coverage audit cannot begin. |
| Extract identifier | Request ID, job ID, or equivalent vendor lineage | Manifest marked incomplete and non-deployable. |

## 5. Ingestion Acceptance Gates

The existing manifest chain is extended with a professional-source profile. The ingestion process must calculate SHA-256 checksums at receipt, preserve the original vendor archive read-only, normalise to a canonical schema without overwriting raw data, and record every rejected row with a reason code.

| Gate | Pass criterion | Failure response |
|---|---|---|
| Licence gate | Documented right to research, train, validate, and retain permitted derivatives | Stop before data transfer. |
| Integrity gate | File checksum matches delivery metadata or a receipt checksum is recorded | Quarantine and re-acquire. |
| Schema gate | Mandatory fields map without ambiguous unit/time interpretation | Stop and request field clarification. |
| Time gate | Strictly monotonic event ordering within provider sequence constraints; UTC semantics verified | Quarantine affected partitions. |
| Quote gate | Crossed, zero, negative, duplicate, and stale quotes quantified by venue/day | No silent forward-fill, deletion, or repair. |
| Coverage gate | All seven pairs and full 2020–2025 range present, with documented gaps | Hold training authorization at `DENIED`. |
| Cross-source gate | Provider quote/spread behaviour reconciled against the independent source over sampled periods | No production-readiness claim. |

## 6. Operator Action Required

To proceed, provide **one** of the following, without sending secrets in plain chat:

1. A licensed LSEG Tick History export in a ZIP/Parquet archive through secure file upload.
2. A provider-managed signed, time-limited download link to the agreed six-year, seven-pair extract.
3. Read-only LSEG delivery access configured through a secure integration, after confirming the data licence permits this project’s research use.

The first action after access is granted will be a schema/entitlement/coverage audit. It will not train models and cannot submit broker orders.

## References

[1]: https://www.lseg.com/en/data-analytics/market-data/data-feeds/tick-history "LSEG Tick History"
[2]: https://www.lseg.com/en/insights/fx/revolutionising-fx-price-transparency-with-tick-history-pcap "LSEG Tick History – PCAP for FX"
[3]: https://www.tickdata.com/product/historical-forex-data/ "Tick Data LLC — Historical Forex Data"
