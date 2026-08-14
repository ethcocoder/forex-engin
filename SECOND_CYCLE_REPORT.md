# Second Research Cycle: Eligibility and Low-Turnover Robustness

**Status date:** 14 August 2026  
**Scope:** Research and no-order shadow simulation only.  
**Broker connection or order action:** None.

## Executive conclusion

The second validation cycle strengthened the system but **did not establish a paper-trading candidate**. The main result is negative in the professionally useful sense: the small positive return found at a very high signal threshold is concentrated entirely in the first chronological third of the held-out sample and has only 60 active observations. It is a post-hoc sensitivity result, not stable evidence of an edge.

> **Decision:** Continue research iteration. Do not connect a broker, open a paper position, or treat the latest model classification as a trading instruction.

## New controls

The research contract now separates two questions that should not be conflated. `MarketDataContract` confirms that a dataset is well formed. `MarketDataEligibilityPolicy` confirms whether it is adequate for an execution-like simulation. The latter requires enough history, executable bid/ask quotes, and observed positive volume.

| Control | Fresh public EUR/USD data result |
|---|---|
| Rows available | 6,143 hourly observations |
| UTC and chronological bars | Passed |
| Minimum 256-bar history | Passed |
| Executable bid/ask quotes | Failed — absent |
| Observed positive FX volume | Failed — all reported volume is zero |
| Execution-like data eligibility | **Failed** |

The source was the public [Yahoo Finance EUR/USD feed](https://finance.yahoo.com/quote/EURUSD=X). Its bars were useful for a pipeline smoke test but not for fill, spread, transaction-cost, margin, financing, or order-state evaluation. The shadow command now consumes the central eligibility report and records the exact blocking reasons in its no-order audit output.

## Threshold-sensitivity diagnostic

The Ridge model’s existing held-out predictions were evaluated using fixed thresholds and the previously declared scenario cost assumption: 0.5 basis points half-spread plus 0.5 basis points slippage, 10% fractional exposure, and a 10% drawdown stop. This is explicitly marked **post-hoc only**. It cannot select a policy for paper or live use because all thresholds are being inspected on previously held-out data.

| Absolute signal threshold | Active observations | Total return | Annualised Sharpe | Assessment |
|---:|---:|---:|---:|---|
| 0.00000 | 5,140 | -0.797% | -0.302 | Fails |
| 0.00010 | 1,651 | -0.416% | -0.254 | Fails |
| 0.00025 | 324 | -0.112% | -0.143 | Fails |
| 0.00050 | 60 | +0.046% | +0.127 | Insufficient and unstable |
| 0.00100 | 1 | -0.003% | -0.294 | Insufficient |
| 0.00200 | 0 | 0.000% | 0.000 | No trading evidence |
| 0.00500 | 0 | 0.000% | 0.000 | No trading evidence |

The apparent improvement at 0.00050 is not robust. A three-period chronological analysis shows all 60 active observations and the entire +0.046% return in the first segment, from 23 July to 31 October 2025. The next two chronological segments contained zero active observations. This violates the requirement for persistent evidence across time and remains below the readiness policy’s annualised-Sharpe threshold of 0.50.

| Chronological period | Active observations | Total return | Annualised Sharpe |
|---|---:|---:|---:|
| 23 Jul 2025 – 31 Oct 2025 | 60 | +0.046% | +0.220 |
| 31 Oct 2025 – 11 Feb 2026 | 0 | 0.000% | 0.000 |
| 11 Feb 2026 – 22 May 2026 | 0 | 0.000% | 0.000 |

## Quality evidence

The full repository test suite completed after this cycle with **96 passing tests**. Added tests cover execution-style data eligibility, post-hoc threshold-labelling, monotonic reduction in active exposure when thresholds rise, and chronological subperiod accounting. The fresh-data shadow observation was rerun with the central policy and remained blocked, with no broker connection or order action.

## Delivered implementation

| File or component | Purpose |
|---|---|
| `research/contracts.py` | Adds executable-simulation eligibility policies and audit reports. |
| `scripts/run_shadow_inference.py` | Uses the central eligibility policy and has no broker, execution-engine, or order-router path. |
| `research/backtest.py` | Adds post-hoc threshold sensitivity and chronological subperiod robustness diagnostics. |
| `scripts/evaluate_threshold_sensitivity.py` | Persists a threshold-sensitivity table and labels it non-promotional. |
| `scripts/evaluate_policy_subperiods.py` | Tests one fixed policy across chronological OOS segments. |

## Next research requirement

The next substantive step is not more threshold tuning. It is a new data and validation design: obtain a source with timestamped executable bid/ask quotes and observed volume, define a threshold on a development/validation interval, and test it once on a completely untouched final holdout. The candidate must then show sufficient active observations, stable performance across chronological subperiods, positive cost-aware results, and pass the existing promotion gates before a paper-only broker adapter is considered.

This is quantitative-research infrastructure, not investment advice, and no live-trading authority is included or implied.
