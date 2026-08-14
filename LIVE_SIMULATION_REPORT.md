# Fresh-Data No-Order Simulation Report

**Simulation timestamp:** 14 August 2026, 16:45 UTC  
**Mode:** `NO_ORDER_SHADOW_INFERENCE`  
**Broker connection:** None  
**Order action:** None

## Outcome

A fresh-data, live-style observation was completed without connecting to a broker, creating a position, or routing an order. The system acquired EUR/USD hourly bars from the public [Yahoo Finance EUR/USD feed](https://finance.yahoo.com/quote/EURUSD=X), regenerated the strict causal feature matrix, and evaluated the latest finite feature vector with the saved Ridge research model.

> **The observation was blocked.** The model generated a hypothetical `SHORT` classification, but no paper position was opened and no P&L was recorded because the model has failed its research-promotion gates and the fresh source does not supply executable bid/ask quotes or usable FX volume.

| Item | Observed value |
|---|---:|
| Fresh source | Yahoo Finance public EUR/USD hourly bars |
| Source coverage | 14 Aug 2025 16:00 UTC to 14 Aug 2026 16:00 UTC |
| Source bars | 6,143 |
| Generated causal features | 29 |
| Latest observed close | 1.15780950 |
| Latest feature timestamp | 14 Aug 2026 16:00 UTC |
| Model run | `ridge-20260813T184038Z-c9e5886f` |
| Model prediction | -0.00842404 |
| Hypothetical direction | `SHORT` |
| Actual action | **None** |

## Controls that blocked a simulated position

The shadow-inference command has no broker, execution-engine, or order-router import. It produces only an audit record. The following controls prevented even a hypothetical position ledger entry.

| Blocking control | Result |
|---|---|
| Research-promotion gate | Failed: prior OOS information coefficient, directional accuracy, and cost-aware Sharpe were below minimum policy thresholds. |
| Executable quote requirement | Failed: the fresh source contains no bid/ask quotes. |
| Observed-volume requirement | Failed: the fresh source reports zero FX volume; the downloader preserved zeros rather than fabricating pseudo-volume. |
| Broker/order interface | Not present in this command. |

The source’s lack of bid/ask and volume means it can support a **data-pipeline smoke test only**. It cannot provide the fill, spread, slippage, financing, margin, partial-fill, or order-status evidence required for a paper-trading simulation. The historical OOS result also remains negative under the documented scenario cost assumptions, so treating the raw model classification as a trade instruction would be inappropriate.

## Verification

The causal-feature generation completed successfully against the fresh source, with the native Kalman acceleration module loaded. The repository’s complete automated suite was rerun after adding the shadow path and completed with **93 passing tests**.

## Reproduction

```bash
cd forex-engin
pip install -r requirements.txt

python3 scripts/download_data.py \
  --source yfinance --pair EUR_USD --years 1 \
  --output artifacts/live_sim/EUR_USD_yahoo_1h.csv

python3 scripts/generate_features.py \
  --input artifacts/live_sim/EUR_USD_yahoo_1h.csv \
  --output artifacts/live_sim/EUR_USD_yahoo_1h_core_features.csv \
  --pair EUR_USD --provider yahoo_finance_public

python3 scripts/run_shadow_inference.py \
  --raw artifacts/live_sim/EUR_USD_yahoo_1h.csv \
  --features artifacts/live_sim/EUR_USD_yahoo_1h_core_features.csv \
  --model-experiment artifacts/experiments/ridge-20260813T184038Z-c9e5886f \
  --output artifacts/live_sim/shadow_inference_report.json
```

## Next requirement for a genuine paper simulation

A future paper-only simulation should use a data source that provides timestamped executable bid/ask prices and, if the model retains volume-dependent features, valid observed tick volume. It must also use a model that first passes the existing OOS, cost-aware, and drawdown promotion gates. No live-trading authority is included or implied by this simulation.
