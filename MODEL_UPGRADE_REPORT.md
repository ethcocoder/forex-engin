# Hardened Model-Research Upgrade Report

**Status date:** 14 August 2026  
**Scope:** Causal model research, no-order inference, and validation controls.  
**Broker connection / order action:** None.

## Executive conclusion

The modelling layer has been materially upgraded from an ad hoc, partially bypassed ensemble path to a **cross-fitted, schema-validated, uncertainty-aware research ensemble**. The new implementation is stronger engineering, not a promise of a profitable strategy. On the repository’s EUR/USD research data, it correctly concluded that it did not have sufficient confidence to act: every held-out forecast was abstained after calibrated uncertainty was applied.

> **Current decision:** The model is more reliable as software because it is willing to abstain. It has not earned paper-trading eligibility, and no broker integration should be enabled.

## What changed

| Prior weakness | Hardened implementation |
|---|---|
| Existing ensemble script bypassed its purged OOS path with `skip_oos=True`. | `HardenedCrossFittedEnsemble` refits base models in every chronological fold and records OOS predictions before fitting final artifacts. |
| Sub-model wrappers could silently fail or depend on external saved artifacts and global scalers. | Three reproducible tabular base learners are instantiated from fixed configuration: Ridge, ElasticNet, and Histogram Gradient Boosting. Fold-local scalers are embedded in the linear-model pipelines. |
| Meta-model could be trained on the same predictions being judged. | The Ridge meta-model for each validation fold receives only earlier completed OOS folds. The first fold uses equal-weight averaging and abstains because no residual history exists. |
| Uncertainty was heuristic MC-dropout output without calibrated prediction intervals. | A split-conformal calibrator is fitted only on earlier OOS residuals and produces auditable intervals with an explicit target coverage. |
| Feature drift could silently corrupt inference. | `FeatureSchema` persists ordered column names and a SHA-256 schema hash; missing, unexpected, reordered, non-numeric, or non-finite features are rejected. |
| A predicted sign could flow into simulated exposure despite model uncertainty. | The hardened model returns `abstain`, `interval_lower`, `interval_upper`, base-model dispersion, and `actionable_prediction`. The backtest forces all abstentions flat. |
| Ensemble outputs could not be fully investigated. | Every OOS record now saves the Ridge, ElasticNet, and Histogram Gradient Boosting base forecasts, ensemble prediction, interval, abstention state, and base-prediction dispersion. |

## Delivered components

| Component | Role |
|---|---|
| `research/hardened_ensemble.py` | Cross-fitted base ensemble, chronological meta-learning, conformal intervals, strict feature schema, artifacts, and deterministic configuration. |
| `scripts/run_hardened_ensemble_experiment.py` | Research-only training entry point using the existing causal matrix, labels, and walk-forward split. |
| `scripts/run_shadow_inference.py` | Now understands hardened schema metadata and calibrated abstention; it has no broker, routing, or order import. |
| `research/backtest.py` | Respects the optional `abstain` column, forcing zero position, turnover, and cost for uncertain predictions. |
| `tests/unit/test_hardened_ensemble.py` | Covers schema enforcement, conformal intervals, cross-fitting, artifacts, loaded inference, abstention, and fixed-seed reproducibility. |

## Out-of-sample evidence

The hardened ensemble was trained on the repository’s causal EUR/USD core matrix: 6,168 prepared rows, 29 approved features, a one-bar forward-return horizon, a one-bar entry lag, three expanding purged folds, and a 0.80 conformal target coverage.

| Metric | Result | Interpretation |
|---|---:|---|
| OOS information coefficient | 0.0218 | Below the 0.03 promotion threshold. |
| OOS directional accuracy | 46.26% | Below the 50% promotion threshold. |
| OOS RMSE | 0.000814 | Regression error only; not a trading-performance measure. |
| Conformal interval coverage where available | 80.91% | Close to the 80% target on later folds. |
| Interval availability | 66.65% | The first fold correctly has no prior OOS residuals for calibration. |
| OOS abstention rate | 100.00% | All prediction intervals crossed zero; the model found no statistically actionable forecast. |
| Eligible OOS observations | 0 | No validated trade-like signal was produced. |

When the hardened artifact was sent through the cost-aware OOS backtest, the result was exactly flat: 0 trades, 0 turnover, 0 estimated cost, 0 simulated return, and 100% abstention. This is the intended safety property—not a backtest failure—because an uncalibrated forecast should not become a simulated position.

## Fresh-data shadow result

The latest public EUR/USD hourly observation produced a point forecast of -0.00054967. Its calibrated interval was [-0.00138056, +0.00028122], which crosses zero. The upgraded shadow path therefore returned `FLAT` and added `model_abstained_due_to_calibrated_uncertainty` to its blockers. The source also lacked executable bid/ask quotes and observed positive volume, so it failed the execution-style data eligibility policy independently of the model.

## Quality evidence

The full repository suite completed with **101 passing tests**. The new test coverage verifies fixed-seed reproducibility, strict schema enforcement, artifact persistence/load, conformal interval construction, chronological cross-fitting, forced-flat backtest abstention, and the no-order fresh-data inference path.

## Model-readiness conclusion

The existing research-promotion gates were rerun on the hardened ensemble. It failed the information-coefficient, directional-accuracy, and cost-aware-Sharpe criteria. The result remains `research_iteration_required`; live trading remains unauthorised.

## Next improvement that matters

The next meaningful model improvement is **not** increasing neural-network depth or searching more thresholds on the same sample. It is obtaining a quality dataset with timestamped executable bid/ask prices and observed volume, freezing a candidate feature set and policy on a development period, and testing it exactly once on a later untouched holdout. A candidate must produce enough non-abstained observations, stable subperiod evidence, positive cost-aware diagnostics, and pass the existing promotion gates before it can enter a paper-only review.

This work is model-research and software infrastructure, not investment advice, a performance promise, or an authorisation to trade.
