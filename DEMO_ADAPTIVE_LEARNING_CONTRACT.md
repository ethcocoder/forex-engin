# Forex Engin: Demo-Only Adaptive Learning & Closed-Loop Safety Contract

**Author:** **Manus AI**  
**Branch:** `elite10x-pr`  
**Date:** August 12, 2026  
**Status:** **Enforced Safety Architecture (Demo-Only Closed-Loop Feedback)**  

---

## Executive Summary

To fulfill the user's requirement for a system that learns continuously from trade execution and feedback while maintaining institutional financial safety, Forex Engin establishes a **Demo-Only Adaptive Learning Pipeline**. 

Uncontrolled online learning in live financial markets is inherently dangerous: feedback loops with noisy rewards can cause policy collapse, catastrophic credit exposure, or gradient divergence. This contract defines how Forex Engin captures broker-simulated demo trade outcomes, processes rewards through offline/online hybrid trainers, and evaluates challenger weights via shadow champions before any parameter update is promoted.

---

## 1. Core Principles of Adaptive Learning

1. **Demo-Only Isolation**: Closed-loop learning and reward feedback operate exclusively against paper trading accounts or broker demo environments. Real-money API keys are programmatically locked.
2. **Immutable Trade Journaling**: Every signal emitted by the GOAT Ensemble Aggregator is logged with unique `signal_id`, feature vector hashes, sub-model predictions, uncertainty scores, and market regime tags.
3. **Reward Attribution & Slippage Correction**: Realized PnL from demo executions is adjusted for simulated bid-ask spread, latency, and swap rates before being fed back into the Bayesian Model Averager (`BayesianModelAverager`) and Online RL policy adapters (`online_adapter.py`).
4. **Champion-Challenger Promotion Gates**: Newly adapted weights or policy checkpoints must outperform the reigning champion over an independent validation buffer (minimum 100 demo trades) before promotion.

---

## 2. Closed-Loop Feedback Architecture

```
[Broker Demo Feed] ---> [Execution & Fill Log]
                                │
                                v
[GOAT Ensemble Signal] ---> [Trade Journal & PnL Attribution]
                                │
                                v
                [Bayesian Model Averager / Online Adapter]
                                │
                                v
                [Shadow Challenger Evaluation Gate]
                                │
                       (Passes Out-of-Sample?)
                         /               \
                     (Yes)               (No)
                       v                   v
            [Promote to Champion]    [Discard Adaptation]
```

---

## 3. Implementation Blueprint

- **Execution Script**: `scripts/run_demo_adaptive_learning.py` manages the closed-loop feedback runner.
- **Journal Storage**: SQLite/JSONL audit log at `/home/ubuntu/forex-engin/data/demo_trade_journal.jsonl`.
- **Promotion Gate**: Requires positive Sharpe ratio, max drawdown $< 15\%$, and win-loss expectancy improvement over the baseline champion.

***
*Contract authored autonomously by **Manus AI**.*
