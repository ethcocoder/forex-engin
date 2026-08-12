# Forex Engin: Main Research Campaign & Model Training Plan

**Author:** **Manus AI**  
**Branch:** `elite10x-pr`  
**Date:** August 12, 2026  
**Status:** **Active Research Campaign (Live Execution Locked)**  

---

## Executive Summary

Having successfully acquired 20 years of daily regime data and 2 years of hourly signal data for EUR/USD (`data/raw/EURUSD_D1_20y.csv`, `data/raw/EURUSD_H1_2y.csv`), Forex Engin is launching its **Main Research Campaign**. This campaign trains and validates the full hierarchical model stack—ranging from HMM regime classifiers to temporal transformers, PPO/SAC reinforcement learning agents, MAML meta-learners, and the GOAT Ensemble Aggregator—while maintaining strict leak-prevention and safety locks.

---

## 1. Research Campaign Phases

| Phase | Module / Component | Dataset | Objective |
| :---: | :--- | :--- | :--- |
| **1** | Regime Classifier (`models/regime/hmm.py`) | EUR/USD D1 (20 Years) | Identify bull, bear, and high/low volatility market regimes. |
| **2** | Temporal Transformer (`models/temporal/transformer.py`) | EUR/USD H1 (2 Years) | Extract multi-horizon non-linear price patterns. |
| **3** | RL Agents (`models/rl_agent/ppo_agent.py`, `sac_agent.py`) | EUR/USD H1 (2 Years) | Optimize dynamic trade sizing and execution timing. |
| **4** | MAML Meta-Learner (`models/meta_learner/maml.py`) | EUR/USD H1 (2 Years) | Enable rapid few-shot adaptation to structural regime shifts. |
| **5** | GOAT Ensemble Aggregator (`models/ensemble/aggregator.py`) | Purged OOF Meta-Features | Combine sub-model predictions with uncertainty gating. |

---

## 2. Safety & Governance Controls

- **Leakage Prevention**: All model evaluations enforce purged walk-forward cross-validation and embargoing.
- **Execution Locks**: Real-money and live brokerage execution paths remain programmatically locked (`DENIED`).
- **Reproducibility**: All training metrics and artifacts are logged with SHA-256 provenance tracking.

***
*Plan authored autonomously by **Manus AI**.*
