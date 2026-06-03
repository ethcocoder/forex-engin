"""
Full pipeline latency + memory profiler (robust version).
Run from repo root:  python scratch/profile_latency.py
"""
import sys, os, time, pickle, gc, warnings, traceback
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

SEQ_LEN = 60
N_FEATS = 57
REPS    = 100
DEVICE  = "cpu"

# ── helpers ──────────────────────────────────────────────────────────────────
def deep_sizeof(obj, seen=None):
    if seen is None: seen = set()
    if id(obj) in seen: return 0
    seen.add(id(obj))
    sz = sys.getsizeof(obj)
    if isinstance(obj, dict):
        sz += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif hasattr(obj, "__dict__"):
        sz += deep_sizeof(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try: sz += sum(deep_sizeof(i, seen) for i in obj)
        except: pass
    return sz

def mbs(obj): return deep_sizeof(obj) / 1024 / 1024
def fsz(path): return os.path.getsize(path) / 1024 / 1024

def bench(label, fn, reps=REPS):
    try:
        for _ in range(3): fn()
        t0 = time.perf_counter_ns()
        for _ in range(reps): fn()
        ns = (time.perf_counter_ns() - t0) / reps
        flag = "  ⚠ SLOW (>1ms)" if ns > 1_000_000 else ""
        print(f"  {label:<52} {ns:>13,.0f} ns  {ns/1e3:>9,.1f} µs{flag}")
        return ns
    except Exception as e:
        print(f"  {label:<52}  SKIP — {e}")
        return None

print("\n" + "="*80)
print("  STAGE-BY-STAGE LATENCY  (avg over", REPS, "reps, CPU only)")
print("="*80)
print(f"  {'Stage':<52} {'ns':>13}   {'µs':>9}")
print("  " + "-"*76)

import torch

# ── 0. Raw numpy ops ──────────────────────────────────────────────────────────
X_full  = np.random.randn(12349, N_FEATS).astype(np.float32)
X_seq   = X_full[:SEQ_LEN]              # (60, 57)
X_flat  = X_seq.flatten()[None]         # (1, 3420)
X_3d    = X_seq[None]                   # (1, 60, 57)

bench("numpy: window slice + flatten", lambda: X_full[:SEQ_LEN].flatten()[None])

# ── 1. TransformerEncoderModel (with strict=False to handle key mismatch) ────
temporal = None
try:
    from models.temporal.transformer import TransformerEncoderModel
    temporal = TransformerEncoderModel(config={"device": DEVICE})
    # use strict=False to skip mismatched keys silently
    sd = torch.load("saved_models/temporal_model.pt", map_location=DEVICE)
    temporal.model.load_state_dict(sd, strict=False)
    temporal.model.eval()
    temporal_mb = mbs(temporal)
    bench("TransformerEncoderModel.predict (seq)", lambda: temporal.predict(X_3d))
except Exception as e:
    print(f"  TransformerEncoderModel                               SKIP — {e}")
    temporal_mb = 0.0

# ── 2. MAMLModel ──────────────────────────────────────────────────────────────
maml = None
try:
    from models.meta_learner.maml import MAMLModel
    maml = MAMLModel(config={"device": DEVICE})
    maml.load("saved_models/maml_model.pt")
    maml_mb = mbs(maml)
    bench("MAMLModel.predict (flat)", lambda: maml.predict(X_flat))
except Exception as e:
    print(f"  MAMLModel                                             SKIP — {e}")
    maml_mb = 0.0

# ── 3. PPO (SB3) ──────────────────────────────────────────────────────────────
ppo = None
try:
    from models.rl_agent.ppo_agent import PPOAgent
    ppo = PPOAgent(obs_dim=N_FEATS, act_dim=3)
    try: ppo.load("saved_models/rl_agent_ppo.zip")
    except: pass
    ppo_mb = mbs(ppo)
    bench("PPOAgent.predict (flat)", lambda: ppo.predict(X_full[:1]))
except Exception as e:
    print(f"  PPOAgent                                              SKIP — {e}")
    ppo_mb = 0.0

# ── 4. RegimeEnsemble ─────────────────────────────────────────────────────────
regime = None
try:
    from models.regime.regime_ensemble import RegimeEnsembleEstimator
    regime = RegimeEnsembleEstimator()
    regime.load("saved_models/regime_ensemble.pkl")
    regime_mb = mbs(regime)
    bench("RegimeEnsemble.predict (seq)", lambda: regime.predict(X_3d, return_proba=True))
except Exception as e:
    print(f"  RegimeEnsemble                                        SKIP — {e}")
    regime_mb = 0.0

# ── 5. EnsembleAggregator full stack ─────────────────────────────────────────
agg = None
try:
    with open("saved_models/ensemble_aggregator.pkl", "rb") as f:
        agg = pickle.load(f)
    if temporal: agg.register_model("temporal", temporal, cluster="core", is_torch=True)
    if maml:     agg.register_model("maml",     maml,     cluster="core", is_torch=True)
    if ppo:      agg.register_model("rl",       ppo,      cluster="core")
    agg_mb = mbs(agg)
    bench("EnsembleAggregator.predict (full stack)", lambda: agg.predict(X_flat, return_signal=True))
except Exception as e:
    print(f"  EnsembleAggregator full stack                         SKIP — {e}")
    agg_mb = 0.0

# ── 6. Risk gate ──────────────────────────────────────────────────────────────
try:
    from risk.risk_engine import AntiFragileRiskEngine, PortfolioState
    from risk.sizing.kelly import KellyPositionSizer
    from models.ensemble.signal_generator import AlphaSignal
    risk = AntiFragileRiskEngine()
    risk.set_sizer(KellyPositionSizer())
    portfolio = PortfolioState(
        current_equity=10000.0, open_positions={},
        daily_pnl=0.0, weekly_pnl=0.0, monthly_pnl=0.0,
        win_rate=0.5, win_loss_ratio=1.0, historical_returns=np.zeros(200))
    sig = AlphaSignal(direction=1, magnitude=1.0, confidence=0.65,
                      uncertainty=0.35, expected_decay_steps=6, regime=0,
                      timestamp=time.time(), metadata={})
    mkt = {"close":1.085,"mid_price":1.085,"spread_pips":1.0,
           "adv":1_000_000.0,"pip_value":0.0001,"volatility":0.0005,"atr":0.0008}
    bench("AntiFragileRiskEngine.gate", lambda: risk.gate(sig, "EURUSD", portfolio, mkt))
except Exception as e:
    print(f"  AntiFragileRiskEngine.gate                            SKIP — {e}")

# ── 7. PaperBroker.place_order ────────────────────────────────────────────────
try:
    from execution.brokers.paper_broker import PaperBroker
    from risk.risk_engine import OrderRequest
    broker = PaperBroker(config={"initial_capital": 10000.0})
    broker.update_market_state({"EURUSD": {
        "mid_price": 1.085, "spread_pips": 1.0,
        "adv": 1_000_000.0, "pip_value": 0.0001
    }})
    order = OrderRequest(pair="EURUSD", direction=1, size=100.0)
    bench("PaperBroker.place_order", lambda: broker.place_order(order))
except Exception as e:
    print(f"  PaperBroker.place_order                               SKIP — {e}")

print()
print("="*80)
print("  IN-MEMORY FOOTPRINT  (live loaded, includes weights)")
print("="*80)
print(f"  {'Component':<42} {'Live MB':>8}   {'Disk MB':>8}")
print("  " + "-"*62)
rows = [
    ("TransformerEncoderModel",  temporal_mb, "saved_models/temporal_model.pt"),
    ("MAMLModel",                maml_mb,     "saved_models/maml_model.pt"),
    ("PPOAgent",                 ppo_mb,      "saved_models/rl_agent_ppo.zip"),
    ("RegimeEnsemble",           regime_mb,   "saved_models/regime_ensemble.pkl"),
    ("EnsembleAggregator shell", agg_mb,      "saved_models/ensemble_aggregator.pkl"),
]
total_live = 0.0
for name, live, path in rows:
    disk = fsz(path) if os.path.exists(path) else 0
    total_live += live
    print(f"  {name:<42} {live:>8.2f} MB   {disk:>8.2f} MB")
print(f"  {'─'*62}")
print(f"  {'TOTAL':<42} {total_live:>8.2f} MB")
print()
print("  KEY INSIGHT:")
print("  ┌──────────────────────────────────────────────────────────────┐")
print("  │ PyTorch CPU inference floor: ~500 µs – 5 ms per forward     │")
print("  │ Python overhead per tick:    ~50 – 200 µs                   │")
print("  │ True nanosecond execution requires ONNX/TRT + C extension   │")
print("  └──────────────────────────────────────────────────────────────┘")
