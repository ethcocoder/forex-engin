import sys, os, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

pt_models = [
    ("temporal_model.pt",        "saved_models/temporal_model.pt"),
    ("maml_model.pt",            "saved_models/maml_model.pt"),
    ("adversarial_attacker.pt",  "saved_models/adversarial_attacker.pt"),
]
all_files = [
    ("temporal_model.pt",        "saved_models/temporal_model.pt"),
    ("maml_model.pt",            "saved_models/maml_model.pt"),
    ("rl_agent_ppo.zip",         "saved_models/rl_agent_ppo.zip"),
    ("ensemble_aggregator.pkl",  "saved_models/ensemble_aggregator.pkl"),
    ("regime_ensemble.pkl",      "saved_models/regime_ensemble.pkl"),
    ("regime_ensemble.pkl.lstm", "saved_models/regime_ensemble.pkl.lstm"),
    ("adversarial_attacker.pt",  "saved_models/adversarial_attacker.pt"),
    ("adversarial_attacker.pkl", "saved_models/adversarial_attacker.pkl"),
]

print("Disk sizes:")
for name, path in all_files:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {name:<36} {size_mb:.2f} MB")

print()
print("Param counts (from .pt state dicts):")
for name, path in pt_models:
    if not os.path.exists(path):
        print(f"  {name:<36} NOT FOUND")
        continue
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        n_bytes  = sum(v.numel() * v.element_size() for v in sd.values() if hasattr(v, "numel"))
        print(f"  {name:<36} {n_params:>12,} params   {n_bytes/1024/1024:.2f} MB fp32")
    except Exception as e:
        print(f"  {name:<36} ERROR: {e}")
