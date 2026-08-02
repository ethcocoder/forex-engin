import sys, os, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

for name, path in [
    ("temporal_model.pt",  "saved_models/temporal_model.pt"),
    ("maml_model.pt",      "saved_models/maml_model.pt"),
]:
    print(f"\n=== {name} ===")
    sd = torch.load(path, map_location="cpu")
    print(f"  type: {type(sd)}")
    if isinstance(sd, dict):
        keys = list(sd.keys())
        print(f"  top-level keys ({len(keys)}): {keys[:10]}")
        # try nested
        for k, v in sd.items():
            if isinstance(v, dict):
                sub = list(v.keys())[:5]
                n = sum(p.numel() for p in v.values() if hasattr(p,"numel"))
                print(f"    [{k}] -> dict with {len(v)} keys, {n:,} params  e.g. {sub}")
            elif hasattr(v, "numel"):
                print(f"    [{k}] tensor {tuple(v.shape)}  {v.numel():,} params")
            else:
                print(f"    [{k}] {type(v).__name__} = {str(v)[:80]}")
    else:
        print(f"  raw object: {type(sd)}")
        if hasattr(sd, "__dict__"):
            for k, v in sd.__dict__.items():
                print(f"    .{k}: {type(v).__name__}")
