import pickle, sys, os
import numpy as np

def deep_sizeof(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(v, seen) for v in obj.values())
        size += sum(deep_sizeof(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += deep_sizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum(deep_sizeof(i, seen) for i in obj)
        except Exception:
            pass
    return size

pkl_path = "saved_models/ensemble_aggregator.pkl"
with open(pkl_path, "rb") as f:
    agg = pickle.load(f)

print("=== Ensemble Aggregator Memory Breakdown ===")
state = agg.__dict__ if hasattr(agg, "__dict__") else {}
for k, v in sorted(state.items(), key=lambda x: deep_sizeof(x[1]), reverse=True):
    mb = deep_sizeof(v) / 1024 / 1024
    print(f"  {k:<30} {mb:.3f} MB  ({type(v).__name__})")

total_mb = deep_sizeof(agg) / 1024 / 1024
file_mb = os.path.getsize(pkl_path) / 1024 / 1024
print(f"\n  TOTAL deep size : {total_mb:.3f} MB")
print(f"  PKL file on disk: {file_mb:.3f} MB")

# Also check the stacker specifically
stacker = state.get("stacker") or state.get("lgbm_stacker")
if stacker is not None:
    print(f"\n  Stacker type: {type(stacker).__name__}")
    stacker_mb = deep_sizeof(stacker) / 1024 / 1024
    print(f"  Stacker size : {stacker_mb:.3f} MB")
    if hasattr(stacker, "booster_"):
        trees = stacker.booster_.num_trees()
        print(f"  LightGBM trees: {trees}")
    if hasattr(stacker, "n_estimators"):
        print(f"  n_estimators: {stacker.n_estimators}")

# Check scaler
scaler = state.get("scaler")
if scaler is not None:
    scaler_mb = deep_sizeof(scaler) / 1024 / 1024
    print(f"\n  Scaler type  : {type(scaler).__name__}")
    print(f"  Scaler size  : {scaler_mb:.3f} MB")

# meta feature names count
mfn = state.get("meta_feature_names")
if mfn is not None:
    print(f"\n  Meta feature names count: {len(mfn)}")
    print(f"  Meta feature names: {mfn}")
