import pickle
import numpy as np

with open("saved_models/adversarial_attacker.pkl", "rb") as f:
    meta = pickle.load(f)

print("Name:", meta.get("name"))
print("Input dim:", meta.get("input_dim"))
mean = meta.get("scaler_mean")
std = meta.get("scaler_std")

print("Mean contains nan:", np.isnan(mean).any() if mean is not None else "None")
print("Mean contains inf:", np.isinf(mean).any() if mean is not None else "None")
print("Std contains nan:", np.isnan(std).any() if std is not None else "None")
print("Std contains inf:", np.isinf(std).any() if std is not None else "None")

if mean is not None:
    print("Mean min/max/mean:", np.min(mean), np.max(mean), np.mean(mean))
if std is not None:
    print("Std min/max/mean:", np.min(std), np.max(std), np.mean(std))

features = meta.get("feature_names", [])
print("Feature count:", len(features))
# Print features where mean or std is nan/inf
for idx, name in enumerate(features):
    m_val = mean[idx] if mean is not None else None
    s_val = std[idx] if std is not None else None
    if (m_val is not None and (np.isnan(m_val) or np.isinf(m_val))) or (s_val is not None and (np.isnan(s_val) or np.isinf(s_val))):
        print(f"Feature '{name}' (idx {idx}) has mean={m_val}, std={s_val}")
