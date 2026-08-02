# 🚀 Google Colab "God Mode" Forex Engine Training Guide

This guide provides the ultimate step-by-step instructions for setting up, training, and running the **"God Mode" Forex Neural Trading Engine** on Google Colab using a T4 GPU. This implementation features nanosecond execution simulations, deep neural synapses, and adversarial AI hardening.

## 🛠️ 1. Environment Setup

### Enable T4 GPU
1. Open [Google Colab](https://colab.research.google.com/).
2. **Runtime** > **Change runtime type** > **T4 GPU** > **Save**.

### Verify GPU Status
```python
!nvidia-smi
```

---

## 📂 2. Repository & Dependencies

### Clone the "God Mode" Branch
```python
import os

# Clone the elite-forex branch
!git clone -b elite-forex https://github.com/ethcocoder/forex-engin.git
%cd forex-engin
```

### Install Production Dependencies
```python
# Install core requirements
!pip install -r requirements.txt

# Install data & utility dependencies
!pip install yfinance structlog
```

### Compile High-Performance C++ Speedups
```python
# Compile Kalman wavelets, RL agent, and MAML meta-learner speedups
!python scripts/compile_speedups.py
```

---

## 📊 3. Data Acquisition & Feature Engineering

### Autonomous Data Sync
Download 20 years of historical data for regime detection and 2 years of hourly data for signal tuning.
```python
# Download historical data (No API key required)
!python scripts/download_data.py
```

### Feature Generation (Neural Synapse Layer)
Convert raw ticks into high-dimensional features including wavelets, volatility estimators, and microstructure spreads.
```python
# Prepare data for processing
!cp data/raw/EURUSD_H1_2y.csv data/EUR_USD_ticks.csv
!sed -i 's/Datetime/timestamp/g; s/Close/close/g; s/Open/open/g; s/High/high/g; s/Low/low/g; s/Volume/volume/g' data/EUR_USD_ticks.csv

# Generate the feature set
!python scripts/generate_features.py --input data/EUR_USD_ticks.csv --output data/EUR_USD_features.csv
```

---

## 🧠 4. Sequential Model Training

The engine requires sequential training as the master ensemble aggregates outputs from all specialized layers.

| Step | Command | Description |
| :--- | :--- | :--- |
| **A. Temporal** | `!python scripts/train_temporal.py --features data/EUR_USD_features.csv --raw data/EUR_USD_ticks.csv --epochs 20` | Trains TCN + Transformer for return prediction. |
| **B. Regime** | `!python scripts/train_regime.py --features data/EUR_USD_features.csv --epochs 15` | Unsupervised HMM + LSTM for market state classification. |
| **C. RL Agent** | `!python scripts/train_rl.py --features data/EUR_USD_features.csv --raw data/EUR_USD_ticks.csv --timesteps 50000` | PPO agent with volatility curriculum learning. |
| **D. Meta-Learner** | `!python scripts/train_meta.py --features data/EUR_USD_features.csv --raw data/EUR_USD_ticks.csv --epochs 50` | MAML for rapid adaptation to new market conditions. |
| **E. Ensemble** | `!python scripts/train_ensemble.py --features data/EUR_USD_features.csv --raw data/EUR_USD_ticks.csv` | LightGBM stacking layer to unify all predictions. |

---

## ⚡ 5. Execution & "God Mode" Simulation

### Run Integrated Real-Time Paper Trading
This script executes the full "God Mode" pipeline, including Global Mesh Arbitrage, Kernel-Bypass simulations, and Deep Neural Synapse integration.
```python
# Execute the integrated God Mode trading loop
!python scripts/run_real_paper_trading.py --features data/EUR_USD_features.csv --raw data/EUR_USD_ticks.csv
```

### Run God Mode Stress Test
Verify the engine's performance under extreme volatility (e.g., Fed Rate Hike scenarios).
```python
!python scripts/god_mode_stress_test.py
```

---

## 💾 6. Exporting Weights & Persistence

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Backup all trained models and scalers
!cp -r saved_models/ /content/drive/MyDrive/forex_god_mode_weights/
```

### Direct Download
```python
from google.colab import files
import glob

for weight_file in glob.glob("saved_models/*.*"):
    files.download(weight_file)
```

---

## 💡 Pro Tips for T4 GPU
* **Mixed Precision**: The engine uses `torch.cuda.amp` for 2x faster training on T4.
* **Thread Optimization**: Use `--threads=8` to maximize CPU throughput during feature generation.
* **Keep-Alive**: Run this in the browser console (F12) to prevent timeouts:
  ```javascript
  function ClickConnect(){
    console.log("Working"); 
    document.querySelector("colab-connect-button").click() 
  }
  setInterval(ClickConnect, 60000)
  ```
