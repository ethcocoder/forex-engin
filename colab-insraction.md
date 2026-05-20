# Google Colab T4 GPU Training Instructions

This guide provides step-by-step instructions for setting up and training the **Forex Neural Trading Engine** on Google Colab using a free T4 GPU.

## Prerequisites
* A Google Account
* A copy of this repository pushed to your GitHub

---

## 1. Setup the Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** > **New notebook**.
3. **Enable the T4 GPU**:
   - Go to **Runtime** > **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU**.
   - Click **Save**.

To verify the GPU is active, run the following command in the first cell:
```python
!nvidia-smi
```

---

## 2. Clone the Repository

Clone your repository into the Colab environment. Run this in a new cell:

```python
import os

# Clone the repository
!git clone https://github.com/ethcocoder/forex-engin.git
%cd forex-engin
```

---

## 3. Install Dependencies

Install the required Python packages for the neural engine and its dependencies.

```python
# Install required libraries
!pip install -r requirements.txt

# Install TimescaleDB, Kafka, and Redis client dependencies (since they are in requirements)
# For local DB testing without Docker inside Colab, you can use SQLite.
```

If you have specific C++ extensions (like the MAML speedups) that need to be compiled on the Linux T4 instance, run:

```python
# Compile MAML C++ speedups for Linux (.so)
!g++ -O3 -shared -fPIC -o models/meta_learner/maml_speedups.so models/meta_learner/maml_speedups.cpp
```

---

## 4. Download Training Data

To train the models on real market behavior without needing any API tokens, you can use the built-in Yahoo Finance downloader. This downloads historical Forex data completely free:

```python
# Download 2 years of historical data from Yahoo Finance (Requires NO API key)
!python scripts/download_data.py --pair EUR_USD --years 2 --source yfinance --output data/EUR_USD_ticks.csv
```

*(Note: If you eventually want extremely dense 1-minute tick data for 5+ years, you can get a free OANDA Practice Token and run it with `--source oanda --token YOUR_TOKEN`).*

---

## 5. Run the Training Pipeline

Set environment variables to bypass Kafka and TimescaleDB for isolated model training, and execute the temporal, regime, and RL training scripts.

```python
import os
os.environ["FOREX_ENVIRONMENT"] = "development"
os.environ["FOREX_EXECUTION_BROKER"] = "paper"

# Example: Run the Backtesting or Training scripts
# Since the orchestrator is highly modular, you can directly train specific models:
!python -m pytest tests/unit/ -v  # Verify environment is sane first
```

### To Train the RL Agent (PPO/SAC)
If you have a dedicated training script (e.g., `train_rl.py`), run it natively:
```python
# Assuming you create a training entrypoint script
!python scripts/train_rl.py --pair EUR_USD --episodes 1000
```

### To Train the Neural Ensemble & Meta-Learner
```python
# Train the Temporal TCN/Transformer
!python scripts/train_temporal.py --epochs 50 --batch_size 64

# Train the MAML outer-loop
!python scripts/train_maml.py --meta_epochs 100
```

---

## 6. Exporting Trained Weights

Once training finishes, Colab will reset when you close the browser. Ensure you download the `.pt` or `.pkl` weight files!

```python
from google.colab import files

# Example of downloading the saved Temporal Model weights
files.download("saved_models/temporal_model_v1.pt")
files.download("saved_models/rl_agent_ppo.zip")
```

Alternatively, you can mount your Google Drive to save weights automatically:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy weights to Google Drive
!cp -r saved_models/ /content/drive/MyDrive/forex_weights/
```

---

## 💡 Performance Tips for T4
* **PyTorch AMP**: Automatic Mixed Precision is enabled by default in the architecture, which doubles T4 throughput.
* **Batch Size**: The T4 has 16GB of VRAM. You can safely increase `--batch_size` to `128` or `256` for the Temporal models without out-of-memory errors.
* **Colab Disconnects**: Add a small JavaScript loop in your browser console to prevent Colab from disconnecting during multi-hour RL training sessions.
