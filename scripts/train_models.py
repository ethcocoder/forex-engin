import os
import sys
import subprocess
import structlog

logger = structlog.get_logger()

def run_command(command: list, description: str):
    logger.info(f"--- {description} ---")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {description}: {e}")
        sys.exit(1)

def main():
    # 1. Compile Speedups
    run_command([sys.executable, "scripts/compile_speedups.py"], "Compiling C++ Speedups")

    # 2. Download Data (if not exists)
    if not os.path.exists("data/EUR_USD_ticks.csv"):
        run_command([sys.executable, "scripts/download_data.py", "--pair", "EUR_USD", "--years", "1"], "Downloading Historical Data")

    # 3. Generate Features
    run_command([sys.executable, "scripts/generate_features.py"], "Generating Features")

    # 4. Train Regime Model
    run_command([sys.executable, "scripts/train_regime.py", "--epochs", "5"], "Training Regime Model")

    # 5. Train Temporal Model
    run_command([sys.executable, "scripts/train_temporal.py", "--epochs", "10"], "Training Temporal Model")

    # 6. Train Meta Learner
    run_command([sys.executable, "scripts/train_meta.py", "--epochs", "10"], "Training Meta Learner")

    # 7. Train RL Agent
    run_command([sys.executable, "scripts/train_rl.py", "--timesteps", "10000"], "Training RL Agent")

    # 8. Train Ensemble Aggregator
    run_command([sys.executable, "scripts/train_ensemble.py"], "Training Ensemble Aggregator")

    logger.info("--- FULL MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()
