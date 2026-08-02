#!/usr/bin/env python3
"""Train pipeline orchestration for the Forex Neural Engine.

This script compiles C++ speedups, downloads the base data, generates features,
and runs the full model training pipeline in sequence.

Usage:
    python train-model.py
    python train-model.py --skip-download --skip-rl

The pipeline steps are intentionally conservative: if any step fails,
it stops and reports the failure.
"""

import argparse
import os
import shutil
import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
RAW_DEFAULT = DATA_DIR / "EUR_USD_ticks.csv"
FEATURES_DEFAULT = DATA_DIR / "EUR_USD_features.csv"
LOG_FORMAT = "[%(levelname)s] %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("train-model")


def run_script(script_name: str, args: Optional[List[str]] = None) -> int:
    args = args or []
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error("Required script not found: %s", str(script_path))
        return 1

    cmd = [sys.executable, str(script_path)] + args
    logger.info("Running script: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        logger.error("Script failed: %s (returncode=%s)", script_name, result.returncode)
    return result.returncode


def compile_execution_speedups() -> bool:
    """Compile the execution C++ shared library for ultra-low latency routing."""
    source = ROOT / "execution" / "execution_speedups.cpp"
    if not source.exists():
        logger.warning("execution_speedups.cpp not found, skipping execution speedups compile")
        return True

    if sys.platform.startswith("win"):
        lib_name = "execution_speedups.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "execution_speedups.dylib"
    else:
        lib_name = "execution_speedups.so"

    destination = source.parent / lib_name
    compiler = "g++"
    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(destination),
    ]

    logger.info("Compiling execution speedups from %s to %s", str(source), str(destination))

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("Compiled execution speedups successfully: %s", str(destination))
        return True
    except FileNotFoundError:
        logger.error("C++ compiler not found on PATH. Install g++ or use WSL / MinGW.")
        return False
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to compile execution speedups: %s", exc.stderr.strip())
        return False


def normalize_yf_output(input_path: Path, output_path: Path) -> bool:
    """Convert downloaded yfinance CSV into the raw tick CSV format expected by the pipeline."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required to normalize downloaded data.")
        return False

    if not input_path.exists():
        logger.error("Downloaded raw file not found: %s", str(input_path))
        return False

    logger.info("Normalizing downloaded raw data from %s to %s", str(input_path), str(output_path))
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    if df.index.name is None:
        df.index.name = "timestamp"
    else:
        df.index.name = "timestamp"

    df.columns = [col.lower() for col in df.columns]
    if "close" not in df.columns:
        logger.error("Downloaded data does not contain a close price column.")
        return False

    if "bid" not in df.columns or "ask" not in df.columns:
        logger.info("Synthesizing bid/ask from close prices")
        spread = df["close"] * 0.0001
        if "bid" not in df.columns:
            df["bid"] = df["close"] - spread / 2.0
        if "ask" not in df.columns:
            df["ask"] = df["close"] + spread / 2.0

    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path)
    logger.info("Normalized raw data saved to %s (%s rows)", str(output_path), len(df))
    return True


def find_downloaded_yf_file(pair: str) -> Optional[Path]:
    pattern = f"{pair.replace('_', '')}*"
    candidates = list((DATA_DIR / "raw").glob(pattern + ".csv"))
    if not candidates:
        return None
    # prefer hourly data over daily if both exist
    for file in candidates:
        if "_H1_" in file.name or file.name.endswith("_H1_2y.csv"):
            return file
    return candidates[0]


def download_data(pair: str) -> bool:
    """Fetch raw historical data for the target forex pair."""
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.download_data import GOATDataPipeline
    except Exception as exc:
        logger.error("Unable to import download_data pipeline: %s", str(exc))
        return False

    pipeline = GOATDataPipeline(data_dir=str(DATA_DIR / "raw"))
    try:
        pipeline.sync_20_year_history(pair)
    except Exception as exc:
        logger.error("download_data.py failed: %s", str(exc))
        return False

    downloaded = find_downloaded_yf_file(pair)
    if downloaded is None:
        logger.warning("No downloaded data file found after download step")
        return True

    if not RAW_DEFAULT.exists():
        if not normalize_yf_output(downloaded, RAW_DEFAULT):
            return False
    return True


def ensure_raw_input(raw_path: Path, pair: str) -> bool:
    if raw_path.exists():
        logger.info("Using existing raw input file: %s", str(raw_path))
        return True

    downloaded = find_downloaded_yf_file(pair)
    if downloaded is None:
        logger.warning("No downloaded raw data file available to normalize.")
        return False

    return normalize_yf_output(downloaded, raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train pipeline orchestrator for the Forex Neural Engine")
    parser.add_argument("--pair", type=str, default="EURUSD", help="Forex pair symbol for download_data pipeline")
    parser.add_argument("--raw", type=str, default=str(RAW_DEFAULT), help="Path to raw CSV input for feature generation")
    parser.add_argument("--features", type=str, default=str(FEATURES_DEFAULT), help="Path to generated features CSV")
    parser.add_argument("--skip-compile", action="store_true", help="Skip C++ compilation step")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature generation step")
    parser.add_argument("--skip-regime", action="store_true", help="Skip regime model training")
    parser.add_argument("--skip-temporal", action="store_true", help="Skip temporal model training")
    parser.add_argument("--skip-meta", action="store_true", help="Skip meta learner training")
    parser.add_argument("--skip-adversarial", action="store_true", help="Skip adversarial attacker training")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL agent training")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble aggregator training")
    parser.add_argument("--download-only", action="store_true", help="Only download data and normalize raw input")
    parser.add_argument("--no-color", action="store_true", help="Disable colored logging output")
    args = parser.parse_args()

    if args.no_color:
        try:
            import colorama
            colorama.deinit()
        except Exception:
            pass

    if not args.skip_compile:
        logger.info("STEP 1: Compile C++ speedups")
        rc = run_script("compile_speedups.py")
        if rc != 0:
            return rc
        if not compile_execution_speedups():
            return 1

    if not args.skip_download:
        logger.info("STEP 2: Download raw data")
        if not download_data(args.pair):
            return 1

    if args.download_only:
        logger.info("Download-only mode complete.")
        return 0

    if not args.skip_features:
        logger.info("STEP 3: Generate features")
        if not ensure_raw_input(Path(args.raw), args.pair):
            return 1
        rc = run_script("generate_features.py", ["--input", args.raw, "--output", args.features])
        if rc != 0:
            return rc

    if not args.skip_regime:
        logger.info("STEP 4: Train regime ensemble")
        rc = run_script("train_regime.py", ["--features", args.features])
        if rc != 0:
            return rc

    if not args.skip_temporal:
        logger.info("STEP 5: Train temporal model")
        rc = run_script("train_temporal.py", ["--features", args.features, "--raw", args.raw])
        if rc != 0:
            return rc

    if not args.skip_meta:
        logger.info("STEP 6: Train meta learner")
        rc = run_script("train_meta.py", ["--features", args.features, "--raw", args.raw])
        if rc != 0:
            return rc

    if not args.skip_adversarial:
        logger.info("STEP 7: Train adversarial attacker model")
        rc = run_script("train_adversarial.py", ["--features", args.features, "--raw", args.raw])
        if rc != 0:
            return rc

    if not args.skip_rl:
        logger.info("STEP 8: Train RL agent")
        rc = run_script("train_rl.py", ["--features", args.features, "--raw", args.raw])
        if rc != 0:
            return rc

    if not args.skip_ensemble:
        logger.info("STEP 9: Train ensemble aggregator")
        rc = run_script("train_ensemble.py", ["--features", args.features, "--raw", args.raw])
        if rc != 0:
            return rc

    logger.info("Training pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
