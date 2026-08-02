#!/usr/bin/env python3
"""Train the Adversarial Attacker Model for the Forex Neural Engine."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the root directory is in the Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.adversarial_ai.attacker_model import AttackerModel
from configs.loader import load_config

logger = structlog.get_logger()


def create_vulnerability_dataset(features_df: pd.DataFrame, raw_df: pd.DataFrame, future_horizon: int, loss_threshold: float):
    """Create a dataset labeling periods with large short-term adverse moves as vulnerabilities."""
    features_df = features_df.copy().fillna(0.0)
    raw_df = raw_df.copy().fillna(method="ffill").fillna(0.0)

    if "close" not in raw_df.columns:
        raise ValueError("Raw data must contain a 'close' column.")

    if not features_df.index.equals(raw_df.index):
        common_index = features_df.index.intersection(raw_df.index)
        features_df = features_df.loc[common_index]
        raw_df = raw_df.loc[common_index]

    close = raw_df["close"].values.astype(np.float32)
    n_samples = len(close) - future_horizon
    if n_samples <= 0:
        raise ValueError("Not enough raw price data to build a vulnerability dataset.")

    X = features_df.iloc[:n_samples].values.astype(np.float32)
    future_close = close[future_horizon:]
    current_close = close[:n_samples]

    future_returns = future_close / current_close - 1.0
    y = (future_returns < -loss_threshold).astype(np.float32)

    logger.info(
        "Created vulnerability dataset",
        samples=n_samples,
        loss_threshold=loss_threshold,
        vulnerable_pct=float(np.mean(y) * 100.0)
    )

    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Adversarial Attacker Model")
    parser.add_argument("--features", type=str, default="data/EUR_USD_features.csv", help="Path to generated features CSV")
    parser.add_argument("--raw", type=str, default="data/EUR_USD_ticks.csv", help="Path to raw ticks CSV")
    parser.add_argument("--output", type=str, default="saved_models/adversarial_attacker", help="Base output path for the adversarial model")
    parser.add_argument("--future_horizon", type=int, default=10, help="Future horizon in bars used to define adversarial vulnerability")
    parser.add_argument("--loss_threshold", type=float, default=0.002, help="Minimum adverse return to label a period vulnerable")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        logger.error("Features file not found", path=args.features)
        return 1

    if not os.path.exists(args.raw):
        logger.error("Raw file not found", path=args.raw)
        return 1

    features_df = pd.read_csv(args.features, index_col="timestamp", parse_dates=True)
    raw_df = pd.read_csv(args.raw, index_col="timestamp", parse_dates=True)

    X, y = create_vulnerability_dataset(
        features_df,
        raw_df,
        future_horizon=args.future_horizon,
        loss_threshold=args.loss_threshold,
    )

    config = {
        "device": args.device,
        "hidden_dims": [256, 128, 64],
        "feature_names": features_df.columns.tolist(),
    }

    attacker = AttackerModel(name="adversarial_attacker", config=config)
    logger.info("Starting adversarial attacker training", epochs=args.epochs, batch_size=args.batch_size)
    attacker.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    output_base = args.output
    attacker.save(output_base)

    logger.info("Adversarial attacker training complete", model_path=output_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
