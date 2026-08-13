"""Run a leakage-safe, research-only temporal FX model experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.labels import ForwardReturnLabelSpec
from research.splits import ExpandingPurgedWalkForwardSplit
from research.temporal import TemporalTrainingConfig, run_temporal_walk_forward
from research.training import BaselineTrainingConfig


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a causal temporal FX research experiment; it cannot place orders."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/EUR_USD_ticks.csv"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/research_data/EUR_USD_core_features.csv"),
    )
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--provider", default="repository_csv")
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--entry-lag-bars", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--min-train-size", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args()

    raw = _load_csv(args.raw)
    features = _load_csv(args.features)
    data_config = BaselineTrainingConfig(
        pair=args.pair,
        provider=args.provider,
        label_spec=ForwardReturnLabelSpec(
            horizon_bars=args.horizon_bars,
            entry_lag_bars=args.entry_lag_bars,
        ),
        split=ExpandingPurgedWalkForwardSplit(
            n_splits=args.n_splits,
            label_horizon=args.horizon_bars + args.entry_lag_bars,
            embargo_bars=args.embargo_bars,
            min_train_size=args.min_train_size,
        ),
        random_seed=args.seed,
    )
    temporal_config = TemporalTrainingConfig(
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )
    result = run_temporal_walk_forward(
        raw,
        features,
        data_config,
        temporal_config,
        args.artifact_root,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
