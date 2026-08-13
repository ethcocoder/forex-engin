"""Run a reproducible, research-only walk-forward Ridge baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.labels import ForwardReturnLabelSpec
from research.splits import ExpandingPurgedWalkForwardSplit
from research.training import BaselineTrainingConfig, run_ridge_walk_forward


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a causal, research-only FX baseline experiment."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/EUR_USD_ticks.csv"))
    parser.add_argument(
        "--features", type=Path, default=Path("data/EUR_USD_features.csv")
    )
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--provider", default="repository_csv")
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--entry-lag-bars", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--min-train-size", type=int, default=1024)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args()

    raw = _load_csv(args.raw)
    features = _load_csv(args.features)
    label_spec = ForwardReturnLabelSpec(
        horizon_bars=args.horizon_bars,
        entry_lag_bars=args.entry_lag_bars,
    )
    split = ExpandingPurgedWalkForwardSplit(
        n_splits=args.n_splits,
        validation_size=args.validation_size,
        label_horizon=args.horizon_bars + args.entry_lag_bars,
        embargo_bars=args.embargo_bars,
        min_train_size=args.min_train_size,
    )
    config = BaselineTrainingConfig(
        pair=args.pair,
        provider=args.provider,
        label_spec=label_spec,
        split=split,
        ridge_alpha=args.ridge_alpha,
        random_seed=args.seed,
    )
    result = run_ridge_walk_forward(
        raw=raw,
        features=features,
        config=config,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
