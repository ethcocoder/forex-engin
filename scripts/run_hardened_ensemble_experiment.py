"""Train the hardened cross-fitted ensemble on causal research data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.hardened_ensemble import (
    HardenedEnsembleConfig,
    run_hardened_ensemble_walk_forward,
)
from research.labels import ForwardReturnLabelSpec
from research.splits import ExpandingPurgedWalkForwardSplit
from research.training import BaselineTrainingConfig, build_research_matrix


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible, no-order hardened ensemble research experiment."
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
    parser.add_argument("--minimum-meta-rows", type=int, default=256)
    parser.add_argument("--hgb-max-iter", type=int, default=150)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=64)
    parser.add_argument("--conformal-coverage", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args()

    raw, features = _load_csv(args.raw), _load_csv(args.features)
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
    X, y, feature_columns, metadata = build_research_matrix(raw, features, data_config)
    ensemble_config = HardenedEnsembleConfig(
        split=data_config.split,
        hgb_max_iter=args.hgb_max_iter,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
        minimum_meta_rows=args.minimum_meta_rows,
        conformal_coverage=args.conformal_coverage,
        random_seed=args.seed,
    )
    result = run_hardened_ensemble_walk_forward(
        X,
        y,
        ensemble_config,
        args.artifact_root,
        research_metadata={
            "data_config": data_config.to_dict(),
            "research_matrix": metadata,
            "feature_columns": feature_columns,
        },
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
