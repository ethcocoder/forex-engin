"""Evaluate one fixed OOS signal policy across chronological subperiods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.backtest import (
    ResearchBacktestConfig,
    evaluate_chronological_subperiods,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed-policy chronological robustness diagnostic; no broker path is used."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--signal-threshold", type=float, required=True)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--half-spread-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--position-fraction", type=float, default=0.10)
    parser.add_argument("--max-drawdown", type=float, default=0.10)
    parser.add_argument("--annualization-factor", type=float, default=252.0)
    args = parser.parse_args()

    prediction_path = args.experiment_dir / "oos_predictions.csv"
    predictions = pd.read_csv(prediction_path, parse_dates=["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions = predictions.set_index("timestamp").sort_index()
    config = ResearchBacktestConfig(
        signal_threshold=args.signal_threshold,
        position_fraction=args.position_fraction,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        max_drawdown=args.max_drawdown,
        annualization_factor=args.annualization_factor,
    )
    table = evaluate_chronological_subperiods(predictions, config, args.periods)
    output_csv = args.experiment_dir / "policy_subperiods.csv"
    table.to_csv(output_csv, index=False)
    summary = {
        "policy": config.to_dict(),
        "periods": args.periods,
        "post_hoc_only": True,
        "promotion_eligible": False,
        "warning": (
            "This report checks robustness of a fixed threshold on existing OOS data. "
            "It cannot establish deployability or authorise a broker connection."
        ),
        "rows": table.to_dict(orient="records"),
    }
    (args.experiment_dir / "policy_subperiods_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
