"""Run a post-hoc, no-order OOS signal-threshold sensitivity diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.backtest import (
    ResearchBacktestConfig,
    evaluate_threshold_sensitivity,
)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one comma-separated threshold is required.")
    if any(threshold < 0.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be non-negative.")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed OOS thresholds; this is post-hoc research only."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=(0.0, 0.0001, 0.00025, 0.0005, 0.001),
    )
    parser.add_argument("--half-spread-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--position-fraction", type=float, default=0.10)
    parser.add_argument("--max-drawdown", type=float, default=0.10)
    parser.add_argument("--annualization-factor", type=float, default=252.0)
    args = parser.parse_args()

    prediction_path = args.experiment_dir / "oos_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing saved OOS predictions: {prediction_path}")
    predictions = pd.read_csv(prediction_path, parse_dates=["timestamp"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
    predictions = predictions.set_index("timestamp").sort_index()
    config = ResearchBacktestConfig(
        signal_threshold=0.0,
        position_fraction=args.position_fraction,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        max_drawdown=args.max_drawdown,
        annualization_factor=args.annualization_factor,
    )
    table = evaluate_threshold_sensitivity(predictions, config, args.thresholds)
    output_csv = args.experiment_dir / "threshold_sensitivity.csv"
    table.to_csv(output_csv, index=False)
    summary = {
        "thresholds": list(args.thresholds),
        "cost_assumptions": config.to_dict(),
        "post_hoc_only": True,
        "promotion_eligible": False,
        "warning": (
            "This grid is a robustness diagnostic on previously held-out data. "
            "It cannot select a threshold for a paper or live deployment."
        ),
        "rows": table.to_dict(orient="records"),
    }
    (args.experiment_dir / "threshold_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
