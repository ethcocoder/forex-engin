"""Evaluate a saved research experiment using explicit cost assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.backtest import ResearchBacktestConfig, run_label_aligned_backtest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a research-only OOS backtest; no broker interface is used."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--position-fraction", type=float, default=0.10)
    parser.add_argument("--half-spread-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--commission-bps", type=float, default=0.0)
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
        initial_equity=args.initial_equity,
        signal_threshold=args.signal_threshold,
        position_fraction=args.position_fraction,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        max_drawdown=args.max_drawdown,
        annualization_factor=args.annualization_factor,
    )
    result = run_label_aligned_backtest(predictions, config)
    result.events.to_csv(args.experiment_dir / "research_backtest_events.csv", index_label="timestamp")
    report = {
        "backtest_config": config.to_dict(),
        "metrics": result.metrics,
        "halted_at": result.halted_at,
        "research_only": True,
        "execution_ready": False,
        "warning": (
            "Costs are scenario assumptions; this result is not a broker-fill, "
            "margin, financing, or live-trading simulation."
        ),
    }
    (args.experiment_dir / "research_backtest_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
