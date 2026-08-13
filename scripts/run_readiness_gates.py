"""Assess whether a research experiment merits paper-candidate review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.readiness import ResearchReadinessPolicy, assess_research_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run research promotion gates; this command never authorises live trading."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--minimum-ic", type=float, default=0.03)
    parser.add_argument("--minimum-directional-accuracy", type=float, default=0.50)
    parser.add_argument("--minimum-sharpe", type=float, default=0.50)
    parser.add_argument("--maximum-drawdown", type=float, default=0.10)
    args = parser.parse_args()

    metadata_path = args.experiment_dir / "run_metadata.json"
    backtest_path = args.experiment_dir / "research_backtest_report.json"
    if not metadata_path.exists() or not backtest_path.exists():
        raise FileNotFoundError(
            "Both run_metadata.json and research_backtest_report.json are required."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    backtest_report = json.loads(backtest_path.read_text(encoding="utf-8"))
    policy = ResearchReadinessPolicy(
        minimum_information_coefficient=args.minimum_ic,
        minimum_directional_accuracy=args.minimum_directional_accuracy,
        minimum_annualized_sharpe=args.minimum_sharpe,
        maximum_drawdown=args.maximum_drawdown,
    )
    result = assess_research_readiness(metadata, backtest_report, policy)
    (args.experiment_dir / "readiness_gate_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
