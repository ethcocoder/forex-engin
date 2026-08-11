#!/usr/bin/env python3
"""Run a non-deployable purged walk-forward research evaluation on real tick bars."""

from __future__ import annotations

import argparse
import hashlib
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.feature_pipeline import LeakageSafeFeaturePipeline
from models.train_harness import ModelTrainingHarness


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest_chain(bars: Path, bars_manifest_path: Path, tick_manifest_path: Path) -> dict:
    bars_manifest = read_json(bars_manifest_path)
    tick_manifest = read_json(tick_manifest_path)
    if bars_manifest.get("kind") != "derived_real_tick_bars":
        raise ValueError("Bar manifest must be a derived_real_tick_bars artifact")
    if tick_manifest.get("kind") != "real_historical_tick_data":
        raise ValueError("Tick manifest must be a verified real_historical_tick_data artifact")
    if bars_manifest.get("output_sha256") != sha256_file(bars):
        raise ValueError("Bar file does not match its recorded output hash")
    if bars_manifest.get("source_tick_sha256") != tick_manifest.get("dataset_sha256"):
        raise ValueError("Bar manifest does not chain to the supplied source tick manifest")
    if bars_manifest.get("instrument") != tick_manifest.get("instrument"):
        raise ValueError("Instrument mismatch across manifest chain")
    return {"bars": bars_manifest, "ticks": tick_manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a baseline on manifest-verified real tick bars")
    parser.add_argument("--bars", required=True, type=Path)
    parser.add_argument("--bars-manifest", required=True, type=Path)
    parser.add_argument("--tick-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", default="5,15,60")
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--purge-bars", type=int, default=None)
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    chain = validate_manifest_chain(args.bars, args.bars_manifest, args.tick_manifest)
    bars = pd.read_csv(args.bars)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="raise", format="ISO8601")
    bars["mid"] = (bars["bid"] + bars["ask"]) / 2.0

    windows = tuple(int(value.strip()) for value in args.windows.split(",") if value.strip())
    pipeline = LeakageSafeFeaturePipeline(window_sizes=windows)
    features = pipeline.compute_features(bars)
    labelled = pipeline.attach_executable_labels(features, horizon_bars=args.horizon_bars)
    result = ModelTrainingHarness(n_estimators=args.n_estimators).evaluate_purged_walk_forward(
        labelled,
        pipeline,
        n_splits=args.n_splits,
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
    )

    report = {
        "report_kind": "historical_research_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result.get("status"),
        "deployment_authorization": "DENIED",
        "reason": "Historical research validation, regardless of result, does not authorize broker-demo or live trading.",
        "manifest_chain": {
            "instrument": chain["bars"]["instrument"],
            "source_tick_sha256": chain["ticks"]["dataset_sha256"],
            "derived_bar_sha256": chain["bars"]["output_sha256"],
            "source_reference": chain["ticks"].get("source_reference"),
        },
        "parameters": {
            "feature_windows": windows,
            "horizon_bars": args.horizon_bars,
            "n_splits": args.n_splits,
            "purge_bars": args.purge_bars or max(windows),
            "embargo_bars": args.embargo_bars,
        },
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "status": result.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
