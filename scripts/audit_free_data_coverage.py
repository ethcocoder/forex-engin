#!/usr/bin/env python3
"""Audit manifest-verified free tick-data coverage before exploratory training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_DENIALS = (
    "institutional_execution_validation",
    "broker_demo_authorization",
    "live_trading_authorization",
)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_valid_manifests(manifest_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(manifest_dir.glob("*.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "real_historical_tick_data":
            continue
        if manifest.get("source_class") != "free_public_broker_historical_export":
            continue
        if manifest.get("research_authorization") != "EXPLORATORY_RESEARCH_ONLY":
            raise ValueError(f"Free data manifest lacks exploratory scope: {path.name}")
        for field in REQUIRED_DENIALS:
            if manifest.get(field) != "DENIED":
                raise ValueError(f"Free data manifest lacks required execution denial ({field}): {path.name}")
        dataset = path.parent.parent / manifest["dataset_path"]
        if not dataset.is_file() or sha256_file(dataset) != manifest.get("dataset_sha256"):
            raise ValueError(f"Free data manifest checksum does not validate: {path.name}")
        records.append((path, manifest))
    return records


def merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    merged = [sorted(intervals)[0]]
    for start, end in sorted(intervals)[1:]:
        prior_start, prior_end = merged[-1]
        if start <= prior_end:
            merged[-1] = (prior_start, max(prior_end, end))
        else:
            merged.append((start, end))
    return merged


def audit_coverage(records: list[tuple[Path, dict[str, Any]]], instruments: list[str], min_research_days: int) -> dict[str, Any]:
    intervals_by_instrument: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    manifests_by_instrument: dict[str, list[str]] = defaultdict(list)
    for path, manifest in records:
        instrument = manifest["instrument"]
        intervals_by_instrument[instrument].append(
            (parse_utc(manifest["requested_start_utc"]), parse_utc(manifest["requested_end_utc_exclusive"]))
        )
        manifests_by_instrument[instrument].append(path.name)

    coverage: dict[str, Any] = {}
    threshold_hours = min_research_days * 24
    for instrument in instruments:
        spans = merge_intervals(intervals_by_instrument[instrument])
        covered_hours = sum((end - start).total_seconds() / 3600 for start, end in spans)
        coverage[instrument] = {
            "manifest_count": len(manifests_by_instrument[instrument]),
            "contiguous_spans": [
                {"start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat(), "hours": (end - start).total_seconds() / 3600}
                for start, end in spans
            ],
            "covered_hours": covered_hours,
            "covered_days": covered_hours / 24,
            "minimum_research_days": min_research_days,
            "meets_minimum_research_coverage": covered_hours >= threshold_hours,
        }

    all_covered = all(item["meets_minimum_research_coverage"] for item in coverage.values())
    return {
        "report_kind": "free_data_coverage_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_class": "free_public_broker_historical_export",
        "research_authorization": "EXPLORATORY_RESEARCH_ONLY",
        "institutional_execution_validation": "DENIED",
        "broker_demo_authorization": "DENIED",
        "live_trading_authorization": "DENIED",
        "training_authorization": "EXPLORATORY_ONLY" if all_covered else "DENIED",
        "status": "EXPLORATORY_COVERAGE_READY" if all_covered else "INSUFFICIENT_COVERAGE",
        "coverage": coverage,
        "disclosure": "Coverage thresholds are local research gates only. Passing them does not validate execution quality or authorize broker-demo/live trading.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manifest-verified free tick-data coverage")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--instruments", required=True, help="Comma-separated symbols, for example EURUSD,GBPUSD")
    parser.add_argument("--min-research-days", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_research_days <= 0:
        raise SystemExit("--min-research-days must be positive")
    instruments = [part.strip().upper().replace("/", "") for part in args.instruments.split(",") if part.strip()]
    if not instruments:
        raise SystemExit("At least one instrument is required")
    report = audit_coverage(load_valid_manifests(args.manifest_dir), instruments, args.min_research_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
