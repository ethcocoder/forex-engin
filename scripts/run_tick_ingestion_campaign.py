#!/usr/bin/env python3
"""Run bounded, resumable historical tick-data ingestion campaigns.

The command orchestrates ``download_dukascopy_ticks.py`` in short, independently
manifested UTC chunks. It never creates synthetic fallback data; incomplete
campaigns remain explicitly incomplete and cannot be treated as model-ready.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class CampaignChunk:
    instrument: str
    start_utc: str
    end_utc_exclusive: str
    manifest_path: str
    status: str
    rows: int | None = None
    error: str | None = None


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def chunk_intervals(start: datetime, end: datetime, hours: int) -> Iterable[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + timedelta(hours=hours), end)
        yield cursor, next_cursor
        cursor = next_cursor


def chunk_stem(instrument: str, start: datetime, end: datetime) -> str:
    return f"{instrument.replace('/', '').upper()}_ticks_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a resumable, manifest-verified FX tick-data campaign")
    parser.add_argument("--instruments", default="EURUSD", help="Comma-separated instruments, e.g. EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--start", required=True, help="UTC ISO-8601 start")
    parser.add_argument("--end", required=True, help="UTC ISO-8601 exclusive end")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--chunk-hours", type=int, default=168, help="Hours per independently verified chunk")
    parser.add_argument("--max-chunks-per-run", type=int, default=4, help="Bound work per run; re-run to resume")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    start, end = parse_utc(args.start), parse_utc(args.end)
    if start >= end:
        raise SystemExit("--end must be after --start")
    if not 1 <= args.chunk_hours <= 168:
        raise SystemExit("--chunk-hours must be between 1 and 168")
    if args.max_chunks_per_run < 1:
        raise SystemExit("--max-chunks-per-run must be positive")

    root = Path(__file__).resolve().parents[1]
    downloader = root / "scripts" / "download_dukascopy_ticks.py"
    campaign_dir = args.output / "campaigns"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    instruments = [item.strip().upper().replace("/", "") for item in args.instruments.split(",") if item.strip()]
    if not instruments:
        raise SystemExit("At least one instrument is required")

    chunks: list[CampaignChunk] = []
    eligible: list[tuple[str, datetime, datetime]] = []
    for instrument in instruments:
        for chunk_start, chunk_end in chunk_intervals(start, end, args.chunk_hours):
            manifest_path = args.output / "manifests" / f"{chunk_stem(instrument, chunk_start, chunk_end)}.manifest.json"
            if manifest_path.exists():
                try:
                    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if existing.get("kind") == "real_historical_tick_data" and int(existing.get("rows", 0)) > 0:
                        chunks.append(
                            CampaignChunk(
                                instrument=instrument,
                                start_utc=chunk_start.isoformat(),
                                end_utc_exclusive=chunk_end.isoformat(),
                                manifest_path=str(manifest_path),
                                status="already_validated",
                                rows=int(existing["rows"]),
                            )
                        )
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            eligible.append((instrument, chunk_start, chunk_end))

    for instrument, chunk_start, chunk_end in eligible[: args.max_chunks_per_run]:
        manifest_path = args.output / "manifests" / f"{chunk_stem(instrument, chunk_start, chunk_end)}.manifest.json"
        command = [
            sys.executable,
            str(downloader),
            "--instrument", instrument,
            "--start", chunk_start.isoformat(),
            "--end", chunk_end.isoformat(),
            "--output", str(args.output),
            "--max-hours", str(args.chunk_hours),
            "--timeout-seconds", str(args.timeout_seconds),
            "--retries", str(args.retries),
        ]
        print(f"campaign: {instrument} {chunk_start.isoformat()} → {chunk_end.isoformat()}", flush=True)
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        if completed.returncode == 0 and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks.append(
                CampaignChunk(
                    instrument=instrument,
                    start_utc=chunk_start.isoformat(),
                    end_utc_exclusive=chunk_end.isoformat(),
                    manifest_path=str(manifest_path),
                    status="validated",
                    rows=int(manifest["rows"]),
                )
            )
        else:
            chunks.append(
                CampaignChunk(
                    instrument=instrument,
                    start_utc=chunk_start.isoformat(),
                    end_utc_exclusive=chunk_end.isoformat(),
                    manifest_path=str(manifest_path),
                    status="failed",
                    error=(completed.stderr or completed.stdout)[-2_000:],
                )
            )

    total_planned = len(instruments) * sum(1 for _ in chunk_intervals(start, end, args.chunk_hours))
    campaign = {
        "schema_version": 1,
        "kind": "historical_tick_ingestion_campaign",
        "source": "Dukascopy Historical Data Export",
        "source_reference": "https://www.dukascopy.com/swiss/english/marketwatch/historical/",
        "source_class": "free_public_broker_historical_export",
        "research_authorization": "EXPLORATORY_RESEARCH_ONLY",
        "institutional_execution_validation": "DENIED",
        "broker_demo_authorization": "DENIED",
        "live_trading_authorization": "DENIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instruments": instruments,
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "chunk_hours": args.chunk_hours,
        "total_chunks_planned": total_planned,
        "validated_chunks": sum(item.status in {"validated", "already_validated"} for item in chunks),
        "failed_chunks": sum(item.status == "failed" for item in chunks),
        "remaining_chunks": max(0, total_planned - len(chunks)),
        "model_training_authorization": "DENIED unless validated_chunks equals total_chunks_planned and a separate data-quality audit passes.",
        "chunks": [asdict(item) for item in chunks],
    }
    campaign_name = f"{','.join(instruments)}_{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}.campaign.json"
    campaign_path = campaign_dir / campaign_name
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"campaign": str(campaign_path), "validated": campaign["validated_chunks"], "planned": total_planned, "remaining": campaign["remaining_chunks"]}, indent=2))
    return 0 if campaign["failed_chunks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
