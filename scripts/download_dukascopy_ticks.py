#!/usr/bin/env python3
"""Download, validate, and catalogue real tick data from Dukascopy.

This utility is intentionally separate from any model-training command. It records
source URLs, raw-file hashes, time bounds, and data-quality statistics so that
training can reject unverified or synthetic input data.

Dukascopy archive convention:
https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM_ZERO_BASED}/{DD}/{HH}h_ticks.bi5

The BI5 format has twenty-byte, big-endian records:
(milliseconds within hour, ask integer, bid integer, ask volume, bid volume).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">IIIff")
RECORD_SIZE = RECORD.size


@dataclass(frozen=True)
class HourManifest:
    instrument: str
    source_url: str
    hour_start_utc: str
    sha256_compressed: str
    compressed_bytes: int
    rows: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None
    invalid_quote_rows: int
    status: str


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 datetime and normalize it to a UTC hour."""
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    if dt.minute or dt.second or dt.microsecond:
        raise ValueError("start and end must be aligned to whole UTC hours")
    return dt


def instrument_scale(instrument: str) -> int:
    """Return the documented integer-price scale for a Forex instrument."""
    compact = instrument.replace("/", "").upper()
    return 1_000 if compact.endswith("JPY") else 100_000


def archive_url(instrument: str, hour: datetime) -> str:
    compact = instrument.replace("/", "").upper()
    # Dukascopy stores months in 0-based form.
    return (
        f"{BASE_URL}/{compact}/{hour.year}/{hour.month - 1:02d}/"
        f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    )


def hour_range(start: datetime, end_exclusive: datetime) -> Iterable[datetime]:
    current = start
    while current < end_exclusive:
        yield current
        current += timedelta(hours=1)


def fetch_archive(url: str, timeout_seconds: int, retries: int) -> bytes | None:
    """Fetch a single compressed archive. A 404 represents an empty/missing hour."""
    request = Request(url, headers={"User-Agent": "Forex-Engin-Research/1.0"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Dukascopy returned HTTP {exc.code} for {url}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Network error retrieving {url}: {exc}") from exc
    raise AssertionError("unreachable")


def decode_bi5(payload: bytes, instrument: str, hour: datetime) -> pd.DataFrame:
    """Decode one BI5 archive and reject malformed lengths and invalid timestamps."""
    try:
        decoded = lzma.decompress(payload, format=lzma.FORMAT_AUTO)
    except lzma.LZMAError as exc:
        raise ValueError("Archive is not a valid LZMA/BI5 payload") from exc

    if len(decoded) % RECORD_SIZE:
        raise ValueError(
            f"Malformed BI5 payload: {len(decoded)} bytes is not divisible by {RECORD_SIZE}"
        )

    scale = instrument_scale(instrument)
    hour_epoch_ms = int(hour.timestamp() * 1_000)
    rows: list[tuple[pd.Timestamp, float, float, float, float]] = []
    for offset in range(0, len(decoded), RECORD_SIZE):
        millis, ask_raw, bid_raw, ask_volume, bid_volume = RECORD.unpack_from(decoded, offset)
        if millis >= 3_600_000:
            raise ValueError(f"Tick timestamp exceeds source hour at record {offset // RECORD_SIZE}")
        rows.append(
            (
                pd.Timestamp(hour_epoch_ms + millis, unit="ms", tz="UTC"),
                bid_raw / scale,
                ask_raw / scale,
                float(bid_volume),
                float(ask_volume),
            )
        )

    return pd.DataFrame(rows, columns=["timestamp", "bid", "ask", "bid_volume", "ask_volume"])


def validate_ticks(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort raw ticks and reject rows that cannot represent executable bid/ask quotes."""
    if df.empty:
        return df, 0

    df = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    invalid = (
        ~df["timestamp"].notna()
        | ~df[["bid", "ask", "bid_volume", "ask_volume"]].notna().all(axis=1)
        | (df["bid"] <= 0)
        | (df["ask"] <= 0)
        | (df["ask"] < df["bid"])
        | (df["bid_volume"] < 0)
        | (df["ask_volume"] < 0)
    )
    invalid_count = int(invalid.sum())
    return df.loc[~invalid].copy(), invalid_count


def write_chunk(df: pd.DataFrame, output: Path, instrument: str, start: datetime, end: datetime) -> Path:
    """Write a self-contained CSV chunk with UTC timestamps and price-derived spread."""
    if df.empty:
        raise ValueError("No valid ticks were returned for the requested range")
    result = df.copy()
    result.insert(0, "instrument", instrument.replace("/", "").upper())
    result["spread"] = result["ask"] - result["bid"]
    result["mid"] = (result["ask"] + result["bid"]) / 2.0

    filename = (
        f"{instrument.replace('/', '').upper()}_ticks_"
        f"{start:%Y%m%dT%H%MZ}_{end:%Y%m%dT%H%MZ}.csv"
    )
    path = output / filename
    result.to_csv(path, index=False)
    return path


def run(args: argparse.Namespace) -> int:
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if start >= end:
        raise ValueError("end must be strictly later than start")
    if end - start > timedelta(hours=args.max_hours):
        raise ValueError(
            f"Requested range exceeds --max-hours={args.max_hours}; split the ingestion into resumable chunks"
        )

    output = Path(args.output).resolve()
    raw_dir = output / "raw" / args.instrument.replace("/", "").upper()
    manifest_dir = output / "manifests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dataframes: list[pd.DataFrame] = []
    hour_manifests: list[HourManifest] = []
    requested_hours = list(hour_range(start, end))
    for hour_number, hour in enumerate(requested_hours, start=1):
        url = archive_url(args.instrument, hour)
        source_path = raw_dir / f"{hour:%Y%m%dT%H}Z.bi5"
        if source_path.exists() and source_path.stat().st_size > 0:
            payload = source_path.read_bytes()
            print(f"[{hour_number}/{len(requested_hours)}] reusing {source_path.name}", flush=True)
        else:
            print(f"[{hour_number}/{len(requested_hours)}] downloading {url}", flush=True)
            payload = fetch_archive(url, args.timeout_seconds, args.retries)
        if payload is None:
            hour_manifests.append(
                HourManifest(
                    instrument=args.instrument.replace("/", "").upper(),
                    source_url=url,
                    hour_start_utc=hour.isoformat(),
                    sha256_compressed="",
                    compressed_bytes=0,
                    rows=0,
                    first_timestamp_utc=None,
                    last_timestamp_utc=None,
                    invalid_quote_rows=0,
                    status="missing_source_hour",
                )
            )
            continue

        decoded = None
        decode_error = None
        for decode_attempt in range(args.retries + 1):
            try:
                decoded = decode_bi5(payload, args.instrument, hour)
                break
            except ValueError as exc:
                decode_error = exc
                # A partial HTTP response must never become a resumable cache entry.
                source_path.unlink(missing_ok=True)
                if decode_attempt >= args.retries:
                    raise
                print(
                    f"[{hour_number}/{len(requested_hours)}] retrying malformed archive {source_path.name}",
                    flush=True,
                )
                payload = fetch_archive(url, args.timeout_seconds, retries=0)
                if payload is None:
                    break

        if payload is None or decoded is None:
            raise ValueError(f"Unable to decode source archive {url}: {decode_error}")
        source_path.write_bytes(payload)
        valid, invalid_count = validate_ticks(decoded)
        dataframes.append(valid)
        hour_manifests.append(
            HourManifest(
                instrument=args.instrument.replace("/", "").upper(),
                source_url=url,
                hour_start_utc=hour.isoformat(),
                sha256_compressed=hashlib.sha256(payload).hexdigest(),
                compressed_bytes=len(payload),
                rows=len(valid),
                first_timestamp_utc=(valid["timestamp"].iloc[0].isoformat() if not valid.empty else None),
                last_timestamp_utc=(valid["timestamp"].iloc[-1].isoformat() if not valid.empty else None),
                invalid_quote_rows=invalid_count,
                status="validated" if not valid.empty else "empty_source_hour",
            )
        )

    nonempty = [frame for frame in dataframes if not frame.empty]
    if not nonempty:
        raise RuntimeError("No valid ticks downloaded; no dataset has been written")

    combined, invalid_final = validate_ticks(pd.concat(nonempty, ignore_index=True))
    chunk_path = write_chunk(combined, output, args.instrument, start, end)
    manifest_path = manifest_dir / f"{chunk_path.stem}.manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "real_historical_tick_data",
        "source": "Dukascopy Historical Data Export",
        "source_reference": "https://www.dukascopy.com/swiss/english/marketwatch/historical/",
        "source_class": "free_public_broker_historical_export",
        "research_authorization": "EXPLORATORY_RESEARCH_ONLY",
        "institutional_execution_validation": "DENIED",
        "broker_demo_authorization": "DENIED",
        "live_trading_authorization": "DENIED",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": args.instrument.replace("/", "").upper(),
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "dataset_path": str(chunk_path.relative_to(output)),
        "dataset_sha256": hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
        "rows": len(combined),
        "columns": list(combined.columns),
        "first_timestamp_utc": combined["timestamp"].iloc[0].isoformat(),
        "last_timestamp_utc": combined["timestamp"].iloc[-1].isoformat(),
        "invalid_quote_rows_removed": invalid_final,
        "hours_requested": len(hour_manifests),
        "hours_validated": sum(item.status == "validated" for item in hour_manifests),
        "hours_missing_or_empty": sum(item.status != "validated" for item in hour_manifests),
        "hour_archives": [asdict(item) for item in hour_manifests],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(chunk_path), "manifest": str(manifest_path), "rows": len(combined)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and validate real Dukascopy BI5 tick archives")
    parser.add_argument("--instrument", default="EURUSD", help="Instrument, e.g. EURUSD or EUR/USD")
    parser.add_argument("--start", required=True, help="UTC hour, e.g. 2024-01-02T00:00:00Z")
    parser.add_argument("--end", required=True, help="Exclusive UTC hour, e.g. 2024-01-02T06:00:00Z")
    parser.add_argument("--output", default="data", help="Root directory for raw archives, CSV chunks, and manifests")
    parser.add_argument("--max-hours", type=int, default=168, help="Safety bound per invocation; use chunks for multi-year ingestion")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
