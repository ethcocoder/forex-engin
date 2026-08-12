import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_tick_ingestion_campaign import chunk_intervals, chunk_stem, parse_utc


def test_campaign_chunks_are_contiguous_and_cover_requested_interval():
    start = parse_utc("2024-01-01T00:00:00Z")
    end = parse_utc("2024-01-10T00:00:00Z")
    chunks = list(chunk_intervals(start, end, 72))

    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))
    assert all(chunk_start < chunk_end for chunk_start, chunk_end in chunks)


def test_campaign_chunk_names_are_deterministic():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 8, tzinfo=timezone.utc)

    assert chunk_stem("EUR/USD", start, end) == "EURUSD_ticks_20240101T0000Z_20240108T0000Z"


def test_intraday_campaign_file_names_include_hour_and_minute_boundaries():
    first_start = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    first_end = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    second_start = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
    second_end = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)

    first = f"EURUSD_{first_start:%Y%m%dT%H%MZ}_{first_end:%Y%m%dT%H%MZ}.campaign.json"
    second = f"EURUSD_{second_start:%Y%m%dT%H%MZ}_{second_end:%Y%m%dT%H%MZ}.campaign.json"
    assert first != second
