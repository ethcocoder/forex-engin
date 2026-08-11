import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from download_histdata_ticks import parse_archive, parse_month


def build_archive(rows: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("DAT_ASCII_EURUSD_T_202001.csv", rows)
    return buffer.getvalue()


def test_histdata_parser_converts_documented_est_clock_to_utc_and_rejects_crossed_quotes():
    payload = build_archive(
        "20200101 000000000,1.1200,1.1202,10\n"
        "20200101 000001000,1.1203,1.1201,10\n"
        "20200101 000002000,1.1204,1.1206,11\n"
    )

    parsed = parse_archive(payload)

    assert len(parsed) == 2
    assert parsed.attrs["invalid_rows"] == 1
    assert str(parsed["timestamp"].iloc[0]) == "2020-01-01 05:00:00+00:00"
    assert list(parsed["bid"]) == [1.1200, 1.1204]
    assert list(parsed["ask"]) == [1.1202, 1.1206]


def test_histdata_month_parser_requires_a_calendar_month():
    assert parse_month("2020-01") == (2020, 1)
    for invalid in ("202001", "2020-13", "20-01"):
        try:
            parse_month(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {invalid}")
