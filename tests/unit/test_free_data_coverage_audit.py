import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_free_data_coverage import audit_coverage, load_valid_manifests


def write_manifest(root: Path, start: str, end: str) -> None:
    data_dir = root / "data"
    manifest_dir = data_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    dataset = data_dir / f"EURUSD_{start[11:13]}.csv"
    dataset.write_text("instrument,timestamp,bid,ask\nEURUSD,2024-01-01T00:00:00Z,1.1,1.1001\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    (manifest_dir / f"EURUSD_{start[11:13]}.manifest.json").write_text(
        json.dumps(
            {
                "kind": "real_historical_tick_data",
                "source_class": "free_public_broker_historical_export",
                "research_authorization": "EXPLORATORY_RESEARCH_ONLY",
                "institutional_execution_validation": "DENIED",
                "broker_demo_authorization": "DENIED",
                "live_trading_authorization": "DENIED",
                "instrument": "EURUSD",
                "requested_start_utc": start,
                "requested_end_utc_exclusive": end,
                "dataset_path": dataset.name,
                "dataset_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_free_coverage_audit_marks_one_day_as_exploratory_ready_at_local_one_day_gate(tmp_path):
    write_manifest(tmp_path, "2024-01-01T00:00:00Z", "2024-01-01T12:00:00Z")
    write_manifest(tmp_path, "2024-01-01T12:00:00Z", "2024-01-02T00:00:00Z")

    report = audit_coverage(load_valid_manifests(tmp_path / "data" / "manifests"), ["EURUSD"], min_research_days=1)

    assert report["status"] == "EXPLORATORY_COVERAGE_READY"
    assert report["training_authorization"] == "EXPLORATORY_ONLY"
    assert report["live_trading_authorization"] == "DENIED"
    assert report["coverage"]["EURUSD"]["covered_hours"] == 24


def test_free_coverage_audit_blocks_training_when_local_coverage_gate_is_not_met(tmp_path):
    write_manifest(tmp_path, "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")

    report = audit_coverage(load_valid_manifests(tmp_path / "data" / "manifests"), ["EURUSD"], min_research_days=90)

    assert report["status"] == "INSUFFICIENT_COVERAGE"
    assert report["training_authorization"] == "DENIED"
    assert report["coverage"]["EURUSD"]["covered_days"] == 1 / 24
