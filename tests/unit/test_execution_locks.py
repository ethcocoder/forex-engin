import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_live_trading_entry_point_fails_closed_before_broker_initialization():
    completed = subprocess.run(
        [sys.executable, "scripts/run_live_trading.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "LIVE EXECUTION IS DISABLED" in completed.stderr


def test_legacy_replay_contains_a_fail_closed_guard_before_setup():
    source = (ROOT / "scripts" / "run_real_paper_trading.py").read_text(encoding="utf-8")
    entry_index = source.index("def run_real_paper_trading")
    guard_index = source.index("raise RuntimeError", entry_index)
    broker_index = source.index("PaperBroker(config=", entry_index)

    assert guard_index < broker_index
