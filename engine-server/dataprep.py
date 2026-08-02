"""Data preparation for the engine service.

Runs the engine's data scripts as subprocesses (they are too heavy / network
bound to run in-process):
  1. ``scripts/download_data.py``      → raw bars in ``data/raw/``
  2. convert the best raw bar file      → ``data/EUR_USD_ticks.csv``
  3. ``scripts/generate_features.py``   → ``data/EUR_USD_features.csv``

Progress lines are streamed to the WebSocket as ``progress`` events. On success
``state.data_ready`` is flipped so ``/api/health`` reports it. Everything runs
in its own thread so the API stays responsive.
"""

import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

ENGINE_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
DATA_DIR = ENGINE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TICKS_FILE = DATA_DIR / "EUR_USD_ticks.csv"
FEATURES_FILE = DATA_DIR / "EUR_USD_features.csv"
RAW_CANDIDATES = (
    DATA_DIR / "raw" / "EURUSD_H1_2y.csv",
    DATA_DIR / "raw" / "EURUSD_D1_20y.csv",
)

_state: Optional[Any] = None
_sink: Optional[Callable[[dict], None]] = None
_lock = threading.Lock()
_active = False


def configure(state: Any, sink: Callable[[dict], None]) -> None:
    """Wire the shared StateStore and the WS broadcast sink."""
    global _state, _sink
    _state = state
    _sink = sink


def is_running() -> bool:
    return _active


def data_ready() -> bool:
    """True when both data files already exist on disk (survives restarts)."""
    return TICKS_FILE.exists() and FEATURES_FILE.exists()


def _emit(message: str) -> None:
    if _sink is not None:
        try:
            _sink({"type": "progress", "ts": time.time(), "data": {"message": message}})
        except Exception:
            pass


def prepare() -> dict:
    """Start data preparation in a background thread (idempotent while running)."""
    global _active
    with _lock:
        if _active:
            return {"started": False, "running": True, "message": "Data prep already running"}
        _active = True
    threading.Thread(target=_run, name="data-prep", daemon=True).start()
    return {"started": True, "running": True, "message": "Data preparation started"}


def _run() -> None:
    global _active
    try:
        if not PYTHON.exists():
            raise RuntimeError(f"Engine venv python not found: {PYTHON}")

        _download()
        _convert_raw()
        _generate_features()

        if _state is not None:
            _state.set_data_ready(data_ready())
        _emit("Data preparation complete. data_ready = true")
    except Exception as exc:
        _emit(f"Data preparation failed: {exc}")
    finally:
        with _lock:
            _active = False


def _run_subprocess(cmd: list) -> None:
    _emit(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ENGINE_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _emit(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _download() -> None:
    _run_subprocess([
        str(PYTHON),
        str(ENGINE_ROOT / "scripts" / "download_data.py"),
    ])


def _convert_raw() -> None:
    """Convert the best raw bar file from data/raw/ into data/EUR_USD_ticks.csv.

    ``download_data.py`` writes yfinance output (index named ``Date``/
    ``Datetime``, mixed-case OHLCV) into ``data/raw/``, not the filename the
    engine expects. Prefer the finer bar file (H1) for a longer tick loop.
    """
    src: Optional[Path] = None
    for candidate in RAW_CANDIDATES:
        if candidate.exists():
            src = candidate
            break
    if src is None:
        raise RuntimeError("download_data.py produced no bar file in data/raw/")

    df = pd.read_csv(src)

    dt_col = None
    for candidate in ("timestamp", "Date", "Datetime"):
        if candidate in df.columns:
            dt_col = candidate
            break
    if dt_col is None:
        dt_col = df.columns[0]
    df = df.rename(columns={dt_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    df.columns = [str(col).lower() for col in df.columns]
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    if "close" not in df.columns:
        raise RuntimeError(f"download_data output missing close column: {list(df.columns)}")

    keep = [col for col in ("open", "high", "low", "close", "volume") if col in df.columns]
    df = df[keep].dropna()
    df = df.sort_index()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TICKS_FILE)
    _emit(f"Converted {src.name} -> {TICKS_FILE.name} ({len(df)} bars)")


def _generate_features() -> None:
    _run_subprocess([
        str(PYTHON),
        str(ENGINE_ROOT / "scripts" / "generate_features.py"),
        "--input", str(TICKS_FILE),
        "--output", str(FEATURES_FILE),
    ])
    if not FEATURES_FILE.exists():
        raise RuntimeError("Feature generation produced no output file")
