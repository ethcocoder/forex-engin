#!/usr/bin/env python3
"""Blocked live-trading entry point.

Live capital execution is deliberately disabled until the independent production
readiness gates, authenticated broker-demo trial, and explicit operator approval
are complete. This entry point must not initialise a broker session or submit an
order under any circumstances.
"""

from __future__ import annotations

import argparse
import sys


LIVE_EXECUTION_DISABLED_REASON = (
    "LIVE EXECUTION IS DISABLED. Forex Engin remains a research and broker-demo "
    "candidate. Complete all evidence gates in MASTER_PRODUCTION_READINESS_PLAN.md, "
    "including five years of manifest-verified real tick data, purged walk-forward "
    "evaluation, and a monitored 30-day broker-demo trial, before proposing any "
    "change to this lock."
)


def run_live_trading(_: str) -> None:
    """Fail closed before any credential, broker, or order-handling code can run."""
    raise RuntimeError(LIVE_EXECUTION_DISABLED_REASON)


def main() -> int:
    parser = argparse.ArgumentParser(description="Blocked live-trading entry point")
    parser.add_argument("--config", default="configs/config.yaml", help="Retained for CLI compatibility")
    parser.parse_args()
    print(LIVE_EXECUTION_DISABLED_REASON, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
