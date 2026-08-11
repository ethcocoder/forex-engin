#!/usr/bin/env python3
"""Deprecated safety stub.

This repository no longer permits randomly generated prices to be labelled as
industrial, historical, institutional, or training-ready market data. Use
``scripts/download_dukascopy_ticks.py`` and
``scripts/prepare_tick_bars.py`` for manifest-verified historical data.
"""

raise SystemExit(
    "Synthetic market-data generation is intentionally disabled. "
    "Download real tick data with scripts/download_dukascopy_ticks.py instead."
)
