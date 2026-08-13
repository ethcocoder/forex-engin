from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import ResearchBacktestConfig, run_label_aligned_backtest


def _predictions() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "prediction": [0.01, -0.01, 0.01, 0.01],
            "target": [0.001, -0.001, -0.20, 0.001],
        },
        index=index,
    )


def test_backtest_charges_cost_on_position_turnover() -> None:
    result = run_label_aligned_backtest(
        _predictions().iloc[:2],
        ResearchBacktestConfig(
            position_fraction=0.10,
            half_spread_bps=1.0,
            slippage_bps=1.0,
            commission_bps=0.0,
        ),
    )
    assert result.events["turnover"].iloc[0] == pytest.approx(0.10)
    assert result.events["turnover"].iloc[1] == pytest.approx(0.20)
    assert result.events["cost_return"].sum() == pytest.approx(0.30 * 0.0002)
    assert result.metrics["trade_events"] == 2.0


def test_backtest_halts_new_risk_after_drawdown_limit() -> None:
    result = run_label_aligned_backtest(
        _predictions(),
        ResearchBacktestConfig(
            position_fraction=1.0,
            half_spread_bps=0.0,
            slippage_bps=0.0,
            max_drawdown=0.10,
        ),
    )
    assert result.halted_at is not None
    halt_position = int(result.events.index.get_loc(pd.Timestamp(result.halted_at)))
    assert result.events["halted"].iloc[halt_position]
    if halt_position + 1 < len(result.events):
        assert result.events["position"].iloc[halt_position + 1] == 0.0


def test_backtest_rejects_missing_oos_columns() -> None:
    with pytest.raises(ValueError, match="Missing OOS backtest columns"):
        run_label_aligned_backtest(
            pd.DataFrame({"prediction": [0.1]}), ResearchBacktestConfig()
        )


def test_cost_configuration_disallows_negative_assumptions() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ResearchBacktestConfig(half_spread_bps=-1.0)
