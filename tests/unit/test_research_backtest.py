from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import (
    ResearchBacktestConfig,
    evaluate_chronological_subperiods,
    evaluate_threshold_sensitivity,
    run_label_aligned_backtest,
)


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


def test_threshold_sensitivity_is_diagnostic_and_reduces_activity() -> None:
    table = evaluate_threshold_sensitivity(
        _predictions(),
        ResearchBacktestConfig(half_spread_bps=1.0, slippage_bps=1.0),
        (0.0, 0.02),
    )
    assert table["post_hoc_only"].all()
    assert table["active_observations"].iloc[1] <= table["active_observations"].iloc[0]
    assert table["active_observations"].iloc[1] == 0


def test_chronological_subperiods_preserve_all_oos_observations() -> None:
    table = evaluate_chronological_subperiods(
        _predictions(), ResearchBacktestConfig(), n_periods=2
    )
    assert table["observations"].sum() == len(_predictions())
    assert table["start_timestamp"].iloc[0] < table["start_timestamp"].iloc[1]
    assert (table["active_observations"] >= 0).all()


def test_backtest_respects_model_abstention() -> None:
    predictions = _predictions().assign(abstain=True)
    result = run_label_aligned_backtest(
        predictions,
        ResearchBacktestConfig(half_spread_bps=1.0, slippage_bps=1.0),
    )
    assert (result.events["position"] == 0.0).all()
    assert (result.events["turnover"] == 0.0).all()
    assert result.metrics["estimated_cost_return"] == 0.0
    assert result.metrics["abstention_rate"] == 1.0
