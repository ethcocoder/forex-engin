"""Cost-aware research backtesting for out-of-sample prediction diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchBacktestConfig:
    """Explicit assumptions for a label-aligned research backtest.

    These parameters are scenario assumptions, not broker quotes. A result is
    deliberately labelled research-only until rerun against executable bid/ask
    prices, broker commissions, financing, margin and actual fill records.
    """

    initial_equity: float = 100_000.0
    signal_threshold: float = 0.0
    position_fraction: float = 0.10
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    commission_bps: float = 0.0
    max_drawdown: float = 0.10
    annualization_factor: float = 252.0

    def __post_init__(self) -> None:
        if self.initial_equity <= 0.0:
            raise ValueError("initial_equity must be positive.")
        if not 0.0 < self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be in (0, 1].")
        if not 0.0 < self.max_drawdown < 1.0:
            raise ValueError("max_drawdown must be in (0, 1).")
        if min(self.half_spread_bps, self.slippage_bps, self.commission_bps) < 0.0:
            raise ValueError("Cost assumptions cannot be negative.")

    @property
    def one_way_cost_fraction(self) -> float:
        return (
            self.half_spread_bps + self.slippage_bps + self.commission_bps
        ) / 10_000.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchBacktestResult:
    """Audit-friendly result from a cost-aware research backtest."""

    metrics: dict[str, float]
    equity_curve: pd.Series
    events: pd.DataFrame
    halted_at: str | None


def _annualized_sharpe(returns: pd.Series, annualization_factor: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(annualization_factor) * returns.mean() / std)


def run_label_aligned_backtest(
    oos_predictions: pd.DataFrame,
    config: ResearchBacktestConfig,
) -> ResearchBacktestResult:
    """Evaluate OOS forecasts against held-out forward-return labels.

    Required columns are `prediction` and `target`, where target is a **log
    return** built by `ForwardReturnLabelSpec`. The function only uses OOS rows;
    it applies cost once per unit of position turnover and stops taking new risk
    after the configured peak-to-trough drawdown limit is breached.
    """
    required = {"prediction", "target"}
    missing = required.difference(oos_predictions.columns)
    if missing:
        raise ValueError(f"Missing OOS backtest columns: {sorted(missing)}.")
    if not isinstance(oos_predictions.index, pd.DatetimeIndex):
        raise ValueError("OOS predictions must use a timestamp DatetimeIndex.")
    if not oos_predictions.index.is_monotonic_increasing:
        raise ValueError("OOS predictions must be chronological.")

    data = oos_predictions.loc[:, ["prediction", "target"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        raise ValueError("No finite OOS predictions and targets are available.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("OOS predictions contain non-finite values.")

    raw_signal = np.where(
        data["prediction"] > config.signal_threshold,
        1.0,
        np.where(data["prediction"] < -config.signal_threshold, -1.0, 0.0),
    )
    desired_position = raw_signal * config.position_fraction
    actual_position = np.zeros(len(data), dtype=float)
    turnover = np.zeros(len(data), dtype=float)
    gross_return = np.zeros(len(data), dtype=float)
    cost_return = np.zeros(len(data), dtype=float)
    net_return = np.zeros(len(data), dtype=float)
    equity = np.zeros(len(data), dtype=float)
    drawdown = np.zeros(len(data), dtype=float)
    halted = np.zeros(len(data), dtype=bool)

    peak_equity = config.initial_equity
    current_equity = config.initial_equity
    prior_position = 0.0
    halted_at: str | None = None

    for row_number, (_, row) in enumerate(data.iterrows()):
        if halted_at is not None:
            desired = 0.0
            halted[row_number] = True
        else:
            desired = float(desired_position[row_number])
        turnover[row_number] = abs(desired - prior_position)
        actual_position[row_number] = desired
        # The target is a log return from the contract. Convert it to simple
        # return before applying fractional notional and transaction costs.
        gross_return[row_number] = desired * float(np.expm1(row["target"]))
        cost_return[row_number] = turnover[row_number] * config.one_way_cost_fraction
        net_return[row_number] = gross_return[row_number] - cost_return[row_number]
        current_equity *= 1.0 + net_return[row_number]
        peak_equity = max(peak_equity, current_equity)
        drawdown[row_number] = 1.0 - current_equity / peak_equity
        equity[row_number] = current_equity
        if halted_at is None and drawdown[row_number] >= config.max_drawdown:
            halted_at = data.index[row_number].isoformat()
            halted[row_number] = True
        prior_position = desired

    equity_curve = pd.Series(equity, index=data.index, name="equity")
    events = pd.DataFrame(
        {
            "prediction": data["prediction"],
            "target_log_return": data["target"],
            "desired_position": desired_position,
            "position": actual_position,
            "turnover": turnover,
            "gross_return": gross_return,
            "cost_return": cost_return,
            "net_return": net_return,
            "equity": equity,
            "drawdown": drawdown,
            "halted": halted,
        },
        index=data.index,
    )
    trade_count = int((turnover > 0.0).sum())
    metrics = {
        "initial_equity": config.initial_equity,
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": float(equity_curve.iloc[-1] / config.initial_equity - 1.0),
        "max_drawdown": float(events["drawdown"].max()),
        "annualized_sharpe": _annualized_sharpe(
            events["net_return"], config.annualization_factor
        ),
        "mean_turnover": float(events["turnover"].mean()),
        "estimated_cost_return": float(events["cost_return"].sum()),
        "trade_events": float(trade_count),
        "halted": float(halted_at is not None),
    }
    return ResearchBacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        events=events,
        halted_at=halted_at,
    )


def evaluate_threshold_sensitivity(
    oos_predictions: pd.DataFrame,
    base_config: ResearchBacktestConfig,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    """Evaluate fixed signal thresholds as a post-hoc robustness diagnostic.

    The returned table is not a model-selection result: thresholds are evaluated
    on already-held-out predictions and therefore cannot promote a candidate on
    their own. Any future threshold must be specified on an earlier validation
    period and confirmed on a separate untouched holdout.
    """
    from dataclasses import replace

    if not thresholds:
        raise ValueError("At least one signal threshold is required.")
    rows: list[dict[str, float | int | bool]] = []
    for threshold in thresholds:
        if threshold < 0.0:
            raise ValueError("Signal thresholds cannot be negative.")
        result = run_label_aligned_backtest(
            oos_predictions, replace(base_config, signal_threshold=float(threshold))
        )
        active_observations = int((result.events["position"] != 0.0).sum())
        rows.append(
            {
                "signal_threshold": float(threshold),
                "active_observations": active_observations,
                "active_fraction": float(active_observations / len(result.events)),
                **result.metrics,
                "post_hoc_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values("signal_threshold").reset_index(drop=True)


def evaluate_chronological_subperiods(
    oos_predictions: pd.DataFrame,
    config: ResearchBacktestConfig,
    n_periods: int = 3,
) -> pd.DataFrame:
    """Evaluate one fixed policy across chronological OOS subperiods.

    This is a robustness check, not a new optimisation. Each subperiod is
    evaluated independently, making it visible when a full-sample metric is
    concentrated in one historical segment.
    """
    if n_periods < 2:
        raise ValueError("n_periods must be at least two.")
    ordered = oos_predictions.sort_index()
    if len(ordered) < n_periods:
        raise ValueError("Not enough OOS rows for the requested subperiod count.")

    rows: list[dict[str, float | int | str]] = []
    for period_number, positions in enumerate(np.array_split(np.arange(len(ordered)), n_periods)):
        segment = ordered.iloc[positions]
        result = run_label_aligned_backtest(segment, config)
        active_observations = int((result.events["position"] != 0.0).sum())
        rows.append(
            {
                "period": period_number + 1,
                "start_timestamp": segment.index[0].isoformat(),
                "end_timestamp": segment.index[-1].isoformat(),
                "observations": int(len(segment)),
                "active_observations": active_observations,
                "active_fraction": float(active_observations / len(segment)),
                **result.metrics,
            }
        )
    return pd.DataFrame(rows)
