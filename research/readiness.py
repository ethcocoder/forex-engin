"""Hard promotion gates between model research and any paper-broker integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResearchReadinessPolicy:
    """Minimum evidence required before a model can be considered for paper trading."""

    minimum_information_coefficient: float = 0.03
    minimum_directional_accuracy: float = 0.50
    minimum_annualized_sharpe: float = 0.50
    maximum_drawdown: float = 0.10
    require_nonzero_cost_assumption: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_research_readiness(
    run_metadata: dict[str, Any],
    backtest_report: dict[str, Any],
    policy: ResearchReadinessPolicy = ResearchReadinessPolicy(),
) -> dict[str, Any]:
    """Return auditable pass/fail criteria; never authorises live execution."""
    aggregate = run_metadata.get("aggregate_metrics", {})
    backtest_metrics = backtest_report.get("metrics", {})
    backtest_config = backtest_report.get("backtest_config", {})

    criteria = {
        "out_of_sample_information_coefficient": float(
            aggregate.get("information_coefficient", float("nan"))
        ) >= policy.minimum_information_coefficient,
        "out_of_sample_directional_accuracy": float(
            aggregate.get("directional_accuracy", float("nan"))
        ) >= policy.minimum_directional_accuracy,
        "cost_aware_annualized_sharpe": float(
            backtest_metrics.get("annualized_sharpe", float("nan"))
        ) >= policy.minimum_annualized_sharpe,
        "maximum_drawdown": float(
            backtest_metrics.get("max_drawdown", float("inf"))
        ) <= policy.maximum_drawdown,
        "drawdown_circuit_breaker_not_triggered": not bool(
            backtest_metrics.get("halted", 1.0)
        ),
        "explicit_nonzero_cost_assumption": (
            not policy.require_nonzero_cost_assumption
            or (
                float(backtest_config.get("half_spread_bps", 0.0))
                + float(backtest_config.get("slippage_bps", 0.0))
                + float(backtest_config.get("commission_bps", 0.0))
            )
            > 0.0
        ),
        "research_artifact_is_not_mislabelled_as_execution_ready": (
            run_metadata.get("execution_ready") is False
            and backtest_report.get("execution_ready") is False
        ),
    }
    passed = all(criteria.values())
    return {
        "policy": policy.to_dict(),
        "criteria": criteria,
        "passed_for_paper_candidate_review": passed,
        "live_trading_authorised": False,
        "next_state": (
            "paper_candidate_review" if passed else "research_iteration_required"
        ),
    }
