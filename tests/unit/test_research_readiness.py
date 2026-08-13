from __future__ import annotations

from research.readiness import ResearchReadinessPolicy, assess_research_readiness


def _metadata(ic: float = 0.05, directional_accuracy: float = 0.52) -> dict:
    return {
        "aggregate_metrics": {
            "information_coefficient": ic,
            "directional_accuracy": directional_accuracy,
        },
        "execution_ready": False,
    }


def _backtest(sharpe: float = 0.6, drawdown: float = 0.05, cost_bps: float = 1.0) -> dict:
    return {
        "metrics": {
            "annualized_sharpe": sharpe,
            "max_drawdown": drawdown,
            "halted": 0.0,
        },
        "backtest_config": {
            "half_spread_bps": cost_bps,
            "slippage_bps": 0.0,
            "commission_bps": 0.0,
        },
        "execution_ready": False,
    }


def test_readiness_policy_passes_only_all_required_evidence() -> None:
    result = assess_research_readiness(_metadata(), _backtest())
    assert result["passed_for_paper_candidate_review"] is True
    assert result["live_trading_authorised"] is False
    assert result["next_state"] == "paper_candidate_review"


def test_readiness_policy_blocks_weak_prediction_quality() -> None:
    result = assess_research_readiness(_metadata(ic=0.01), _backtest())
    assert result["passed_for_paper_candidate_review"] is False
    assert result["criteria"]["out_of_sample_information_coefficient"] is False
    assert result["next_state"] == "research_iteration_required"


def test_readiness_policy_requires_cost_scenario_by_default() -> None:
    result = assess_research_readiness(_metadata(), _backtest(cost_bps=0.0))
    assert result["passed_for_paper_candidate_review"] is False
    assert result["criteria"]["explicit_nonzero_cost_assumption"] is False


def test_readiness_policy_respects_custom_thresholds() -> None:
    result = assess_research_readiness(
        _metadata(ic=0.02, directional_accuracy=0.49),
        _backtest(sharpe=0.1, drawdown=0.08),
        ResearchReadinessPolicy(
            minimum_information_coefficient=0.01,
            minimum_directional_accuracy=0.48,
            minimum_annualized_sharpe=0.0,
            maximum_drawdown=0.10,
        ),
    )
    assert result["passed_for_paper_candidate_review"] is True
