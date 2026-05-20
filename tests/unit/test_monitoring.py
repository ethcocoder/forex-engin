import time
import numpy as np
import pytest

from risk.monitoring.alert_manager import AlertLevel, AlertManager
from risk.monitoring.portfolio_monitor import PortfolioMonitor
from risk.monitoring.pnl_attribution import PnLAttribution
from risk.risk_engine import PortfolioState


def test_alert_manager_throttling():
    """
    Verify AlertManager logs alerts and properly applies cooldown throttling.
    """
    manager = AlertManager(cooldown_seconds=1.0)
    
    # First alert should trigger
    triggered_first = manager.trigger(
        level=AlertLevel.WARNING,
        metric_name="drawdown",
        value=0.05,
        message="Drawdown limit near warning"
    )
    assert triggered_first is True
    
    # Immediate repeat alert should be throttled
    triggered_second = manager.trigger(
        level=AlertLevel.WARNING,
        metric_name="drawdown",
        value=0.06,
        message="Drawdown limit near warning 2"
    )
    assert triggered_second is False
    
    # Alert with different metric or level should trigger
    triggered_different_metric = manager.trigger(
        level=AlertLevel.WARNING,
        metric_name="leverage",
        value=26.0,
        message="Leverage warning"
    )
    assert triggered_different_metric is True
    
    triggered_different_level = manager.trigger(
        level=AlertLevel.CRITICAL,
        metric_name="drawdown",
        value=0.05,
        message="Critical drawdown alert"
    )
    assert triggered_different_level is True

    # Sleep to expire cooldown
    time.sleep(1.1)
    
    # Now same alert should trigger again
    triggered_after_cooldown = manager.trigger(
        level=AlertLevel.WARNING,
        metric_name="drawdown",
        value=0.07,
        message="Drawdown warning post-cooldown"
    )
    assert triggered_after_cooldown is True

    # History sizes and filters
    alerts = manager.get_alerts()
    assert len(alerts) == 4  # 4 distinct triggered alerts
    
    critical_alerts = manager.get_alerts(min_level=AlertLevel.CRITICAL)
    assert len(critical_alerts) == 1
    assert critical_alerts[0].metric_name == "drawdown"
    assert critical_alerts[0].level == AlertLevel.CRITICAL

    # Clear cooldowns
    manager.clear_cooldowns()
    triggered_immediately_after_clear = manager.trigger(
        level=AlertLevel.WARNING,
        metric_name="drawdown",
        value=0.08,
        message="Drawdown warning immediate after clear"
    )
    assert triggered_immediately_after_clear is True


def test_portfolio_monitor_limits():
    """
    Verify PortfolioMonitor correctly detects breaches and calculates metrics.
    """
    alert_mgr = AlertManager(cooldown_seconds=100.0)
    monitor = PortfolioMonitor(
        alert_manager=alert_mgr,
        config={
            "max_drawdown": 0.10,
            "max_leverage": 10.0,
            "max_concentration": 0.30,
            "var_confidence": 0.95
        }
    )

    # 1. Normal state
    normal_state = PortfolioState(
        current_equity=10000.0,
        open_positions={"EURUSD": 2000.0, "GBPUSD": -1000.0},
        daily_pnl=0.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
        win_rate=0.5,
        win_loss_ratio=1.0,
        historical_returns=np.random.normal(0.0001, 0.005, 100) # safe
    )
    
    metrics = monitor.update(normal_state)
    assert metrics["drawdown"] == 0.0
    assert metrics["leverage"] == 0.3  # (2000 + 1000) / 10000
    assert metrics["max_concentration"] == 0.2  # 2000 / 10000
    assert len(alert_mgr.get_alerts()) == 0

    # 2. Drawdown Breach
    drawdown_state = PortfolioState(
        current_equity=8900.0,  # 11% drawdown from peak 10000.0
        open_positions={},
        daily_pnl=-1100.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
        win_rate=0.5,
        win_loss_ratio=1.0,
        historical_returns=np.zeros(100)
    )
    metrics_dd = monitor.update(drawdown_state)
    assert metrics_dd["drawdown"] == 0.11
    # Check alert history
    alerts = alert_mgr.get_alerts()
    assert len(alerts) == 1
    assert alerts[0].metric_name == "drawdown"
    assert alerts[0].level == AlertLevel.CRITICAL

    # 3. Leverage Breach
    leverage_state = PortfolioState(
        current_equity=8900.0,
        # 8900 * 11 = 97900 exposure (exceeds 10.0x)
        open_positions={"EURUSD": 97900.0},
        daily_pnl=0.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
        win_rate=0.5,
        win_loss_ratio=1.0,
        historical_returns=np.zeros(100)
    )
    monitor.update(leverage_state)
    alerts = alert_mgr.get_alerts()
    # Should have triggered leverage alert (critical) and concentration warning
    assert len(alerts) == 3
    metric_names = [a.metric_name for a in alerts]
    assert "leverage" in metric_names
    assert "concentration" in metric_names


def test_portfolio_monitor_var():
    """
    Verify PortfolioMonitor Value at Risk logic.
    """
    alert_mgr = AlertManager(cooldown_seconds=100.0)
    monitor = PortfolioMonitor(alert_manager=alert_mgr, config={"var_confidence": 0.95})
    
    # Construct state with highly volatile/negative returns history
    returns = np.concatenate([np.zeros(90), np.full(10, -0.09)])
    risky_state = PortfolioState(
        current_equity=10000.0,
        open_positions={},
        daily_pnl=0.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
        win_rate=0.5,
        win_loss_ratio=1.0,
        historical_returns=returns
    )
    
    metrics = monitor.update(risky_state)
    assert metrics["value_at_risk"] > 0.0
    alerts = alert_mgr.get_alerts()
    assert len(alerts) == 1
    assert alerts[0].metric_name == "value_at_risk"
    assert alerts[0].level == AlertLevel.WARNING


def test_pnl_attribution():
    """
    Verify PnLAttribution decomposes trade profits/losses accurately.
    """
    attribution = PnLAttribution()

    # 1. Attribute trade 1 (profitable, but with some cost)
    # Long trade EURUSD, quantity 100,000
    # Expected gross entry = 1.1000, filled entry = 1.1002 (slippage entry = 0.0002)
    # Expected gross exit = 1.1100, filled exit = 1.1097 (slippage exit = 0.0003)
    # Spread paid = 0.0001
    res1 = attribution.attribute_trade(
        pair="EURUSD",
        direction=1,
        entry_price=1.1002,
        exit_price=1.1097,
        quantity=100000.0,
        slippage_entry=0.0002,
        slippage_exit=0.0003,
        spread_paid=0.0001,
        regime=1
    )

    # realized_pnl = 1 * (1.1097 - 1.1002) * 100,000 = 0.0095 * 100,000 = 950.0
    assert res1["realized_pnl"] == pytest.approx(950.0)
    # slippage_cost = (0.0002 + 0.0003) * 100,000 = 50.0
    assert res1["slippage_cost"] == pytest.approx(50.0)
    # spread_cost = 0.0001 * 100,000 = 10.0
    assert res1["spread_cost"] == pytest.approx(10.0)
    # gross_pnl = 950 + 50 + 10 = 1010.0
    assert res1["gross_pnl"] == pytest.approx(1010.0)

    # 2. Attribute trade 2 (losing trade, short GBPUSD)
    res2 = attribution.attribute_trade(
        pair="GBPUSD",
        direction=-1,
        entry_price=1.2000,
        exit_price=1.2050,
        quantity=50000.0,
        slippage_entry=0.0001,
        slippage_exit=0.0001,
        spread_paid=0.0002,
        regime=0
    )

    # realized_pnl = -1 * (1.2050 - 1.2000) * 50,000 = -0.0050 * 50,000 = -250.0
    assert res2["realized_pnl"] == pytest.approx(-250.0)
    # slippage_cost = (0.0001 + 0.0001) * 50,000 = 10.0
    assert res2["slippage_cost"] == pytest.approx(10.0)
    # spread_cost = 0.0002 * 50,000 = 10.0
    assert res2["spread_cost"] == pytest.approx(10.0)
    # gross_pnl = -250 + 10 + 10 = -230.0
    assert res2["gross_pnl"] == pytest.approx(-230.0)

    # 3. Check cumulative aggregates
    summary = attribution.get_performance_summary()
    assert summary["totals"]["realized_pnl"] == pytest.approx(700.0)  # 950 - 250
    assert summary["totals"]["gross_pnl"] == pytest.approx(780.0)     # 1010 - 230
    assert summary["totals"]["slippage_cost"] == pytest.approx(60.0)  # 50 + 10
    assert summary["totals"]["spread_cost"] == pytest.approx(20.0)    # 10 + 10
    assert summary["trade_count"] == 2

    # Breakdowns
    assert "EURUSD" in summary["by_pair"]
    assert summary["by_pair"]["EURUSD"]["realized_pnl"] == pytest.approx(950.0)
    assert summary["by_regime"][1]["realized_pnl"] == pytest.approx(950.0)
    assert summary["by_regime"][0]["realized_pnl"] == pytest.approx(-250.0)

    # Reset test
    attribution.reset()
    assert attribution.get_performance_summary()["trade_count"] == 0
