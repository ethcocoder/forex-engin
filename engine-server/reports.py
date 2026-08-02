"""Tear-sheet and export helpers for the engine service.

Tear sheet metrics come from the engine's own end-of-run report (built from the
PerformanceTracker). If a run hasn't finished, we compute a minimal fallback
from the state ring buffers so the endpoint always answers.
"""

import csv
import io
import json
from typing import Any, Dict, Tuple


def _fallback_metrics(state) -> Dict[str, Any]:
    equity_entries = state.history("equity")
    trades = state.history("trades")

    equity = [e.get("equity", 0.0) for e in equity_entries]
    if len(equity) > 1:
        returns = [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, len(equity))]
        total_return_pct = (equity[-1] / equity[0] - 1.0) * 100.0
        running_max = 0.0
        max_dd = 0.0
        for value in equity:
            running_max = max(running_max, value)
            if running_max > 0:
                max_dd = max(max_dd, (running_max - value) / running_max)
        mean_ret = sum(returns) / len(returns) if returns else 0.0
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns) if returns else 0.0
        sharpe = mean_ret / (variance ** 0.5) if variance > 0 else 0.0
    else:
        total_return_pct = 0.0
        max_dd = 0.0
        sharpe = 0.0

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    win_rate_pct = (len(wins) / len(trades) * 100.0) if trades else 0.0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0.0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "initial_capital": state.account.get("initial_capital", 0.0),
        "ending_capital": equity[-1] if equity else state.account.get("initial_capital", 0.0),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "sharpe": round(sharpe, 4),
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate_pct, 1),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "total_slippage_pips": round(
            sum(t.get("slippage_pips", 0.0) for t in trades), 2
        ),
    }


def tear_sheet(state) -> Dict[str, Any]:
    """Return ``{markdown, metrics}`` for the Reports screen."""
    if getattr(state, "reports", None):
        return state.reports
    return {
        "markdown": "No completed run yet. Start a simulation to generate a tear sheet.",
        "metrics": _fallback_metrics(state),
    }


def export(format_: str, state) -> Tuple[str, bytes]:
    """Return ``(filename, bytes)`` for ``/api/reports/export``."""
    if format_ == "json":
        payload = {
            "trades": state.history("trades"),
            "signals": state.history("signals"),
            "equity": state.history("equity"),
            "orders": state.history("orders"),
            "alerts": state.history("alerts"),
            "account": state.account,
            "reports": state.reports,
        }
        content = json.dumps(payload, indent=2, default=str).encode("utf-8")
        return "forexdesk_export.json", content

    rows = state.history("trades")
    buffer = io.StringIO()
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return "forexdesk_trades.csv", buffer.getvalue().encode("utf-8")
