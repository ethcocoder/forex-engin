from typing import Any, Dict, List
import structlog

logger = structlog.get_logger()


class PnLAttribution:
    """
    Decomposes realized trading profits and losses into constituent components:
    - Alpha / Gross Market Return (the ideal price move captured)
    - Slippage Drag (losses due to execution latency / market impact)
    - Bid-Ask Spread cost (inherent transaction friction)
    
    Provides breakdowns aggregated by currency pair and market regime.
    """

    def __init__(self) -> None:
        self.reset()
        logger.info("PnLAttribution module initialized")

    def reset(self) -> None:
        """
        Resets all historical attribution ledgers.
        """
        self.ledger: List[Dict[str, Any]] = []
        self.totals = {
            "realized_pnl": 0.0,
            "gross_pnl": 0.0,
            "slippage_cost": 0.0,
            "spread_cost": 0.0
        }
        self.by_pair: Dict[str, Dict[str, float]] = {}
        self.by_regime: Dict[int, Dict[str, float]] = {}

    def attribute_trade(
        self,
        pair: str,
        direction: int,  # 1 for Long, -1 for Short
        entry_price: float,
        exit_price: float,
        quantity: float,
        slippage_entry: float,  # slip in price units at entry (e.g. entry_fill - entry_target)
        slippage_exit: float,   # slip in price units at exit (e.g. exit_target - exit_fill)
        spread_paid: float,      # average bid-ask spread during trade execution
        regime: int = 0
    ) -> Dict[str, float]:
        """
        Analyzes a single completed trade, decomposes its PnL, and updates the attribution metrics.
        
        Args:
            pair: Currency pair string (e.g. "EURUSD")
            direction: 1 for long, -1 for short.
            entry_price: The final filled entry price.
            exit_price: The final filled exit price.
            quantity: Trade volume in units.
            slippage_entry: Slippage price units (must be >= 0.0 representing cost).
            slippage_exit: Slippage price units (must be >= 0.0 representing cost).
            spread_paid: Spread in price units (must be >= 0.0).
            regime: Integer index of the market regime during execution.
            
        Returns:
            A dictionary containing the decomposition of the trade's PnL.
        """
        # 1. Total realized PnL from filled execution prices
        # For FX, price * quantity yields PnL in quote currency
        realized_pnl = direction * (exit_price - entry_price) * quantity

        # 2. Frictional costs (Slippage cost & Spread cost)
        # We define slippage_entry/exit as price deviations causing drag
        slippage_cost = (slippage_entry + slippage_exit) * quantity
        spread_cost = spread_paid * quantity

        # 3. Gross alpha returns (pnl without friction)
        # Gross = Realized PnL + costs (since costs are negative drags)
        gross_pnl = realized_pnl + slippage_cost + spread_cost

        attribution = {
            "realized_pnl": realized_pnl,
            "gross_pnl": gross_pnl,
            "slippage_cost": slippage_cost,
            "spread_cost": spread_cost
        }

        # Update historical ledger
        record = {
            "pair": pair,
            "direction": direction,
            "quantity": quantity,
            "regime": regime,
            **attribution
        }
        self.ledger.append(record)

        # Update absolute totals
        self.totals["realized_pnl"] += realized_pnl
        self.totals["gross_pnl"] += gross_pnl
        self.totals["slippage_cost"] += slippage_cost
        self.totals["spread_cost"] += spread_cost

        # Update category breakdowns
        self._update_breakdown(self.by_pair, pair, attribution)
        self._update_breakdown(self.by_regime, regime, attribution)

        logger.debug("Trade PnL attributed", pair=pair, realized_pnl=realized_pnl, gross_pnl=gross_pnl)
        return attribution

    def _update_breakdown(self, categories: Dict[Any, Dict[str, float]], key: Any, attribution: Dict[str, float]) -> None:
        """Helper to increment dictionary aggregates."""
        if key not in categories:
            categories[key] = {
                "realized_pnl": 0.0,
                "gross_pnl": 0.0,
                "slippage_cost": 0.0,
                "spread_cost": 0.0,
                "count": 0.0
            }
        
        for k, v in attribution.items():
            categories[key][k] += v
        categories[key]["count"] += 1.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Returns a performance and cost breakdown summary.
        """
        return {
            "totals": self.totals,
            "by_pair": self.by_pair,
            "by_regime": self.by_regime,
            "trade_count": len(self.ledger)
        }
