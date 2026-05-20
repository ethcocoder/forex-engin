import pandas as pd
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()


class RegimeStressTester:
    """
    Evaluates strategy performance across different market regimes (e.g. ranging vs trending).
    Segments files by regime index columns (from the HMM Regime model).
    """

    def __init__(self) -> None:
        logger.info("RegimeStressTester initialized")

    def run_regime_test(
        self,
        full_df: pd.DataFrame,
        backtest_runner: Any,
        regime_col: str = "regime"
    ) -> Dict[int, Dict[str, Any]]:
        """
        Segments backtest results by regime state index.
        """
        if regime_col not in full_df.columns:
            logger.warning(f"Regime column '{regime_col}' not found in dataframe. Injecting mock regime for safety.")
            # Injecting mock alternating regimes 0 and 1
            full_df[regime_col] = (pd.Series(range(len(full_df))) % 2).values

        unique_regimes = full_df[regime_col].unique()
        results = {}

        # Cache original data
        original_data = backtest_runner.data_handler.data.copy()

        for regime in unique_regimes:
            logger.info("Evaluating regime segment", regime=int(regime))
            
            # Extract bars matching this specific regime
            regime_df = full_df[full_df[regime_col] == regime]
            
            if len(regime_df) < 50: # Ignore tiny segments
                continue
                
            # Override data handler
            backtest_runner.data_handler.data[backtest_runner.config.get("primary_pair", "EURUSD")] = regime_df
            
            try:
                run_res = backtest_runner.run()
                results[int(regime)] = {
                    "regime_index": int(regime),
                    "samples": len(regime_df),
                    "metrics": run_res["performance"],
                    "trades": run_res["trades"]
                }
            except Exception as e:
                logger.error("Failed to run regime slice", regime=regime, error=str(e))

        # Restore original data
        backtest_runner.data_handler.data = original_data
        
        return results
