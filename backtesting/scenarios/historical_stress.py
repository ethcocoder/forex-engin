import pandas as pd
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()


class HistoricalStressTester:
    """
    Evaluates strategy during historical periods of market stress (black swan events).
    Segments files by date ranges of known historical financial events.
    """

    # Event date definitions (Forex specific)
    EVENTS = {
        "CHF_DEPEG_2015": {
            "name": "Swiss Franc Depegging (Black Thursday)",
            "start": "2015-01-14",
            "end": "2015-01-16"
        },
        "BREXIT_VOTE_2016": {
            "name": "Brexit Referendum Night",
            "start": "2016-06-23",
            "end": "2016-06-25"
        },
        "COVID_CRASH_2020": {
            "name": "COVID-19 Financial Liquidity Crash",
            "start": "2020-03-09",
            "end": "2020-03-20"
        }
    }

    def __init__(self) -> None:
        logger.info("HistoricalStressTester initialized")

    def run_stress_test(
        self,
        full_df: pd.DataFrame,
        backtest_runner: Any
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs the backtest specifically sliced on historical black-swan dates.
        """
        results = {}

        for event_key, info in self.EVENTS.items():
            logger.info("Running stress test segment", event=info["name"])
            
            # Slice dataframe by event dates
            event_df = full_df.loc[info["start"]:info["end"]]
            
            if event_df.empty:
                logger.warning(f"No data available for date range {info['start']} to {info['end']}")
                continue

            # Override data handler in runner with sliced data
            backtest_runner.data_handler.data[backtest_runner.config.get("primary_pair", "EURUSD")] = event_df
            
            # Run engine
            try:
                run_res = backtest_runner.run()
                results[event_key] = {
                    "name": info["name"],
                    "start": info["start"],
                    "end": info["end"],
                    "metrics": run_res["performance"],
                    "total_trades": run_res["trades"].get("total_trades", 0)
                }
            except Exception as e:
                logger.error("Failed to run stress test slice", event=event_key, error=str(e))
                
        return results
