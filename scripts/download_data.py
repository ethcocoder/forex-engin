import time
import requests
import pandas as pd
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()


class OandaDataDownloader:
    """
    Downloads historical candle data from OANDA v20 API for model training.
    """

    def __init__(self, access_token: str, account_id: str, is_practice: bool = True):
        self.domain = "api-fxpractice.oanda.com" if is_practice else "api-fxtrade.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def fetch_candles(self, pair: str, granularity: str, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Fetch historical candles.
        Granularity: M1 (1 min), H1 (1 hour), D (1 day), etc.
        """
        # Format pair (EURUSD -> EUR_USD)
        instrument = f"{pair[:3]}_{pair[3:]}" if len(pair) == 6 else pair
        
        url = f"https://{self.domain}/v3/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "from": start_time,
            "to": end_time,
            "price": "M" # Midpoint pricing
        }
        
        logger.info(f"Downloading {granularity} data for {instrument} from {start_time} to {end_time}")
        response = self.session.get(url, params=params)
        
        if response.status_code != 200:
            logger.error("Failed to download data", status=response.status_code, error=response.text)
            return pd.DataFrame()
            
        candles = response.json().get("candles", [])
        
        data = []
        for c in candles:
            if c.get("complete"):
                data.append({
                    "timestamp": pd.to_datetime(c["time"]),
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c["volume"])
                })
                
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            
        return df

    def download_bulk(self, pair: str, granularity: str, days_back: int, output_file: str):
        """
        Downloads data in chunks (OANDA limits to 5000 candles per request)
        """
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days_back)
        
        # OANDA max count is 5000. For M1 data, 5000 minutes is ~3.4 days.
        # We step in 3-day chunks to be safe.
        chunk_days = 3
        
        current_start = start_dt
        all_dfs = []
        
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            
            # Format RFC3339
            from_str = current_start.isoformat("T") + "Z"
            to_str = current_end.isoformat("T") + "Z"
            
            df = self.fetch_candles(pair, granularity, from_str, to_str)
            if not df.empty:
                all_dfs.append(df)
                
            current_start = current_end
            time.sleep(0.5) # Respect rate limits
            
        if all_dfs:
            final_df = pd.concat(all_dfs)
            # Remove duplicates just in case
            final_df = final_df[~final_df.index.duplicated(keep='first')]
            final_df.to_csv(output_file)
            logger.info(f"Successfully saved {len(final_df)} rows to {output_file}")
            return final_df
        else:
            logger.warning("No data downloaded")
            return pd.DataFrame()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical forex tick data from OANDA")
    parser.add_argument("--pair", type=str, default="EUR_USD", help="Currency pair (e.g., EUR_USD)")
    parser.add_argument("--years", type=int, default=1, help="Number of years of history to download")
    parser.add_argument("--output", type=str, default="data/EUR_USD_ticks.csv", help="Output CSV path")
    parser.add_argument("--token", type=str, default=None, help="OANDA Practice API Token (or set FOREX_OANDA_TOKEN env var)")
    parser.add_argument("--account", type=str, default="000-000-0000000-000", help="OANDA Account ID (or set FOREX_OANDA_ACCOUNT env var)")
    
    args = parser.parse_args()
    
    import os
    token = args.token or os.environ.get("FOREX_OANDA_TOKEN")
    account = args.account or os.environ.get("FOREX_OANDA_ACCOUNT")
    
    if not token:
        print("ERROR: Please provide an OANDA API token.")
        print("You can pass it via --token YOUR_TOKEN or set the FOREX_OANDA_TOKEN environment variable.")
        print("Example: !FOREX_OANDA_TOKEN='abc123def' python scripts/download_data.py --pair EUR_USD --years 5 --output data/EUR_USD_ticks.csv")
        exit(1)
        
    downloader = OandaDataDownloader(access_token=token, account_id=account)
    
    # Calculate days from years
    days = args.years * 365
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Download data
    downloader.download_bulk(
        pair=args.pair, 
        granularity="M1", 
        days_back=days, 
        output_file=args.output
    )
