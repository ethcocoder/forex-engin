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

if __name__ == "__main__":
    # Example usage:
    # 1. Get an API key from OANDA Practice account.
    # 2. Insert it here.
    
    TOKEN = "YOUR_OANDA_PRACTICE_TOKEN"
    ACCOUNT_ID = "YOUR_OANDA_ACCOUNT_ID"
    
    if TOKEN != "YOUR_OANDA_PRACTICE_TOKEN":
        downloader = OandaDataDownloader(access_token=TOKEN, account_id=ACCOUNT_ID)
        
        # Download 30 days of 1-minute data for EURUSD
        downloader.download_bulk(
            pair="EURUSD", 
            granularity="M1", 
            days_back=30, 
            output_file="../data/EURUSD_M1_30D.csv"
        )
    else:
        print("Please edit scripts/download_data.py to add your OANDA API token.")
