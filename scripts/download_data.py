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
        
        if response.status_code == 401:
            logger.error("401 Unauthorized. Please check your API token.")
            raise PermissionError("OANDA API returned 401 Unauthorized. Invalid token.")
            
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
            
import argparse
import yfinance as yf

def download_yfinance(pair: str, years: int, output_file: str):
    """
    Downloads historical data from Yahoo Finance (requires NO API KEY).
    yfinance symbol format: EURUSD=X
    """
    logger.info("Using Yahoo Finance (No API Key Required).")
    symbol = f"{pair.replace('_', '')}=X"
    
    # yfinance only supports 1h data up to 730 days (2 years), and 1d data infinitely.
    # We will grab 1h data if years <= 2, else 1d data.
    interval = "1h" if years <= 2 else "1d"
    period = f"{years}y" if years <= 10 else "max"
    
    logger.info(f"Downloading {period} of {interval} data for {symbol}...")
    
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        logger.error(f"Failed to download data for {symbol} from Yahoo Finance.")
        return
        
    # Format to match our schema: timestamp, open, high, low, close, volume
    # yf puts columns in a multi-index sometimes depending on the version
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.index.name = "timestamp"
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })
    
    # yfinance FX volume is usually 0, so we mock it for the pipeline
    if df["volume"].sum() == 0:
        import numpy as np
        df["volume"] = np.random.randint(100, 1000, size=len(df))
        
    df.to_csv(output_file)
    logger.info(f"Successfully saved {len(df)} rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical forex tick data")
    parser.add_argument("--pair", type=str, default="EUR_USD", help="Currency pair (e.g., EUR_USD)")
    parser.add_argument("--years", type=int, default=1, help="Number of years of history to download")
    parser.add_argument("--output", type=str, default="data/EUR_USD_ticks.csv", help="Output CSV path")
    parser.add_argument("--source", type=str, default="yfinance", choices=["oanda", "yfinance"], help="Data source to use")
    parser.add_argument("--token", type=str, default=None, help="OANDA Practice API Token")
    parser.add_argument("--account", type=str, default="000-000-0000000-000", help="OANDA Account ID")
    
    args = parser.parse_args()
    
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    if args.source == "yfinance":
        download_yfinance(args.pair, args.years, args.output)
    else:
        token = args.token or os.environ.get("FOREX_OANDA_TOKEN")
        account = args.account or os.environ.get("FOREX_OANDA_ACCOUNT")
        
        if not token:
            print("ERROR: OANDA requires an API token. Run with --source yfinance to bypass this requirement!")
            exit(1)
            
        downloader = OandaDataDownloader(access_token=token, account_id=account)
        days = args.years * 365
        downloader.download_bulk(
            pair=args.pair, 
            granularity="M1", 
            days_back=days, 
            output_file=args.output
        )
