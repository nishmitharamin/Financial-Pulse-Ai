import yfinance as yf
import pandas as pd

def fetch_market_data(ticker, period="1mo", interval="1h"):
    """
    Pulls real-time financial data from Yahoo Finance API.
    
    Parameters:
    - ticker: Stock/Crypto symbol (e.g., 'AAPL' or 'BTC-USD')
    - period: Data range (e.g., '1mo', '6mo', '1y')
    - interval: Data frequency (e.g., '1h', '1d')
    
    Returns:
    - pd.DataFrame: Cleaned market data or None if failed.
    """
    try:
        # Fetching data using the yfinance library
        data = yf.download(ticker, period=period, interval=interval)
        
        # Validation: Ensure data is not empty
        if data.empty:
            print(f"Warning: No data found for ticker {ticker}")
            return None
            
        # Standardize: Drop any missing values to prevent math errors
        data = data.dropna()
        
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None