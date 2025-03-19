from datetime import datetime
import yfinance as yf
import pandas as pd
import os
import numpy

from utils.dataframe import print_dataframe_debugging_info, verify_dataframe_structure
from utils.logger import log_info, log_success, log_error, log_warning

RAW_DATA_EXPECTED_COLUMNS = {
    'Open': float,
    'High': float,
    'Low': float,
    'Close': float,
    'Volume': int,
    'Dividends': float,
    'Stock Splits': float
}

class TickerHistory:
    """
    Utility class for downloading ticker data from yfinance.
    """

    @staticmethod
    def load_dataframe(ticker, period=None, start_date=None, end_date=None):
        """
        Downloads ticker data from yfinance.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            period (str, optional): Time period for data download (e.g., "1mo", "1y", "max").
            start_date (datetime, optional): Start date for data download.
            end_date (datetime, optional): End date for data download.
            
        Returns:
            pandas.DataFrame: Dataframe containing ticker history data
        """
        try:
            log_info(f"Downloading data for {ticker} from yfinance...")
            ticker_yf = yf.Ticker(ticker)
            df = ticker_yf.history(period=period, start=start_date, end=end_date)
            log_success(f"Successfully downloaded data for {ticker}")
            
            df = TickerHistory._sanitize_dataframe(df)
            
        except Exception as e:
            log_error(f"Error downloading data for {ticker}: {e}")
            raise e
        
        return df
    
    @staticmethod
    def _sanitize_dataframe(df):
        """
        Sanitizes the dataframe to ensure it has the expected structure and format.
        
        Args:
            df (pandas.DataFrame): Dataframe to sanitize
            
        Returns:
            pandas.DataFrame: Sanitized dataframe
        """
        if not verify_dataframe_structure(df, RAW_DATA_EXPECTED_COLUMNS, expected_index_name='Date', expected_index_dtype=numpy.datetime64):
            log_error(f"Data is in unexpected format")
            raise Exception("Invalid raw ticker data")

        df = df.copy()
        df.dropna(inplace=True)

        df.index = df.index.tz_localize(None)

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        return df 