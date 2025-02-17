from datetime import datetime
import yfinance as yf
import pandas as pd
import os
import numpy

from utils.dataframe import print_dataframe_debugging_info, verify_dataframe_structure

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
    Represents raw ticker data fetched from yfinance.
    Handles downloading, saving to CSV, and loading from CSV for raw ticker data.
    """

    def __init__(self, ticker):
        """
        Initializes RawTickerData object.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            raw_data_dir (str): Directory to save and load raw data CSVs from.
        """
        self.ticker = ticker
        self.raw_df = None  # DataFrame to hold raw data

    def download_data(self, period=None, start_date=None, end_date=None):
        """
        Downloads raw daily adjusted stock data from yfinance.

        Args:
            period (str): Time period for data download (e.g., "1mo", "1y", "max").
                           See yfinance documentation for valid periods.
        Returns:
            bool: True if download successful, False otherwise.
        """
        try:
            print(f"Downloading raw data for {self.ticker} from yfinance...")
            ticker_yf = yf.Ticker(self.ticker)
            self.raw_df = ticker_yf.history(period=period, start=start_date, end=end_date)
            print(f"✅ Successfully downloaded raw data for {self.ticker}")

            self.sanitize()
            return True
        except Exception as e:
            print(f"❌ Error downloading data for {self.ticker}: {e}")
            return False
            

    def sanitize(self):
        if not verify_dataframe_structure(self.raw_df, RAW_DATA_EXPECTED_COLUMNS, expected_index_name='Date', expected_index_dtype=numpy.datetime64):
            print(f"❌ Data is in unexpected format")
            raise Exception("Invalid raw ticker data")

        self.raw_df.dropna(inplace=True)

        self.raw_df.index = self.raw_df.index.tz_localize(None)

        if not self.raw_df.index.is_monotonic_increasing:
            self.raw_df = self.raw_df.sort_index()

        return

    def save_to_csv(self, file_path):
        """
        Saves the downloaded raw data to a CSV file in the specified directory.
        """
        if self.raw_df is not None:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
            self.raw_df.to_csv(file_path)
            print(f"✅ Raw data for {self.ticker} saved to: {file_path}")
            return True
        else:
            print(f"❌ No raw data to save for {self.ticker}. Download data first.")
            return False

    def load_from_csv(self, file_path):
        """
        Loads raw data from a CSV file in the specified raw data directory.
        """
        if not os.path.exists(file_path):
            print(f"❌ CSV data file not found for {self.ticker}: {file_path}")
            return False
        try:
            self.raw_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            print(f"✅ Raw data for {self.ticker} loaded from: {file_path}")
            
            self.sanitize()

            return True
        except Exception as e:
            print(f"❌ Error loading CSV data for {self.ticker} from {file_path}: {e}")
            return False
        
    def dataframe(self):
        """
        Retrieves the inner, standardized dataframe
        """
        return self.raw_df;