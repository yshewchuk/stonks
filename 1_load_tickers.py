"""
ETL - Extract Script: Downloads and saves raw stock data
"""

from data_sources.ticker_history import TickerHistory
from utils.process import Process
from datetime import datetime, timedelta

from config import TICKERS, OUTPUT_DIR, CONFIG

# Define configuration constants
START_DATE = 'start_date'
END_DATE = 'end_date'

# Create configuration dictionary
CONFIG = CONFIG | {
    START_DATE: datetime.today() - timedelta(days=int(365 * 20)),
    END_DATE: datetime.today(),
    OUTPUT_DIR: "data/1_raw_data"
}

def download_and_save_data():
    """
    Downloads historical stock data for each ticker and saves it in parquet format
    """
    # Download data
    data = {}
    for ticker_symbol in TICKERS:
        # TickerHistory now always downloads from yfinance without saving to file
        data[ticker_symbol] = TickerHistory.load_dataframe(
            ticker=ticker_symbol,
            start_date=CONFIG[START_DATE],
            end_date=CONFIG[END_DATE]
        )
    
    # Save data using Process utility
    Process.write_dataframes_to_parquet(data, CONFIG)
    
    # Start the process (creates backup and writes metadata)
    Process.start_process(CONFIG)
    
    return data


if __name__ == '__main__':
    print("🚀 Starting Stock Data Extraction...")
    downloaded_data = download_and_save_data()  # Run the data download and save process
    print("\n🎉 Stock Data Extraction Complete!")
    print(f"Raw data saved to: {CONFIG[OUTPUT_DIR]}")