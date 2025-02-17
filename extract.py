"""
ETL - Extract Script: Downloads and saves raw stock data
"""

from model.ticker_history import TickerHistory
from datetime import datetime, timedelta

from config import RAW_DATA_DIR, TICKERS, YEARS_BACK

def download_and_save_data():
    """
    Downloads historical stock data for each ticker
    """

    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(365 * YEARS_BACK))  # Integer cast for timedelta
    
    data = {}
    for ticker_symbol in TICKERS: # Renamed ticker to ticker_symbol for clarity
        history = TickerHistory(ticker_symbol)
        history.download_data(start_date=start_date, end_date=end_date)
        history.save_to_csv(f'{RAW_DATA_DIR}/{ticker_symbol}.csv')
    return data


if __name__ == '__main__':
    print("🚀 Starting Stock Data Extraction...")
    downloaded_data = download_and_save_data()  # Run the data download and save process
    print("\n🎉 Stock Data Extraction Complete!")
    print(f"Raw data saved to: {RAW_DATA_DIR}")