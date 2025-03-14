"""
Retrieve Ticker Data: Downloads and saves raw stock data
"""

from data_sources.ticker_history import TickerHistory
from utils.process import Process
from datetime import datetime, timedelta
from utils.dataframe import write_dataframes_to_parquet

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

# Start the process (creates backup and writes metadata)
Process.start_process(CONFIG)

print("🚀 Starting Stock Data Extraction...")

# Download data
data = {}
for ticker_symbol in CONFIG[TICKERS]:
    # TickerHistory now always downloads from yfinance without saving to file
    data[ticker_symbol] = TickerHistory.load_dataframe(
        ticker=ticker_symbol,
        start_date=CONFIG[START_DATE],
        end_date=CONFIG[END_DATE]
    )

# Save data using Process utility
write_dataframes_to_parquet(data, CONFIG)

print("\n🎉 Stock Data Extraction Complete!")
print(f"Raw data saved to: {CONFIG[OUTPUT_DIR]}")