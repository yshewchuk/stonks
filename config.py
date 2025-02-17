import os

# Data extraction configuration
RAW_DATA_DIR = 'raw_stock_data'  # Directory where raw data CSVs are stored
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
YEARS_BACK = 14 # Number of years to get data for

RAW_DATA_ALL_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'] # Columns that yfinance history returns
RAW_DATA_USED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']  # Columns that are used from raw dataset

# Data transformation and preparation configuration
TRANSFORMED_DATA_DIR = 'transformed_stock_data'

PREPARED_DATA_DIR = 'prepared_stock_data' # Directory for prepared data
PREPARED_TRAIN_DATA_X_FILE = os.path.join(PREPARED_DATA_DIR, 'train_data_X.npy')
PREPARED_TRAIN_DATES_FILE = os.path.join(PREPARED_DATA_DIR, 'train_dates.npy')
PREPARED_EVAL_DATA_X_FILE = os.path.join(PREPARED_DATA_DIR, 'eval_data_X.npy')
PREPARED_EVAL_DATES_FILE = os.path.join(PREPARED_DATA_DIR, 'eval_dates.npy')

FEATURE_COLUMNS_STOCK_DATA = ['Open', 'High', 'Low', 'Close', 'Volume',  # Base stock data features
                   'MA5', 'MA20', 'MA50',  # Moving Averages
                   'Hi5', 'Hi20', 'Hi50', 'Lo5', 'Lo20', 'Lo50'] # High/Low indicators

RECENT_YEARS_EVAL = 1  # Number of recent years to use for evaluation data
N_STEPS = 60