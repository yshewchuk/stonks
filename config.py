import multiprocessing

# Common configuration keys
TICKERS = 'tickers'
INPUT_DIR = "input_dir"
OUTPUT_DIR = 'output_dir'
MAX_WORKERS = 'max_workers'
DESCRIPTION = 'description'
STEP_NAME = 'step_name'

# Training/Evaluation split configuration
EVALUATION_ROWS = 'evaluation_rows'
TRAINING_EVALUATION_CONFIG = {
    EVALUATION_ROWS: 400,  # Number of rows to extract for evaluation
}

# Performance calculation configuration
RAW_DATA_USED_COLUMNS = 'raw_data_used_columns'
WINDOWS = 'windows'
LAG_PERIODS = 'lag_periods'
PREDICTIONS_CONFIG = 'predictions_config'
START_DAYS_FUTURE = 'start_days_future'
END_DAYS_FUTURE = 'end_days_future'
PERCENT_CHANGE_BOUNDS = 'percent_change_bounds'

PERFORMANCE_CONFIG = {
    RAW_DATA_USED_COLUMNS: ['Open', 'High', 'Low', 'Close', 'Volume'],
    WINDOWS: [5, 20, 50],
    LAG_PERIODS: [1, 2, 3, 5, 7, 9, 12, 15],
    PREDICTIONS_CONFIG: [{
        START_DAYS_FUTURE: 1,
        END_DAYS_FUTURE: 3,
        PERCENT_CHANGE_BOUNDS: [-7, -3, -1, 1, 3, 7]
    }, {
        START_DAYS_FUTURE: 4,
        END_DAYS_FUTURE: 10,
        PERCENT_CHANGE_BOUNDS: [-10, -4, -2, 2, 4, 10]
    }, {
        START_DAYS_FUTURE: 11,
        END_DAYS_FUTURE: 30,
        PERCENT_CHANGE_BOUNDS: [-12, -5, -3, 3, 5, 12]
    }]
}

# Historical data configuration
HISTORICAL_COLUMN_PREFIXES = 'historical_column_prefixes'

# Historical filter configuration (previously named MERGE_CONFIG)
HISTORICAL_FILTER_CONFIG = {
    HISTORICAL_COLUMN_PREFIXES: [
        "Open", "High", "Low", "Close", "Volume",  # Original price data
        "MA", "Hi", "Lo", "RSI", "MoACD",  # Technical indicators
    ]
}

# Time window configuration
WINDOW_SIZE = 'window_size'
STEP_SIZE = 'step_size'
DROP_WINDOWS_WITH_NA = 'drop_windows_with_na'   

# Time window configuration
TIME_WINDOW_CONFIG = {
    WINDOW_SIZE: 60,           # 60-day windows
    STEP_SIZE: 1,             # Slide by 1 day for each new window
    DROP_WINDOWS_WITH_NA: True # Drop windows containing any NaN values
}

# Scaling configuration
PRICE_COLUMN_TAGS = 'price_column_tags'
VOLUME_PREFIX = 'volume_prefix'
RSI_PREFIX = 'rsi_prefix'
MACD_PREFIX = 'macd_prefix'

SCALING_CONFIG = {
    PRICE_COLUMN_TAGS: ['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo'],
    VOLUME_PREFIX: 'Volume',
    RSI_PREFIX: 'RSI',
    MACD_PREFIX: 'MoACD',
}

# Base configuration
CONFIG = {
    TICKERS: ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
    MAX_WORKERS: min(8, max(multiprocessing.cpu_count() - 1, 1)),  # Use CPU count as a guide
}