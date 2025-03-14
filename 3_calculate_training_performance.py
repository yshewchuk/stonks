"""
ETL - Process Script: Calculate performance indicators on training data
"""

import pandas as pd
import os
from pathlib import Path
from config import CONFIG, OUTPUT_DIR, INPUT_DIR, TICKERS
from utils.process import Process
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from transforms.raw_stock_preprocessor import RawStockPreProcessor

RAW_DATA_USED_COLUMNS = 'raw_data_used_columns'
WINDOWS = 'windows'
LAG_PERIODS = 'lag_periods'
PREDICTIONS_CONFIG = 'predictions_config'
START_DAYS_FUTURE = 'start_days_future'
END_DAYS_FUTURE = 'end_days_future'
PERCENT_CHANGE_BOUNDS = 'percent_change_bounds'

# Configure input and output directories
CONFIG = CONFIG | {
    INPUT_DIR: "data/2_isolate_training_data",
    OUTPUT_DIR: "data/3_training_performance",
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

# Start the process and write metadata
Process.start_process(CONFIG)

print(f"🔍 Loading isolated training data from {CONFIG[INPUT_DIR]}")

# Read all parquet files from the input directory
ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])

if not ticker_dataframes:
    print("❌ No ticker data found in the input directory")
    exit(1)

print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes")

# Initialize the RawStockPreProcessor with configuration from CONFIG
preprocessor = RawStockPreProcessor(
    required_columns=CONFIG[RAW_DATA_USED_COLUMNS],
    windows=CONFIG[WINDOWS],
    lag_periods=CONFIG[LAG_PERIODS],
    ppc_configs=CONFIG[PREDICTIONS_CONFIG]
)

# Process all ticker dataframes
print("🔄 Processing ticker data with performance indicators...")
processed_dataframes = {}

for ticker, df in ticker_dataframes.items():
    print(f"⌛ Processing ticker: {ticker}...")
    processed_df = preprocessor.process(df)
    
    if processed_df is not None:
        processed_dataframes[ticker] = processed_df
        print(f"✅ Successfully processed {ticker}: {processed_df.shape}")
    else:
        print(f"❌ Failed to process {ticker}")

if not processed_dataframes:
    print("❌ No processed data available")
    exit(1)

# Save the processed dataframes to parquet files
print(f"💾 Saving processed dataframes to {CONFIG[OUTPUT_DIR]}...")
success = write_dataframes_to_parquet(processed_dataframes, CONFIG)

if success:
    print(f"✅ Saved {len(processed_dataframes)} processed dataframes to {CONFIG[OUTPUT_DIR]}")
    
    # Summarize the features generated for the first ticker (as an example)
    first_ticker = next(iter(processed_dataframes))
    first_df = processed_dataframes[first_ticker]
    print(f"\nExample of features generated for {first_ticker}:")
    print(f"Total features: {len(first_df.columns)}")
    print(f"Total rows: {len(first_df)}")
    
    # Show sample column names by category
    column_categories = {
        "Original": [col for col in first_df.columns if col in CONFIG[RAW_DATA_USED_COLUMNS]],
        "Moving Average": [col for col in first_df.columns if col.startswith("MA")],
        "Hi-Lo": [col for col in first_df.columns if col.startswith("Hi") or col.startswith("Lo")],
        "RSI": [col for col in first_df.columns if col.startswith("RSI")],
        "MACD": [col for col in first_df.columns if "MoACD" in col],
        "Price Change Prob": [col for col in first_df.columns if col.startswith("PPCProb")],
        "Lagged Features": [col for col in first_df.columns if "_Lag" in col]
    }
    
    for category, columns in column_categories.items():
        example_cols = columns[:3] if len(columns) > 3 else columns
        print(f"\n{category} features ({len(columns)} total): {', '.join(example_cols)}...")
else:
    print("❌ Failed to save processed dataframes") 