"""
ETL - Process Script: Isolates training data by removing recent records
"""

import pandas as pd
import os
from pathlib import Path
from config import CONFIG, OUTPUT_DIR, INPUT_DIR
from utils.process import Process
from utils.dataframe import read_parquet_files_from_directory, truncate_recent_data, write_dataframes_to_parquet

# Configure input and output directories
ROWS_TO_ISOLATE = "rows_to_isolate"

# Update configuration
CONFIG = CONFIG | {
    INPUT_DIR: "data/1_raw_data",
    OUTPUT_DIR: "data/2_isolate_training_data",
    ROWS_TO_ISOLATE: 400
}

# Start the process and write metadata
Process.start_process(CONFIG)

print(f"🔍 Loading ticker data from {CONFIG[INPUT_DIR]}")

# Read all parquet files from the input directory
ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])

if not ticker_dataframes:
    print("❌ No ticker data found in the input directory")
    exit(1)

print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes")

# Dictionary to store truncated dataframes
isolated_dataframes = {}

# Process each ticker dataframe
for ticker, df in ticker_dataframes.items():
    print(f"⌛ Processing {ticker}...")
    
    # Truncate the most recent data
    truncated_df = truncate_recent_data(
        df=df, 
        rows_to_remove=CONFIG[ROWS_TO_ISOLATE],
        min_rows_required=500
    )
    
    if truncated_df is not None:
        isolated_dataframes[ticker] = truncated_df
        print(f"✅ {ticker}: Isolated {len(truncated_df)} rows for training (removed {CONFIG[ROWS_TO_ISOLATE]} most recent rows)")
    else:
        print(f"⚠️ {ticker}: Not enough data to isolate training set")

# Save the isolated dataframes to parquet files
if isolated_dataframes:
    success = write_dataframes_to_parquet(isolated_dataframes, CONFIG)
    if success:
        print(f"✅ Saved {len(isolated_dataframes)} isolated dataframes to {CONFIG[OUTPUT_DIR]}")
    else:
        print("❌ Failed to save isolated dataframes")
else:
    print("❌ No dataframes to save") 