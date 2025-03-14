"""
ETL - Process Script: Create 60-day time windows from merged historical data
"""

import pandas as pd
import os
from pathlib import Path
from config import CONFIG, OUTPUT_DIR, INPUT_DIR
from utils.process import Process
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet, create_time_windows

WINDOW_SIZE = 'window_size'
STEP_SIZE = 'step_size'
DROP_WINDOWS_WITH_NA = 'drop_windows_with_na'

# Configure input and output directories
CONFIG = CONFIG | {
    INPUT_DIR: "data/4_training_merge_historical",
    OUTPUT_DIR: "data/5_training_time_windows",
    # Configuration for time windows
    WINDOW_SIZE: 60,           # 60-day windows
    STEP_SIZE: 1,             # Slide by 1 day for each new window
    DROP_WINDOWS_WITH_NA: True # Drop windows containing any NaN values
}

# Start the process and write metadata
Process.start_process(CONFIG)

print(f"🔍 Loading merged historical data from {CONFIG[INPUT_DIR]}")

# Read merged historical data
merged_data = read_parquet_files_from_directory(CONFIG[INPUT_DIR])

if not merged_data or "merged_historical" not in merged_data:
    print("❌ Merged historical data not found")
    exit(1)

# Extract the merged DataFrame
merged_df = merged_data["merged_historical"]
print(f"✅ Loaded merged historical data: {merged_df.shape}")

# Generate time windows
print(f"🔄 Creating {CONFIG[WINDOW_SIZE]}-day time windows...")

windows = create_time_windows(
    merged_df, 
    window_size=CONFIG[WINDOW_SIZE],
    step_size=CONFIG[STEP_SIZE],
    dropna=CONFIG[DROP_WINDOWS_WITH_NA]
)

if not windows:
    print("❌ No time windows could be created")
    exit(1)

print(f"✅ Created {len(windows)} time windows")

# Prepare windows for saving
windows_dict = {f"window_{i+1}_{window.name}": window for i, window in enumerate(windows)}

# Save the time windows
print(f"💾 Saving time windows to {CONFIG[OUTPUT_DIR]}...")
success = write_dataframes_to_parquet(windows_dict, CONFIG)

if success:
    print(f"✅ Saved {len(windows_dict)} time windows to {CONFIG[OUTPUT_DIR]}")
    
    # Print summary
    print("\n📊 Time Windows Summary:")
    print(f"Total windows: {len(windows)}")
    print(f"Window size: {CONFIG[WINDOW_SIZE]} days")
    print(f"Step size: {CONFIG[STEP_SIZE]} days")
    
    # Summarize a sample window
    if windows:
        sample_window = windows[0]
        print(f"\n📈 Sample Window ({sample_window.name}):")
        print(f"Shape: {sample_window.shape}")
        print(f"Date range: {sample_window.index[0]} to {sample_window.index[-1]}")
        print(f"Tickers: {len(sample_window.columns.levels[0])}")
        print(f"Features per ticker: {len(sample_window.columns.levels[1])}")
else:
    print("❌ Failed to save time windows")
    exit(1) 