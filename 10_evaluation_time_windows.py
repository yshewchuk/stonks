#!/usr/bin/env python3
"""
Script to split merged evaluation data into time windows.

This script:
1. Loads the merged evaluation data from step 9
2. Splits the data into fixed-length time windows
3. Saves each time window as a separate file

Uses multiprocessing for all operations to maximize performance.
"""

import os
import json
import time
import multiprocessing
from datetime import datetime

import pandas as pd

from config import (
    CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, TIME_WINDOW_CONFIG, 
    WINDOW_SIZE, STEP_SIZE, DROP_WINDOWS_WITH_NA
)
from utils.dataframe import (
    read_parquet_files_from_directory, 
    write_dataframes_to_parquet, 
    create_time_windows
)
from utils.process import Process

# Configuration
CONFIG = CONFIG | TIME_WINDOW_CONFIG | {
    INPUT_DIR: "data/9_evaluation_merged",
    OUTPUT_DIR: f"data/10_evaluation_time_windows",
    "description": f"Evaluation data split into {TIME_WINDOW_CONFIG[WINDOW_SIZE]}-day time windows"
}

def _save_window_to_parquet(args):
    """
    Helper function to save a single time window to a parquet file.
    Used by ProcessPoolExecutor for parallel file writing.
    
    Args:
        args (tuple): Tuple containing (window_df, window_name, output_dir)
        
    Returns:
        tuple: (window_name, success, error_message)
    """
    window_df, window_name, output_dir = args
    
    try:
        # Create a filename from the window name
        filename = f"{window_name}.parquet"
        filepath = os.path.join(output_dir, filename)
        
        # Save the window to a parquet file
        window_df.to_parquet(filepath, index=True, compression='snappy')
        
        return (window_name, True, None)
    except Exception as e:
        return (window_name, False, str(e))

def main():
    """Main function to run the evaluation data time windowing process."""
    start_time = time.time()
    
    # Initialize the process
    print(f"🚀 Starting evaluation data time windowing: {CONFIG[INPUT_DIR]} → {CONFIG[OUTPUT_DIR]}")
    Process.start_process(CONFIG)
    
    # Display processor configuration
    print(f"ℹ️ System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"ℹ️ Using up to {CONFIG[MAX_WORKERS]} worker processes")
    print(f"ℹ️ Creating {CONFIG[WINDOW_SIZE]}-day time windows")
    
    # Load merged evaluation data
    print(f"🔍 Loading merged evaluation data from {CONFIG[INPUT_DIR]} (multiprocessing)")
    data_dict = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not data_dict or "evaluation_historical_data" not in data_dict:
        print(f"❌ Error: No merged evaluation data found in {CONFIG[INPUT_DIR]}")
        return
    
    # Get the merged dataframe
    merged_df = data_dict["evaluation_historical_data"]
    
    load_time = time.time()
    print(f"✅ Loaded merged evaluation data: {merged_df.shape} in {load_time - start_time:.2f} seconds")
    
    # Create time windows
    print(f"🔄 Creating {CONFIG[WINDOW_SIZE]}-day time windows...")
    windows = create_time_windows(merged_df, window_size=CONFIG[WINDOW_SIZE], step_size=CONFIG[STEP_SIZE], dropna=CONFIG[DROP_WINDOWS_WITH_NA])
    
    window_time = time.time()
    print(f"✅ Created {len(windows)} time windows in {window_time - load_time:.2f} seconds")
    
    if not windows:
        print(f"❌ Error: Failed to create time windows or no windows were created")
        return
    
    # Create a dictionary of window DataFrames
    window_dfs = {window.name: window for window in windows}
    
    # Save time windows using multiprocessing
    print(f"💾 Saving {len(window_dfs)} time windows to {CONFIG[OUTPUT_DIR]} (multiprocessing)...")
    success = write_dataframes_to_parquet(window_dfs, CONFIG)
    
    save_time = time.time()
    
    if success:
        print(f"✅ Successfully saved {len(window_dfs)} time windows to {CONFIG[OUTPUT_DIR]} in {save_time - window_time:.2f} seconds")
        
        # Calculate how many tickers are in each window
        tickers_per_window = len(merged_df.columns.get_level_values(0).unique())
        
        # Save additional metadata
        metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'window_info.json')
        window_info = {
            "window_size_days": CONFIG[WINDOW_SIZE],
            "total_windows": len(window_dfs),
            "step_size": CONFIG[STEP_SIZE], # Overlapping windows
            "tickers_per_window": tickers_per_window,
            "features_per_ticker": len(merged_df.columns.get_level_values(1).unique()),
            "multiprocessing_used": True,
            "workers_used": CONFIG[MAX_WORKERS],
            "cpu_cores_available": multiprocessing.cpu_count(),
            "window_date_ranges": {window: windows[i].index[0].strftime('%Y-%m-%d') + " to " + windows[i].index[-1].strftime('%Y-%m-%d') 
                                  for i, window in enumerate(window_dfs.keys())},
            "processing_time_seconds": {
                "loading": round(load_time - start_time, 2),
                "windowing": round(window_time - load_time, 2),
                "saving": round(save_time - window_time, 2),
                "total": round(save_time - start_time, 2)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(window_info, f, indent=2, default=str)
        
        print(f"✅ Saved window information to {metadata_path}")
        print(f"🎉 Total processing time: {save_time - start_time:.2f} seconds")
    else:
        print(f"❌ Error: Failed to save time windows")

if __name__ == "__main__":
    main() 