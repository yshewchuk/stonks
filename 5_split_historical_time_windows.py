#!/usr/bin/env python3
"""
Script to split historical data into time windows.

This script:
1. Loads the filtered historical data from step 4
2. Splits the data into fixed-length time windows
3. Saves each time window as a separate file

Always uses multiprocessing for all operations to maximize performance.
"""

import os
import time
import concurrent.futures
import multiprocessing
from datetime import datetime

import pandas as pd
import numpy as np

from config import (
    CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, TIME_WINDOW_CONFIG, 
    WINDOW_SIZE, STEP_SIZE, DROP_WINDOWS_WITH_NA, DESCRIPTION, STEP_NAME
)
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet, create_time_windows
from utils.process import Process
from utils.logger import log_step_start, log_step_complete, log_info, log_success, log_error, log_warning, log_progress, log_section

# Configuration
CONFIG = CONFIG | TIME_WINDOW_CONFIG | {
    INPUT_DIR: "data/4_historical_data",
    OUTPUT_DIR: "data/5_time_windows",
    DESCRIPTION: f"Historical data split into {TIME_WINDOW_CONFIG[WINDOW_SIZE]}-day time windows",
    STEP_NAME: "Split Historical Time Windows"
}

def main():
    """Main function to run the historical data time windowing process."""
    start_time = time.time()
    
    # Initialize the process
    log_step_start(CONFIG)
    Process.start_process(CONFIG)
    
    # Display processor configuration
    log_info(f"System has {multiprocessing.cpu_count()} CPU cores available")
    log_info(f"Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Load filtered historical data from step 4
    log_section("Loading Data")
    log_info(f"Loading filtered historical data from {CONFIG[INPUT_DIR]}")
    
    # Since we expect a single historical dataframe, we need to adapt the loading approach
    historical_df = None
    dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not dataframes:
        log_error(f"No data found in {CONFIG[INPUT_DIR]}")
        return
    
    # Find the historical data (should be a single file)
    if 'historical_data' in dataframes:
        historical_df = dataframes['historical_data']
    else:
        # If not found by name, take the first (and should be only) dataframe
        historical_df = next(iter(dataframes.values()))
    
    load_time = time.time()
    log_success(f"Loaded historical data with shape {historical_df.shape} in {load_time - start_time:.2f} seconds")
    
    # Display time window configuration
    log_section("Time Window Configuration")
    log_info(f"Window size: {CONFIG[WINDOW_SIZE]} days")
    log_info(f"Step size: {CONFIG[STEP_SIZE]} days")
    log_info(f"Drop windows with NaN: {CONFIG[DROP_WINDOWS_WITH_NA]}")
    
    # Create time windows
    log_section("Creating Time Windows")
    log_info(f"Creating time windows from historical data...")
    
    windows = create_time_windows(
        historical_df, 
        window_size=CONFIG[WINDOW_SIZE], 
        step_size=CONFIG[STEP_SIZE], 
        dropna=CONFIG[DROP_WINDOWS_WITH_NA]
    )
    
    if not windows:
        log_error("Failed to create time windows or no windows were created")
        return
    
    window_time = time.time()
    log_success(f"Created {len(windows)} time windows in {window_time - load_time:.2f} seconds")
    
    # Create a dictionary of window DataFrames
    window_dfs = {window.name: window for window in windows}
    
    # Save time windows using multiprocessing
    log_section("Saving Data")
    log_info(f"Saving {len(window_dfs)} time windows to {CONFIG[OUTPUT_DIR]}")
    
    success = write_dataframes_to_parquet(window_dfs, CONFIG)
    
    save_time = time.time()
    
    if success:
        log_success(f"Successfully saved {len(window_dfs)} time windows to {CONFIG[OUTPUT_DIR]} in {save_time - window_time:.2f} seconds")
        
        # Calculate metadata about windows
        tickers_in_windows = len(historical_df.columns.get_level_values(0).unique())
        features_per_ticker = len(historical_df.columns.get_level_values(1).unique()) // tickers_in_windows
        
        # Create window date ranges dictionary
        window_date_ranges = {}
        for i, window_name in enumerate(window_dfs.keys()):
            window = windows[i]
            start_date = window.index[0].strftime('%Y-%m-%d')
            end_date = window.index[-1].strftime('%Y-%m-%d')
            window_date_ranges[window_name] = f"{start_date} to {end_date}"
        
        # Save execution metadata using the utility method
        window_info = {
            "window_size_days": CONFIG[WINDOW_SIZE],
            "total_windows": len(window_dfs),
            "step_size": CONFIG[STEP_SIZE],
            "tickers_per_window": tickers_in_windows,
            "features_per_ticker": features_per_ticker,
            "features_per_window": len(historical_df.columns),
            "window_date_ranges": window_date_ranges,
            "average_rows_per_window": sum(len(window) for window in windows) / len(windows) if windows else 0
        }
        
        time_markers = {
            "load": load_time,
            "window": window_time,
            "save": save_time
        }
        
        Process.save_execution_metadata(
            config=CONFIG,
            filename='window_info.json',
            metadata=window_info,
            start_time=start_time,
            time_markers=time_markers
        )
        
        log_step_complete(start_time)
    else:
        log_error(f"Failed to save time windows")

if __name__ == "__main__":
    main() 