#!/usr/bin/env python3
"""
Script to filter the merged data to keep only historical columns.

This script:
1. Loads the merged data from step 3
2. Filters to keep only historical columns based on configuration
3. Saves the filtered historical data to a new directory

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
    CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, TICKERS,
    HISTORICAL_COLUMN_PREFIXES, HISTORICAL_FILTER_CONFIG, DESCRIPTION, STEP_NAME
)
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from utils.process import Process
from utils.logger import log_step_start, log_step_complete, log_info, log_success, log_error, log_warning, log_progress, log_section

# Configuration
CONFIG = CONFIG | HISTORICAL_FILTER_CONFIG | {
    INPUT_DIR: "data/3_merged_data",
    OUTPUT_DIR: "data/4_historical_data",
    DESCRIPTION: "Filtered historical data for price prediction",
    STEP_NAME: "Filter Historical Data"
}

def filter_historical_columns(df, historical_prefixes):
    """
    Filter DataFrame to keep only columns with historical prefixes.
    
    Args:
        df (pd.DataFrame): DataFrame with multi-level columns (ticker, feature)
        historical_prefixes (list): List of column prefixes to keep
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only historical columns
    """
    if df is None or df.empty:
        return None
    
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        log_error("DataFrame must have multi-level columns (ticker, feature)")
        return None
    
    # Get all tickers
    tickers = df.columns.get_level_values(0).unique()
    
    # Create a list to store columns to keep
    columns_to_keep = []
    
    # For each ticker, find columns matching prefixes
    for ticker in tickers:
        features = df.loc[:, ticker].columns
        
        # Keep columns that start with any of the historical prefixes
        for feature in features:
            if any(feature.startswith(prefix) for prefix in historical_prefixes):
                columns_to_keep.append((ticker, feature))
    
    # Filter the DataFrame to keep only the historical columns
    if not columns_to_keep:
        log_error("No columns match the historical prefixes")
        return None
    
    filtered_df = df.loc[:, columns_to_keep]
    
    return filtered_df

def main():
    """Main function to run the historical data filtering process."""
    start_time = time.time()
    
    # Initialize the process
    log_step_start(CONFIG)
    Process.start_process(CONFIG)
    
    # Display processor configuration
    log_info(f"System has {multiprocessing.cpu_count()} CPU cores available")
    log_info(f"Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Load merged data from step 3
    log_section("Loading Data")
    log_info(f"Loading merged data from {CONFIG[INPUT_DIR]}")
    
    # Since we expect a single merged dataframe, we need to adapt the loading approach
    merged_df = None
    dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not dataframes:
        log_error(f"No data found in {CONFIG[INPUT_DIR]}")
        return
    
    # Find the merged data (should be a single file)
    if 'merged_data' in dataframes:
        merged_df = dataframes['merged_data']
    else:
        # If not found by name, take the first (and should be only) dataframe
        merged_df = next(iter(dataframes.values()))
    
    load_time = time.time()
    log_success(f"Loaded merged data with shape {merged_df.shape} in {load_time - start_time:.2f} seconds")
    
    # Get historical column prefixes from config
    historical_prefixes = CONFIG[HISTORICAL_COLUMN_PREFIXES]
    log_section("Filtering Data")
    log_info(f"Filtering columns using the following historical prefixes: {historical_prefixes}")
    
    # Filter to keep only historical columns
    filtered_df = filter_historical_columns(merged_df, historical_prefixes)
    
    if filtered_df is None or filtered_df.empty:
        log_error("Failed to filter historical columns")
        return
    
    filter_time = time.time()
    log_success(f"Filtered data to {filtered_df.shape[1]} historical columns in {filter_time - load_time:.2f} seconds")
    
    # Log details about filtered data
    tickers_in_filtered = len(filtered_df.columns.get_level_values(0).unique())
    features_per_ticker = len(filtered_df.columns.get_level_values(1).unique()) // tickers_in_filtered
    
    log_info(f"Kept {filtered_df.shape[1]} columns out of {merged_df.shape[1]} ({filtered_df.shape[1]/merged_df.shape[1]*100:.1f}%)")
    log_info(f"Number of tickers: {tickers_in_filtered}")
    log_info(f"Average historical features per ticker: {features_per_ticker}")
    
    # Check for missing values
    missing_values = filtered_df.isna().sum().sum()
    missing_pct = (missing_values / (filtered_df.shape[0] * filtered_df.shape[1])) * 100
    log_info(f"Missing values: {missing_values} ({missing_pct:.2f}%)")
    
    # Save filtered dataframe
    log_section("Saving Data")
    log_info(f"Saving filtered historical data to {CONFIG[OUTPUT_DIR]}")
    
    # Use write_dataframes_to_parquet with a dictionary containing a single dataframe
    success = write_dataframes_to_parquet({'historical_data': filtered_df}, CONFIG)
    
    save_time = time.time()
    
    if success:
        log_success(f"Successfully saved filtered historical data to {CONFIG[OUTPUT_DIR]} in {save_time - filter_time:.2f} seconds")
        
        # Save execution metadata using the utility method
        filter_info = {
            "original_shape": {
                "rows": merged_df.shape[0],
                "columns": merged_df.shape[1]
            },
            "filtered_shape": {
                "rows": filtered_df.shape[0],
                "columns": filtered_df.shape[1]
            },
            "historical_prefixes": historical_prefixes,
            "tickers_included": list(filtered_df.columns.get_level_values(0).unique()),
            "date_range": {
                "start": filtered_df.index.min().strftime('%Y-%m-%d'),
                "end": filtered_df.index.max().strftime('%Y-%m-%d')
            },
            "missing_values": {
                "count": int(missing_values),
                "percentage": float(f"{missing_pct:.2f}")
            },
            "columns_per_level": {
                "tickers": tickers_in_filtered,
                "features_per_ticker": features_per_ticker
            }
        }
        
        time_markers = {
            "load": load_time,
            "filter": filter_time,
            "save": save_time
        }
        
        Process.save_execution_metadata(
            config=CONFIG,
            filename='filter_info.json',
            metadata=filter_info,
            start_time=start_time,
            time_markers=time_markers
        )
        
        log_step_complete(start_time)
    else:
        log_error(f"Failed to save filtered historical data")

if __name__ == "__main__":
    main() 