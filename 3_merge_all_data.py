#!/usr/bin/env python3
"""
Script to merge multiple ticker data into a single multi-level dataframe.

This script:
1. Loads the processed stock data with performance metrics from step 2
2. Merges all ticker data into a single dataframe with tickers as columns
3. Creates a multi-level index for columns: (ticker, feature)
4. Saves the merged dataframe to a new directory

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
    DESCRIPTION, STEP_NAME
)
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from utils.process import Process
from utils.logger import log_step_start, log_step_complete, log_info, log_success, log_error, log_warning, log_progress, log_section

# Configuration
CONFIG = CONFIG | {
    INPUT_DIR: "data/2_performance_metrics",
    OUTPUT_DIR: "data/3_merged_data",
    DESCRIPTION: "Merged stock data with all tickers combined by date",
    STEP_NAME: "Merge All Data"
}

def clean_and_prepare_df(df, ticker):
    """
    Clean and prepare a dataframe for merging.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        ticker (str): Ticker symbol
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for merging
    """
    # Make sure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        else:
            log_warning(f"DataFrame for {ticker} has no DatetimeIndex or 'Date' column, skipping")
            return None
    
    # Create multi-level column names with (ticker, feature)
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    
    return df

def merge_dataframes(dataframes):
    """
    Merge multiple dataframes on their DatetimeIndex.
    
    Args:
        dataframes (dict): Dictionary of {ticker: dataframe}
        
    Returns:
        pd.DataFrame: Merged dataframe with multi-level columns
    """
    if not dataframes:
        return None
    
    # Clean and prepare dataframes for merging
    prepared_dfs = {}
    for ticker, df in dataframes.items():
        prepared_df = clean_and_prepare_df(df, ticker)
        if prepared_df is not None:
            prepared_dfs[ticker] = prepared_df
    
    if not prepared_dfs:
        return None
    
    # Merge all dataframes on DatetimeIndex
    tickers = list(prepared_dfs.keys())
    result = prepared_dfs[tickers[0]]
    
    for ticker in tickers[1:]:
        result = result.join(prepared_dfs[ticker], how='outer')
    
    # Sort by date
    result = result.sort_index()
    
    return result

def main():
    """Main function to run the data merge process."""
    start_time = time.time()
    
    # Initialize the process
    log_step_start(CONFIG)
    Process.start_process(CONFIG)
    
    # Display processor configuration
    log_info(f"System has {multiprocessing.cpu_count()} CPU cores available")
    log_info(f"Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Load ticker data from step 2
    log_section("Loading Data")
    log_info(f"Loading ticker data from {CONFIG[INPUT_DIR]}")
    ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not ticker_dataframes:
        log_error(f"No ticker data found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    log_success(f"Loaded {len(ticker_dataframes)} ticker dataframes in {load_time - start_time:.2f} seconds")
    
    # Log data information
    log_section("Data Information")
    total_rows = sum(len(df) for df in ticker_dataframes.values())
    total_columns = sum(len(df.columns) for df in ticker_dataframes.values())
    log_info(f"Total rows across all tickers: {total_rows}")
    log_info(f"Total columns across all tickers: {total_columns}")
    
    # Merge ticker dataframes
    log_section("Merging Data")
    log_info(f"Merging {len(ticker_dataframes)} ticker dataframes by date...")
    merged_df = merge_dataframes(ticker_dataframes)
    
    if merged_df is None or merged_df.empty:
        log_error("Failed to merge ticker dataframes")
        return
    
    merge_time = time.time()
    log_success(f"Successfully merged {len(ticker_dataframes)} ticker dataframes in {merge_time - load_time:.2f} seconds")
    
    # Log merged data information
    tickers_in_merged = len(merged_df.columns.get_level_values(0).unique())
    features_per_ticker = len(merged_df.columns.get_level_values(1).unique()) // tickers_in_merged
    date_range = f"{merged_df.index.min().strftime('%Y-%m-%d')} to {merged_df.index.max().strftime('%Y-%m-%d')}"
    
    log_info(f"Merged data shape: {merged_df.shape}")
    log_info(f"Date range: {date_range}")
    log_info(f"Number of tickers: {tickers_in_merged}")
    log_info(f"Average features per ticker: {features_per_ticker}")
    
    # Check for missing values
    missing_values = merged_df.isna().sum().sum()
    missing_pct = (missing_values / (merged_df.shape[0] * merged_df.shape[1])) * 100
    log_info(f"Missing values: {missing_values} ({missing_pct:.2f}%)")
    
    # Save merged dataframe
    log_section("Saving Data")
    log_info(f"Saving merged data to {CONFIG[OUTPUT_DIR]}")
    
    # Use write_dataframes_to_parquet with a dictionary containing a single dataframe
    success = write_dataframes_to_parquet({'merged_data': merged_df}, CONFIG)
    
    save_time = time.time()
    
    if success:
        log_success(f"Successfully saved merged data to {CONFIG[OUTPUT_DIR]} in {save_time - merge_time:.2f} seconds")
        
        # Save execution metadata using the utility method
        merge_info = {
            "total_tickers_merged": len(ticker_dataframes),
            "date_range": date_range,
            "merged_shape": {
                "rows": merged_df.shape[0],
                "columns": merged_df.shape[1]
            },
            "tickers_included": list(ticker_dataframes.keys()),
            "missing_values": {
                "count": int(missing_values),
                "percentage": float(f"{missing_pct:.2f}")
            },
            "columns_per_level": {
                "tickers": tickers_in_merged,
                "features_per_ticker": features_per_ticker
            }
        }
        
        time_markers = {
            "load": load_time,
            "merge": merge_time,
            "save": save_time
        }
        
        Process.save_execution_metadata(
            config=CONFIG,
            filename='merge_info.json',
            metadata=merge_info,
            start_time=start_time,
            time_markers=time_markers
        )
        
        log_step_complete(start_time)
    else:
        log_error(f"Failed to save merged data")

if __name__ == "__main__":
    main() 