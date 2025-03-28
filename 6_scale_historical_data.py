#!/usr/bin/env python3
"""
Script to scale historical time windows data.

This script:
1. Loads time windows generated by step 5
2. Scales each window using HistoricalDataScaler
3. Saves the scaled windows to the output directory

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
    CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS,
    PRICE_COLUMN_TAGS, VOLUME_PREFIX, RSI_PREFIX, MACD_PREFIX, SCALING_CONFIG,
    DESCRIPTION, STEP_NAME
)
from transforms.historical_data_scaler import HistoricalDataScaler
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from utils.process import Process
from utils.logger import log_step_start, log_step_complete, log_info, log_success, log_error, log_warning, log_progress, log_section

# Configuration - use the shared scaling parameters
CONFIG = CONFIG | SCALING_CONFIG | {
    INPUT_DIR: "data/5_time_windows",
    OUTPUT_DIR: "data/6_scaled_data",
    DESCRIPTION: "Scaled historical time windows data",
    STEP_NAME: "Scale Historical Data"
}

def _scale_data(args):
    """
    Helper function to scale a single window.
    Used by ProcessPoolExecutor for parallel scaling.
    
    Args:
        args (tuple): Tuple containing (window_name, window_df, price_column_tags, volume_prefix, rsi_prefix, macd_prefix)
        
    Returns:
        tuple: (window_name, scaled_df, error_message)
    """
    window_name, window_df, price_column_tags, volume_prefix, rsi_prefix, macd_prefix = args
    
    try:
        # Create a scaler just for this process
        scaler = HistoricalDataScaler(
            price_column_tags=price_column_tags,
            volume_prefix=volume_prefix,
            rsi_prefix=rsi_prefix,
            macd_prefix=macd_prefix
        )
        
        # Scale the window
        scaled_df = scaler.scale_dataframe(window_df)
        
        return (window_name, scaled_df, None)
    except Exception as e:
        return (window_name, None, str(e))

def main():
    """Main function to run the historical data scaling process."""
    start_time = time.time()
    
    # Initialize the process
    log_step_start(CONFIG)
    Process.start_process(CONFIG)
    
    # Display processor configuration
    log_info(f"System has {multiprocessing.cpu_count()} CPU cores available")
    log_info(f"Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get scaling parameters from config
    price_column_tags = CONFIG[PRICE_COLUMN_TAGS]
    volume_prefix = CONFIG[VOLUME_PREFIX]
    rsi_prefix = CONFIG[RSI_PREFIX]
    macd_prefix = CONFIG[MACD_PREFIX]
    
    # Log scaling configuration
    log_section("Scaling Configuration")
    log_info(f"Price column tags: {price_column_tags}")
    log_info(f"Volume prefix: {volume_prefix}")
    log_info(f"RSI prefix: {rsi_prefix}")
    log_info(f"MACD prefix: {macd_prefix}")
    
    # Load time windows from input directory
    log_section("Loading Data")
    log_info(f"Loading time windows from {CONFIG[INPUT_DIR]}")
    
    windows_dict = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not windows_dict:
        log_error(f"No time windows found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    log_success(f"Loaded {len(windows_dict)} time windows in {load_time - start_time:.2f} seconds")
    
    # Scale windows using multiprocessing
    log_section("Scaling Data")
    log_info(f"Scaling {len(windows_dict)} time windows using multiprocessing")
    
    scaled_windows = {}
    total_windows = len(windows_dict)
    
    # Prepare tasks for the process pool
    tasks = [
        (window_name, window_df, price_column_tags, volume_prefix, rsi_prefix, macd_prefix) 
        for window_name, window_df in windows_dict.items()
    ]
    
    # Use ProcessPoolExecutor for parallel scaling
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG[MAX_WORKERS]) as executor:
        futures = {executor.submit(_scale_data, task): task[0] for task in tasks}
        
        # Process results as they complete with progress tracking
        completed = 0
        failed = 0
        
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            window_name, scaled_df, error_message = future.result()
            
            if scaled_df is not None:
                scaled_windows[window_name] = scaled_df
                # Log progress at appropriate intervals
                log_progress(completed, total_windows, "Windows scaled")
            else:
                failed += 1
                log_error(f"Error scaling window {window_name}: {error_message}")
    
    scale_time = time.time()
    
    if scaled_windows:
        log_success(f"Successfully scaled {len(scaled_windows)} windows in {scale_time - load_time:.2f} seconds")
        if failed > 0:
            log_warning(f"Failed to scale {failed} windows")
    else:
        log_error("Failed to scale any windows")
        return
    
    # Save scaled windows using multiprocessing
    log_section("Saving Data")
    log_info(f"Saving {len(scaled_windows)} scaled windows to {CONFIG[OUTPUT_DIR]}")
    
    success = write_dataframes_to_parquet(scaled_windows, CONFIG)
    
    save_time = time.time()
    
    if success:
        log_success(f"Successfully saved {len(scaled_windows)} scaled windows to {CONFIG[OUTPUT_DIR]} in {save_time - scale_time:.2f} seconds")
        
        # Calculate some additional statistics
        avg_window_shape = {
            "rows": sum(df.shape[0] for df in scaled_windows.values()) / len(scaled_windows) if scaled_windows else 0,
            "columns": sum(df.shape[1] for df in scaled_windows.values()) / len(scaled_windows) if scaled_windows else 0
        }
        
        # Prepare metadata for saving
        scaling_info = {
            "scaling_methods": {
                "price": "relative to first open price",
                "volume": "min-max scaling within window",
                "rsi": "fixed range scaling (0-100)",
                "macd": "min-max scaling within window"
            },
            "total_windows_scaled": len(scaled_windows),
            "original_windows": len(windows_dict),
            "failed_windows": failed,
            "avg_window_shape": avg_window_shape,
            "scaling_configuration": {
                "price_column_tags": price_column_tags,
                "volume_prefix": volume_prefix,
                "rsi_prefix": rsi_prefix,
                "macd_prefix": macd_prefix
            }
        }
        
        time_markers = {
            "load": load_time,
            "scale": scale_time,
            "save": save_time
        }
        
        # Save execution metadata using the utility method
        Process.save_execution_metadata(
            config=CONFIG,
            filename='scaling_info.json',
            metadata=scaling_info,
            start_time=start_time,
            time_markers=time_markers
        )
        
        log_step_complete(start_time)
    else:
        log_error(f"Failed to save scaled windows")

if __name__ == "__main__":
    main() 