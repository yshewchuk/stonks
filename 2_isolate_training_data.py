#!/usr/bin/env python3
"""
Script to isolate training data by removing recent records.

This script:
1. Loads the original stock data from step 1
2. Removes the most recent rows from each ticker's data (to be used for evaluation)
3. Saves the training data to a new directory

Uses multiprocessing for all operations to maximize performance.
"""

import os
import json
import time
import concurrent.futures
import multiprocessing
from datetime import datetime

import pandas as pd

from config import (
    CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, 
    EVALUATION_ROWS, TRAINING_EVALUATION_CONFIG
)
from utils.dataframe import read_parquet_files_from_directory, extract_data_range, write_dataframes_to_parquet
from utils.process import Process

# Configuration
CONFIG = CONFIG | TRAINING_EVALUATION_CONFIG | {
    INPUT_DIR: "data/1_raw_data",
    OUTPUT_DIR: "data/2_isolate_training_data",
    "description": "Training data (original data with recent rows removed)"
}

def _extract_training_data(args):
    """
    Helper function to extract training data from a single ticker's DataFrame.
    Used by ProcessPoolExecutor for parallel processing.
    
    Args:
        args (tuple): Tuple containing (ticker, df, eval_rows)
        
    Returns:
        tuple: (ticker, training_df, error_message)
    """
    ticker, df, eval_rows = args
    
    try:
        # Use the extract_data_range function with extract_recent=False to remove recent rows
        training_df = extract_data_range(df, num_rows=eval_rows, extract_recent=False, min_rows_required=500)
        
        if training_df is None:
            return (ticker, None, f"DataFrame has insufficient rows for extraction")
        
        return (ticker, training_df, None)
    except Exception as e:
        return (ticker, None, str(e))

def main():
    """Main function to run the training data isolation process."""
    start_time = time.time()
    
    # Initialize the process
    print(f"🚀 Starting training data isolation: {CONFIG[INPUT_DIR]} → {CONFIG[OUTPUT_DIR]}")
    Process.start_process(CONFIG)
    
    # Display processor configuration
    print(f"ℹ️ System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"ℹ️ Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get evaluation rows from config (to remove from training data)
    eval_rows = CONFIG[EVALUATION_ROWS]
    print(f"ℹ️ Will remove the last {eval_rows} rows from each ticker's data for training")
    
    # Load original stock data using multiprocessing
    print(f"🔍 Loading original stock data from {CONFIG[INPUT_DIR]} (multiprocessing)")
    ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not ticker_dataframes:
        print(f"❌ Error: No stock data found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes in {load_time - start_time:.2f} seconds")
    
    # Extract training data using multiprocessing
    print(f"🔄 Removing last {eval_rows} rows from each ticker (multiprocessing)...")
    training_dataframes = {}
    total_tickers = len(ticker_dataframes)
    
    # Prepare tasks for the process pool
    tasks = [
        (ticker, df, eval_rows) 
        for ticker, df in ticker_dataframes.items()
    ]
    
    # Use ProcessPoolExecutor for parallel extraction
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG[MAX_WORKERS]) as executor:
        futures = {executor.submit(_extract_training_data, task): task[0] for task in tasks}
        
        # Process results as they complete with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            ticker, training_df, error_message = future.result()
            
            if training_df is not None:
                training_dataframes[ticker] = training_df
                # Print progress every 10 tickers or at the end
                if completed % 10 == 0 or completed == total_tickers:
                    print(f"  ✅ Progress: {completed}/{total_tickers} tickers processed ({completed / total_tickers * 100:.1f}%)")
            else:
                print(f"  ⚠️ Skipping ticker {ticker}: {error_message}")
    
    extract_time = time.time()
    print(f"✅ Extracted training data for {len(training_dataframes)} tickers in {extract_time - load_time:.2f} seconds")
    
    # Save training dataframes using multiprocessing
    print(f"💾 Saving training data to {CONFIG[OUTPUT_DIR]} (multiprocessing)...")
    success = write_dataframes_to_parquet(training_dataframes, CONFIG)
    
    save_time = time.time()
    
    if success:
        print(f"✅ Successfully saved training data for {len(training_dataframes)} tickers to {CONFIG[OUTPUT_DIR]} in {save_time - extract_time:.2f} seconds")
        
        # Save additional metadata
        metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'training_info.json')
        training_info = {
            "evaluation_rows_removed": eval_rows,
            "total_tickers": len(training_dataframes),
            "original_tickers": len(ticker_dataframes),
            "skipped_tickers": len(ticker_dataframes) - len(training_dataframes),
            "multiprocessing_used": True,
            "workers_used": CONFIG[MAX_WORKERS],
            "cpu_cores_available": multiprocessing.cpu_count(),
            "processing_time_seconds": {
                "loading": round(load_time - start_time, 2),
                "extraction": round(extract_time - load_time, 2),
                "saving": round(save_time - extract_time, 2),
                "total": round(save_time - start_time, 2)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(training_info, f, indent=2, default=str)
        
        print(f"✅ Saved training information to {metadata_path}")
        print(f"🎉 Total processing time: {save_time - start_time:.2f} seconds")
    else:
        print(f"❌ Error: Failed to save training data")

if __name__ == "__main__":
    main() 