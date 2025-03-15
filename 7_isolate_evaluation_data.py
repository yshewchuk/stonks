#!/usr/bin/env python3
"""
Script to isolate evaluation data from the original dataset.

This script:
1. Loads the original stock data from step 1
2. Extracts the last 400 rows from each ticker's data
3. Saves the evaluation data to a new directory

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
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet, extract_data_range
from utils.process import Process

# Configuration
CONFIG = CONFIG | TRAINING_EVALUATION_CONFIG |{
    INPUT_DIR: "data/1_raw_data",
    OUTPUT_DIR: "data/7_isolate_evaluation_data",
    "description": "Isolated evaluation data (last 400 rows from original dataset)"
}

def _extract_evaluation_data(args):
    """
    Helper function to extract evaluation data from a single ticker's DataFrame.
    Used by ProcessPoolExecutor for parallel processing.
    
    Args:
        args (tuple): Tuple containing (ticker, df, eval_rows)
        
    Returns:
        tuple: (ticker, eval_df, error_message)
    """
    ticker, df, eval_rows = args
    
    try:
        # Use the extract_data_range function with extract_recent=True
        eval_df = extract_data_range(df, num_rows=eval_rows, extract_recent=True)
        
        if eval_df is None:
            return (ticker, None, f"DataFrame has insufficient rows for extraction")
        
        return (ticker, eval_df, None)
    except Exception as e:
        return (ticker, None, str(e))

def main():
    """Main function to run the evaluation data isolation process."""
    start_time = time.time()
    
    # Initialize the process
    print(f"🚀 Starting evaluation data isolation: {CONFIG[INPUT_DIR]} → {CONFIG[OUTPUT_DIR]}")
    Process.start_process(CONFIG)
    
    # Display processor configuration
    print(f"ℹ️ System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"ℹ️ Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get evaluation rows from config
    eval_rows = CONFIG[EVALUATION_ROWS]
    print(f"ℹ️ Will extract the last {eval_rows} rows from each ticker's data")
    
    # Load original stock data using multiprocessing
    print(f"🔍 Loading original stock data from {CONFIG[INPUT_DIR]} (multiprocessing)")
    ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not ticker_dataframes:
        print(f"❌ Error: No stock data found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes in {load_time - start_time:.2f} seconds")
    
    # Extract evaluation data using multiprocessing
    print(f"🔄 Extracting last {eval_rows} rows from each ticker (multiprocessing)...")
    evaluation_dataframes = {}
    total_tickers = len(ticker_dataframes)
    
    # Prepare tasks for the process pool
    tasks = [
        (ticker, df, eval_rows) 
        for ticker, df in ticker_dataframes.items()
    ]
    
    # Use ProcessPoolExecutor for parallel extraction
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG[MAX_WORKERS]) as executor:
        futures = {executor.submit(_extract_evaluation_data, task): task[0] for task in tasks}
        
        # Process results as they complete with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            ticker, eval_df, error_message = future.result()
            
            if eval_df is not None:
                evaluation_dataframes[ticker] = eval_df
                # Print progress every 10 tickers or at the end
                if completed % 10 == 0 or completed == total_tickers:
                    print(f"  ✅ Progress: {completed}/{total_tickers} tickers processed ({completed / total_tickers * 100:.1f}%)")
            else:
                print(f"  ⚠️ Skipping ticker {ticker}: {error_message}")
    
    extract_time = time.time()
    print(f"✅ Extracted evaluation data for {len(evaluation_dataframes)} tickers in {extract_time - load_time:.2f} seconds")
    
    # Save evaluation dataframes using multiprocessing
    print(f"💾 Saving evaluation data to {CONFIG[OUTPUT_DIR]} (multiprocessing)...")
    success = write_dataframes_to_parquet(evaluation_dataframes, CONFIG)
    
    save_time = time.time()
    
    if success:
        print(f"✅ Successfully saved evaluation data for {len(evaluation_dataframes)} tickers to {CONFIG[OUTPUT_DIR]} in {save_time - extract_time:.2f} seconds")
        
        # Save additional metadata
        metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'evaluation_info.json')
        evaluation_info = {
            "evaluation_rows_per_ticker": eval_rows,
            "total_tickers": len(evaluation_dataframes),
            "original_tickers": len(ticker_dataframes),
            "skipped_tickers": len(ticker_dataframes) - len(evaluation_dataframes),
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
            json.dump(evaluation_info, f, indent=2, default=str)
        
        print(f"✅ Saved evaluation information to {metadata_path}")
        print(f"🎉 Total processing time: {save_time - start_time:.2f} seconds")
    else:
        print(f"❌ Error: Failed to save evaluation data")

if __name__ == "__main__":
    main() 