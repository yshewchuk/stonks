#!/usr/bin/env python3
"""
Script to calculate performance metrics on training data.

This script:
1. Loads the isolated training data from step 2
2. Calculates technical indicators and performance metrics
3. Saves the processed training data to a new directory

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
    RAW_DATA_USED_COLUMNS, WINDOWS, LAG_PERIODS, PREDICTIONS_CONFIG, PERFORMANCE_CONFIG
)
from transforms.raw_stock_preprocessor import RawStockPreProcessor
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from utils.process import Process

# Configuration
CONFIG = CONFIG | PERFORMANCE_CONFIG | {
    INPUT_DIR: "data/2_isolate_training_data",
    OUTPUT_DIR: "data/3_training_performance",
    "description": "Training data with calculated performance metrics"
}

def _process_ticker_data(args):
    """
    Helper function to process a single ticker's DataFrame.
    Used by ProcessPoolExecutor for parallel processing.
    
    Args:
        args (tuple): Tuple containing (ticker, df, required_columns, windows, lag_periods, ppc_configs)
        
    Returns:
        tuple: (ticker, processed_df, error_message)
    """
    ticker, df, required_columns, windows, lag_periods, ppc_configs = args
    
    try:
        # Create a preprocessor for this process
        preprocessor = RawStockPreProcessor(
            required_columns=required_columns,
            windows=windows,
            lag_periods=lag_periods,
            ppc_configs=ppc_configs
        )
        
        # Process the DataFrame
        processed_df = preprocessor.process(df)
        
        # Check if processing was successful
        if processed_df is None or processed_df.empty:
            return (ticker, None, "Processing resulted in empty DataFrame")
        
        return (ticker, processed_df, None)
    except Exception as e:
        return (ticker, None, str(e))

def main():
    """Main function to run the training performance calculation process."""
    start_time = time.time()
    
    # Initialize the process
    print(f"🚀 Starting training performance calculation: {CONFIG[INPUT_DIR]} → {CONFIG[OUTPUT_DIR]}")
    Process.start_process(CONFIG)
    
    # Display processor configuration
    print(f"ℹ️ System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"ℹ️ Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get performance parameters from config
    raw_data_used_columns = CONFIG[RAW_DATA_USED_COLUMNS]
    windows = CONFIG[WINDOWS]
    lag_periods = CONFIG[LAG_PERIODS]
    predictions_config = CONFIG[PREDICTIONS_CONFIG]
    
    print(f"ℹ️ Using performance metrics configuration:")
    print(f"  - Raw data used columns: {raw_data_used_columns}")
    print(f"  - Windows: {windows}")
    print(f"  - Lag periods: {lag_periods}")
    print(f"  - Predictions config: {len(predictions_config)} configurations")
    
    # Load training data using multiprocessing
    print(f"🔍 Loading training data from {CONFIG[INPUT_DIR]} (multiprocessing)")
    ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not ticker_dataframes:
        print(f"❌ Error: No training data found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes in {load_time - start_time:.2f} seconds")
    
    # Process training data using multiprocessing
    print(f"🔄 Processing training data for {len(ticker_dataframes)} tickers (multiprocessing)...")
    processed_dataframes = {}
    total_tickers = len(ticker_dataframes)
    
    # Prepare tasks for the process pool
    tasks = [
        (ticker, df, raw_data_used_columns, windows, lag_periods, predictions_config) 
        for ticker, df in ticker_dataframes.items()
    ]
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG[MAX_WORKERS]) as executor:
        futures = {executor.submit(_process_ticker_data, task): task[0] for task in tasks}
        
        # Process results as they complete with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            ticker, processed_df, error_message = future.result()
            
            if processed_df is not None:
                processed_dataframes[ticker] = processed_df
                # Print progress every 10 tickers or at the end
                if completed % 10 == 0 or completed == total_tickers:
                    print(f"  ✅ Progress: {completed}/{total_tickers} tickers processed ({completed / total_tickers * 100:.1f}%)")
            else:
                print(f"  ❌ Error processing ticker {ticker}: {error_message}")
    
    process_time = time.time()
    print(f"✅ Processed training data for {len(processed_dataframes)} tickers in {process_time - load_time:.2f} seconds")
    
    # Save processed dataframes using multiprocessing
    print(f"💾 Saving processed training data to {CONFIG[OUTPUT_DIR]} (multiprocessing)...")
    success = write_dataframes_to_parquet(processed_dataframes, CONFIG)
    
    save_time = time.time()
    
    if success:
        print(f"✅ Successfully saved processed training data for {len(processed_dataframes)} tickers to {CONFIG[OUTPUT_DIR]} in {save_time - process_time:.2f} seconds")
        
        # Save additional metadata
        metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'processing_info.json')
        processing_info = {
            "total_tickers_processed": len(processed_dataframes),
            "original_tickers": len(ticker_dataframes),
            "failed_tickers": len(ticker_dataframes) - len(processed_dataframes),
            "technical_indicators": {
                "raw_data_used_columns": raw_data_used_columns,
                "windows": windows,
                "lag_periods": lag_periods,
                "predictions_config": predictions_config
            },
            "multiprocessing_used": True,
            "workers_used": CONFIG[MAX_WORKERS],
            "cpu_cores_available": multiprocessing.cpu_count(),
            "processing_time_seconds": {
                "loading": round(load_time - start_time, 2),
                "processing": round(process_time - load_time, 2),
                "saving": round(save_time - process_time, 2),
                "total": round(save_time - start_time, 2)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(processing_info, f, indent=2, default=str)
        
        print(f"✅ Saved processing information to {metadata_path}")
        
        # Summarize the features generated for the first ticker (as an example)
        first_ticker = next(iter(processed_dataframes))
        first_df = processed_dataframes[first_ticker]
        print(f"\n📊 Example of features generated for {first_ticker}:")
        print(f"Total features: {len(first_df.columns)}")
        print(f"Total rows: {len(first_df)}")
        
        # Show sample column names by category
        column_categories = {
            "Original": [col for col in first_df.columns if col in raw_data_used_columns],
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
        
        print(f"🎉 Total processing time: {save_time - start_time:.2f} seconds")
    else:
        print(f"❌ Error: Failed to save processed training data")

if __name__ == "__main__":
    main() 