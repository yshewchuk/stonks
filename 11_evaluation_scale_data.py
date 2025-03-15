#!/usr/bin/env python3
"""
Script to scale evaluation data for model evaluation.

This script:
1. Loads the evaluation time windows from step 10
2. Scales the data using HistoricalDataScaler with the same parameters as training
3. Saves the scaled evaluation data to the output directory

Uses multiprocessing for all CPU-intensive operations to maximize performance.
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
    PRICE_COLUMN_TAGS, VOLUME_PREFIX, RSI_PREFIX, MACD_PREFIX, SCALING_CONFIG
)
from transforms.historical_data_scaler import HistoricalDataScaler
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from utils.process import Process

# Configuration - use the same scaling parameters as training
CONFIG = CONFIG | SCALING_CONFIG | {
    INPUT_DIR: "data/10_evaluation_time_windows",
    OUTPUT_DIR: "data/11_evaluation_scaled",
    "description": "Scaled evaluation data for model evaluation"
}

def _scale_data(args):
    """
    Helper function to scale a single DataFrame.
    Used by ProcessPoolExecutor for parallel scaling.
    
    Args:
        args (tuple): Tuple containing (name, df, price_column_tags, volume_prefix, rsi_prefix, macd_prefix)
        
    Returns:
        tuple: (name, scaled_df, error_message)
    """
    name, df, price_column_tags, volume_prefix, rsi_prefix, macd_prefix = args
    
    try:
        # Create a scaler just for this process
        scaler = HistoricalDataScaler(
            price_column_tags=price_column_tags,
            volume_prefix=volume_prefix,
            rsi_prefix=rsi_prefix,
            macd_prefix=macd_prefix
        )
        
        # Scale the DataFrame
        scaled_df = scaler.scale_dataframe(df)
        
        return (name, scaled_df, None)
    except Exception as e:
        return (name, None, str(e))

def main():
    """Main function to run the evaluation data scaling process."""
    start_time = time.time()
    
    # Initialize the process
    print(f"🚀 Starting evaluation data scaling: {CONFIG[INPUT_DIR]} → {CONFIG[OUTPUT_DIR]}")
    Process.start_process(CONFIG)
    
    # Display processor configuration
    print(f"ℹ️ System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"ℹ️ Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get scaling parameters from config
    price_column_tags = CONFIG[PRICE_COLUMN_TAGS]
    volume_prefix = CONFIG[VOLUME_PREFIX]
    rsi_prefix = CONFIG[RSI_PREFIX]
    macd_prefix = CONFIG[MACD_PREFIX]
    
    print(f"ℹ️ Using scaling configuration:")
    print(f"  - Price column tags: {price_column_tags}")
    print(f"  - Volume prefix: {volume_prefix}")
    print(f"  - RSI prefix: {rsi_prefix}")
    print(f"  - MACD prefix: {macd_prefix}")
    
    # Load evaluation time windows
    print(f"🔍 Loading evaluation time windows from {CONFIG[INPUT_DIR]} (multiprocessing)")
    data_dict = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not data_dict:
        print(f"❌ Error: No evaluation time windows found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    print(f"✅ Loaded {len(data_dict)} time windows in {load_time - start_time:.2f} seconds")
    
    # Scale evaluation data using multiprocessing
    print(f"🔄 Scaling evaluation time windows (multiprocessing)...")
    scaled_data = {}
    total_dfs = len(data_dict)
    
    # Prepare tasks for the process pool
    tasks = [
        (name, df, price_column_tags, volume_prefix, rsi_prefix, macd_prefix) 
        for name, df in data_dict.items()
    ]
    
    # Use ProcessPoolExecutor for parallel scaling
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG[MAX_WORKERS]) as executor:
        futures = {executor.submit(_scale_data, task): task[0] for task in tasks}
        
        # Process results as they complete with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            name, scaled_df, error_message = future.result()
            
            if scaled_df is not None:
                scaled_data[name] = scaled_df
                # Print progress every 5 windows or at the end
                if completed % 5 == 0 or completed == total_dfs:
                    print(f"  ✅ Progress: {completed}/{total_dfs} windows scaled ({completed / total_dfs * 100:.1f}%)")
            else:
                print(f"  ❌ Error scaling window {name}: {error_message}")
    
    scale_time = time.time()
    print(f"✅ Scaled {len(scaled_data)} time windows in {scale_time - load_time:.2f} seconds")
    
    # Save scaled data using multiprocessing
    print(f"💾 Saving scaled evaluation data to {CONFIG[OUTPUT_DIR]} (multiprocessing)...")
    success = write_dataframes_to_parquet(scaled_data, CONFIG)
    
    save_time = time.time()
    
    if success:
        print(f"✅ Successfully saved scaled evaluation data to {CONFIG[OUTPUT_DIR]} in {save_time - scale_time:.2f} seconds")
        
        # Save additional metadata
        metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'scaling_info.json')
        scaling_info = {
            "price_scaling": "relative to first open price",
            "volume_scaling": "min-max scaling within window",
            "rsi_scaling": "fixed range scaling (0-100)",
            "macd_scaling": "min-max scaling within window",
            "total_windows_scaled": len(scaled_data),
            "original_windows": len(data_dict),
            "failed_windows": len(data_dict) - len(scaled_data),
            "multiprocessing_used": True,
            "workers_used": CONFIG[MAX_WORKERS],
            "cpu_cores_available": multiprocessing.cpu_count(),
            "scaling_configuration": {
                "price_column_tags": price_column_tags,
                "volume_prefix": volume_prefix,
                "rsi_prefix": rsi_prefix,
                "macd_prefix": macd_prefix
            },
            "processing_time_seconds": {
                "loading": round(load_time - start_time, 2),
                "scaling": round(scale_time - load_time, 2),
                "saving": round(save_time - scale_time, 2),
                "total": round(save_time - start_time, 2)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(scaling_info, f, indent=2, default=str)
        
        print(f"✅ Saved scaling information to {metadata_path}")
        print(f"🎉 Total processing time: {save_time - start_time:.2f} seconds")
    else:
        print(f"❌ Error: Failed to save scaled evaluation data")

if __name__ == "__main__":
    main() 