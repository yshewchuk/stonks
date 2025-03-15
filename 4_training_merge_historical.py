#!/usr/bin/env python3
"""
Script to merge training data into a single historical dataset.

This script:
1. Loads the processed training data from step 3
2. Merges all ticker data into a single DataFrame with multi-index columns
3. Saves the merged historical data to a new directory

Uses multiprocessing for all operations to maximize performance.
"""

import os
import json
import time
import multiprocessing
from datetime import datetime

import pandas as pd

from config import CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, MERGE_CONFIG, HISTORICAL_COLUMN_PREFIXES
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from utils.process import Process
from transforms.historical_data_merger import HistoricalDataMerger

# Configuration
CONFIG = CONFIG | MERGE_CONFIG | {
    INPUT_DIR: "data/3_training_performance",
    OUTPUT_DIR: "data/4_training_merge_historical",
    "description": "Merged training data with multi-index columns"
}

def main():
    """Main function to run the training data merging process."""
    start_time = time.time()
    
    # Initialize the process
    print(f"🚀 Starting training data merging: {CONFIG[INPUT_DIR]} → {CONFIG[OUTPUT_DIR]}")
    Process.start_process(CONFIG)
    
    # Display processor configuration
    print(f"ℹ️ System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"ℹ️ Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Load processed training data using multiprocessing
    print(f"🔍 Loading processed training data from {CONFIG[INPUT_DIR]} (multiprocessing)")
    ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])
    
    if not ticker_dataframes:
        print(f"❌ Error: No processed training data found in {CONFIG[INPUT_DIR]}")
        return
    
    load_time = time.time()
    print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes in {load_time - start_time:.2f} seconds")
    
    # Initialize the HistoricalDataMerger with configuration
    merger = HistoricalDataMerger(
        historical_column_prefixes=CONFIG[HISTORICAL_COLUMN_PREFIXES]
    )
    
    # Merge ticker DataFrames using the HistoricalDataMerger
    print(f"🔄 Merging historical data from all tickers by date...")
    try:
        merged_df = merger.merge(ticker_dataframes)
        
        merge_time = time.time()
        print(f"✅ Merged ticker DataFrames in {merge_time - load_time:.2f} seconds")
        
        # Save merged DataFrame
        print(f"💾 Saving merged training data to {CONFIG[OUTPUT_DIR]}...")
        
        # Create a dictionary with a single entry for the merged DataFrame
        merged_dict = {"merged_historical": merged_df}
        
        # Save using multiprocessing (though there's only one file, the write function handles it)
        success = write_dataframes_to_parquet(merged_dict, CONFIG)
        
        save_time = time.time()
        
        if success:
            print(f"✅ Successfully saved merged training data to {CONFIG[OUTPUT_DIR]} in {save_time - merge_time:.2f} seconds")
            
            # Save additional metadata
            metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'merge_info.json')
            merge_info = {
                "total_tickers_merged": len(ticker_dataframes),
                "merged_dataframe_shape": {
                    "rows": merged_df.shape[0],
                    "columns": merged_df.shape[1]
                },
                "date_range": {
                    "start": merged_df.index.min().strftime('%Y-%m-%d'),
                    "end": merged_df.index.max().strftime('%Y-%m-%d')
                },
                "multiprocessing_used": True,
                "workers_used": CONFIG[MAX_WORKERS],
                "cpu_cores_available": multiprocessing.cpu_count(),
                "processing_time_seconds": {
                    "loading": round(load_time - start_time, 2),
                    "merging": round(merge_time - load_time, 2),
                    "saving": round(save_time - merge_time, 2),
                    "total": round(save_time - start_time, 2)
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(merge_info, f, indent=2, default=str)
            
            print(f"✅ Saved merge information to {metadata_path}")
            
            # Print summary of the merged dataset
            print("\n📊 Merged Dataset Summary:")
            print(f"Total dates: {len(merged_df)}")
            print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
            print(f"Total tickers: {len(merged_df.columns.levels[0])}")
            print(f"Features per ticker: {len(merged_df.columns.levels[1])}")
            print(f"Total columns: {len(merged_df.columns)}")
            
            # Check for missing values by ticker
            print("\n🧩 Completeness by ticker:")
            for ticker in merged_df.columns.levels[0]:
                ticker_data = merged_df.loc[:, ticker]
                missing_values = ticker_data.isna().any(axis=1).sum()
                missing_percent = (missing_values / len(merged_df)) * 100
                print(f"- {ticker}: {len(merged_df) - missing_values}/{len(merged_df)} complete dates ({missing_percent:.1f}% missing)")
            
            print(f"🎉 Total processing time: {save_time - start_time:.2f} seconds")
        else:
            print(f"❌ Error: Failed to save merged training data")
            
    except Exception as e:
        print(f"❌ Error during merging: {e}")
        return

if __name__ == "__main__":
    main() 