"""
ETL - Process Script: Merge historical data from multiple tickers into a single dataset by date
"""

import pandas as pd
import os
from pathlib import Path
from config import CONFIG, OUTPUT_DIR, INPUT_DIR
from utils.process import Process
from utils.dataframe import read_parquet_files_from_directory, write_dataframes_to_parquet
from transforms.historical_data_merger import HistoricalDataMerger

HISTORICAL_COLUMN_PREFIXES = 'historical_column_prefixes'

# Configure input and output directories
CONFIG = CONFIG | {
    INPUT_DIR: "data/3_training_performance",
    OUTPUT_DIR: "data/4_training_merge_historical",
    # List of prefixes that identify historical columns
    HISTORICAL_COLUMN_PREFIXES: [
        "Open", "High", "Low", "Close", "Volume",  # Original price data
        "MA", "Hi", "Lo", "RSI", "MoACD",  # Technical indicators
    ]
}

# Start the process and write metadata
Process.start_process(CONFIG)

print(f"🔍 Loading processed data from {CONFIG[INPUT_DIR]}")

# Read all parquet files from the input directory
ticker_dataframes = read_parquet_files_from_directory(CONFIG[INPUT_DIR])

if not ticker_dataframes:
    print("❌ No ticker data found in the input directory")
    exit(1)

print(f"✅ Loaded {len(ticker_dataframes)} ticker dataframes")

# Initialize the HistoricalDataMerger with configuration
merger = HistoricalDataMerger(
    historical_column_prefixes=CONFIG[HISTORICAL_COLUMN_PREFIXES]
)

# Merge all historical data by date
print("🔄 Merging historical data from all tickers by date...")
try:
    merged_df = merger.merge(ticker_dataframes)
    
    # Save the merged DataFrame
    print(f"💾 Saving merged dataset to {CONFIG[OUTPUT_DIR]}...")
    success = write_dataframes_to_parquet({"merged_historical": merged_df}, CONFIG)
    
    if success:
        print(f"✅ Saved merged dataset to {CONFIG[OUTPUT_DIR]}")
        
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
        
        # Show sample of the column structure
        sample_columns = []
        for ticker in merged_df.columns.levels[0][:2]:  # First two tickers
            for feature in merged_df.columns.levels[1][:3]:  # First three features
                sample_columns.append((ticker, feature))
                
        print("\n📋 Sample column structure:")
        for column in sample_columns:
            print(f"- {column[0]}, {column[1]}")
            
    else:
        print("❌ Failed to save merged dataset")
        
except Exception as e:
    print(f"❌ Error during merging: {e}")
    exit(1) 