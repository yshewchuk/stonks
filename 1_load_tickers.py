#!/usr/bin/env python3
"""
Script to retrieve and save raw stock data.

This script:
1. Downloads historical stock data for configured tickers
2. Sanitizes and validates the downloaded data
3. Saves the raw data to the output directory

Always uses multiprocessing for all operations to maximize performance.
"""

import os
import json
import time
import concurrent.futures
import multiprocessing
from datetime import datetime, timedelta

import pandas as pd

from config import (CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, TICKERS, 
                   DESCRIPTION, STEP_NAME)
from data_sources.ticker_history import TickerHistory
from utils.dataframe import write_dataframes_to_parquet
from utils.process import Process
from utils.logger import log_step_start, log_step_complete, log_info, log_success, log_error, log_warning

START_DATE = 'start_date'
END_DATE = 'end_date'

# Configuration
CONFIG = CONFIG | {
    START_DATE: datetime.today() - timedelta(days=int(365 * 20)),
    END_DATE: datetime.today(),
    OUTPUT_DIR: "data/1_raw_data",
    DESCRIPTION: "Raw stock data downloaded from Yahoo Finance",
    STEP_NAME: "Load Tickers"
}

def _download_ticker_data(args):
    """
    Helper function to download data for a single ticker.
    Used by ProcessPoolExecutor for parallel downloading.
    
    Args:
        args (tuple): Tuple containing (ticker, start_date, end_date)
        
    Returns:
        tuple: (ticker, dataframe, error_message)
    """
    ticker, start_date, end_date = args
    
    try:
        # Download data from Yahoo Finance
        df = TickerHistory.load_dataframe(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            return (ticker, None, "Downloaded data is empty")
            
        return (ticker, df, None)
    except Exception as e:
        return (ticker, None, str(e))

def main():
    """Main function to run the stock data download process."""
    start_time = time.time()
    
    # Initialize the process
    log_step_start(CONFIG)
    Process.start_process(CONFIG)
    
    # Display processor configuration
    log_info(f"System has {multiprocessing.cpu_count()} CPU cores available")
    log_info(f"Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get configuration parameters
    start_date = CONFIG[START_DATE]
    end_date = CONFIG[END_DATE]
    tickers = CONFIG[TICKERS]
    
    log_info(f"Downloading data for {len(tickers)} tickers")
    log_info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Download ticker data using multiprocessing
    ticker_dataframes = {}
    total_tickers = len(tickers)
    
    # Prepare tasks for the process pool
    tasks = [
        (ticker, start_date, end_date) 
        for ticker in tickers
    ]
    
    # Always use parallel processing
    log_info(f"Using parallel processing for {total_tickers} tickers")
    # Use ProcessPoolExecutor for parallel downloading
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG[MAX_WORKERS]) as executor:
        futures = {executor.submit(_download_ticker_data, task): task[0] for task in tasks}
        
        # Process results as they complete with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            ticker, df, error_message = future.result()
            
            if df is not None:
                ticker_dataframes[ticker] = df
                log_success(f"Downloaded data for {ticker}: {len(df)} rows ({completed}/{total_tickers})")
            else:
                log_error(f"Failed to download data for {ticker}: {error_message} ({completed}/{total_tickers})")
    
    download_time = time.time()
    log_success(f"Downloaded data for {len(ticker_dataframes)} tickers in {download_time - start_time:.2f} seconds")
    
    if not ticker_dataframes:
        log_error(f"Failed to download data for any tickers")
        return
    
    # Save ticker dataframes
    log_info(f"Saving raw stock data to {CONFIG[OUTPUT_DIR]}...")
    success = write_dataframes_to_parquet(ticker_dataframes, CONFIG)
    
    save_time = time.time()
    
    if success:
        log_success(f"Successfully saved raw data for {len(ticker_dataframes)} tickers to {CONFIG[OUTPUT_DIR]} in {save_time - download_time:.2f} seconds")
        
        # Save additional metadata
        metadata_path = os.path.join(CONFIG[OUTPUT_DIR], 'download_info.json')
        download_info = {
            "date_range": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d')
            },
            "total_tickers_downloaded": len(ticker_dataframes),
            "requested_tickers": tickers,
            "failed_tickers": [ticker for ticker in tickers if ticker not in ticker_dataframes],
            "rows_per_ticker": {ticker: len(df) for ticker, df in ticker_dataframes.items()},
            "multiprocessing_used": True,
            "workers_used": CONFIG[MAX_WORKERS],
            "cpu_cores_available": multiprocessing.cpu_count(),
            "processing_time_seconds": {
                "downloading": round(download_time - start_time, 2),
                "saving": round(save_time - download_time, 2),
                "total": round(save_time - start_time, 2)
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(download_info, f, indent=2, default=str)
        
        log_success(f"Saved download information to {metadata_path}")
        log_step_complete(start_time)
    else:
        log_error(f"Failed to save raw stock data")

if __name__ == "__main__":
    main()