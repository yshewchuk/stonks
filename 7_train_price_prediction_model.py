#!/usr/bin/env python3
"""
Script to train ticker-specific price prediction models.

This script:
1. Loads scaled historical time windows from step 6
2. Loads merged data with PPC columns from step 3
3. Splits the data into training and evaluation sets
4. Trains price prediction models for each ticker
5. Evaluates model performance and saves results

Always uses multiprocessing for all operations to maximize performance.
"""

import os
import time
import json
import shutil
import argparse
import concurrent.futures
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional

from config import (
    CONFIG, OUTPUT_DIR, INPUT_DIR, MAX_WORKERS, TICKERS,
    EVALUATION_ROWS, TRAINING_EVALUATION_CONFIG, DESCRIPTION, STEP_NAME,
    WINDOW_SIZE, TIME_WINDOW_CONFIG
)
from utils.dataframe import read_parquet_files_from_directory
from utils.process import Process
from utils.logger import log_step_start, log_step_complete, log_info, log_success, log_error, log_warning, log_progress, log_section
from model.model_data import ModelData
from model.price_prediction_training_agent import PricePredictionTrainingAgent
from model.model_builder import ModelBuilder
from model.model_storage_manager import ModelStorageManager
from model.model_identifier import ModelIdentifier

# Local constants for this script
HISTORICAL_INPUT_DIR = 'historical_input_dir'
PPC_INPUT_DIR = 'ppc_input_dir'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
EARLY_STOPPING_PATIENCE = 'early_stopping_patience'
REDUCE_LR_PATIENCE = 'reduce_lr_patience'
MODEL_PARAMS = 'model_params'
MODEL_OUTPUT_DIR = 'model_output_dir'
# Get default model parameters from ModelBuilder
default_model_params = ModelBuilder.get_default_model_params()

# Configuration
CONFIG = CONFIG | TRAINING_EVALUATION_CONFIG | TIME_WINDOW_CONFIG | {
    HISTORICAL_INPUT_DIR: "data/6_scaled_data",
    PPC_INPUT_DIR: "data/3_merged_data",
    OUTPUT_DIR: "data/7_models",
    MODEL_OUTPUT_DIR: "data/models",
    DESCRIPTION: "Trained price prediction models",
    STEP_NAME: "Train Price Prediction Models",

    # Training configuration
    EPOCHS: 1000,
    BATCH_SIZE: 32,
    EARLY_STOPPING_PATIENCE: 30,
    REDUCE_LR_PATIENCE: 10,
    # Start with default model parameters and override specific settings
    MODEL_PARAMS: {
        **default_model_params,  # Use all defaults from ModelBuilder
        
        # Override specific parameters
        'n_steps': TIME_WINDOW_CONFIG[WINDOW_SIZE],  # Use window size from config
        'lstm_units': 128,                  # Larger network than default
        'dropout_rate': 0.3,             # Higher dropout for better regularization
        'learning_rate': 0.001,          # Different learning rate
        
        # Basic parameters that will be overridden by --model-type argument
        'cnn_filters': 32,               # No CNN by default, override with --model-type
        'cnn_kernel_size': 3,          # No CNN by default, override with --model-type
        'l2_reg': 0.0,                   # No regularization by default
        'use_batch_norm': False          # No batch normalization by default
    }
}

def create_model_data_objects(historical_windows, ppc_data, tickers, evaluation_rows):
    """
    Create ModelData objects from historical windows and PPC data for each ticker.
    
    Args:
        historical_windows (dict): Dictionary of historical time window DataFrames
        ppc_data (pd.DataFrame): DataFrame with PPC data for all tickers
        tickers (list): List of tickers to create ModelData objects for
        evaluation_rows (int): Number of rows to use for evaluation
        
    Returns:
        dict: Dictionary with training and evaluation ModelData objects
    """
    log_section("Creating ModelData Objects")
    
    # Dictionaries to store results
    train_model_data = {ticker: [] for ticker in tickers}
    eval_model_data = {ticker: [] for ticker in tickers}
    summary = {ticker: {"train_windows": 0, "eval_windows": 0, "skipped": 0} for ticker in tickers}
    
    # Calculate the window size in days
    window_size_days = CONFIG[WINDOW_SIZE]
    
    # Calculate a buffer to ensure no overlap between training and evaluation data
    # We use window_size_days to create a clear separation
    buffer_days = window_size_days
    
    # Calculate the evaluation cutoff date
    # Data after this date will be used for evaluation
    if len(ppc_data) >= evaluation_rows:
        # Get the date that is evaluation_rows days from the end
        evaluation_start_date = ppc_data.index[-evaluation_rows]
        
        # Calculate the training cutoff date by moving back buffer_days from evaluation_start_date
        # This ensures no overlap between training and evaluation windows
        training_cutoff_date = evaluation_start_date - pd.Timedelta(days=buffer_days)
        
        log_info(f"Training data cutoff: {training_cutoff_date.strftime('%Y-%m-%d')}")
        log_info(f"Evaluation data starts: {evaluation_start_date.strftime('%Y-%m-%d')}")
        log_info(f"Buffer between training and evaluation: {buffer_days} days")
    else:
        # If there's not enough data for evaluation, use all data for training
        training_cutoff_date = None
        evaluation_start_date = None
        log_warning(f"Not enough data for evaluation. Using all data for training.")
    
    # Check if we're using tuple column names (old style) or simpler columns (new style)
    using_tuple_columns = isinstance(ppc_data.columns[0], tuple) if len(ppc_data.columns) > 0 else False
    log_info(f"Data format: {'Using tuple column names' if using_tuple_columns else 'Using simple column names'}")
    
    # For each window in historical data
    for window_name, historical_df in historical_windows.items():
        if historical_df.empty:
            log_warning(f"Empty historical DataFrame for {window_name}. Skipping.")
            continue
        
        # Get dates from the window
        start_date = historical_df.index.min()
        end_date = historical_df.index.max()
        
        # Create a date range mask for this window
        date_mask = (ppc_data.index >= start_date) & (ppc_data.index <= end_date)
        window_ppc_data = ppc_data.loc[date_mask]
        
        if window_ppc_data.empty:
            log_warning(f"No matching PPC data found for window {window_name}. Skipping.")
            continue
        
        # Determine if this is a training or evaluation window
        # Windows that end after the training cutoff date will be skipped to maintain a clear separation
        # Windows that contain the evaluation start date or later will be used for evaluation
        if training_cutoff_date is not None and evaluation_start_date is not None:
            # Skip windows that cross the boundary between training and evaluation
            if end_date > training_cutoff_date and start_date < evaluation_start_date:
                log_info(f"Skipping window {window_name} as it crosses the training/evaluation boundary")
                continue
                
            # Determine if this is an evaluation window
            is_eval_window = end_date >= evaluation_start_date
        else:
            # If no clear cutoff (not enough data), use all for training
            is_eval_window = False
        
        # For each ticker, create a ModelData object
        for ticker in tickers:
            try:
                # Check if ticker exists in both datasets
                if ticker not in historical_df.columns.get_level_values(0).unique():
                    summary[ticker]["skipped"] += 1
                    continue
                
                # Check if PPC columns exist for this ticker
                ticker_ppc_cols = [col for col in window_ppc_data.columns 
                                  if col[0] == ticker and 'PPC' in col[1]]
                
                if not ticker_ppc_cols:
                    log_warning(f"No PPC columns found for ticker {ticker} in window {window_name}. Skipping.")
                    summary[ticker]["skipped"] += 1
                    continue
                
                # Create ModelData object with correct parameter names
                model_data = ModelData(
                    historical_data_df=historical_df, 
                    complete_data_df=window_ppc_data,
                    tickers=[ticker], 
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Add to appropriate list based on whether it's training or evaluation data
                if is_eval_window:
                    eval_model_data[ticker].append(model_data)
                    summary[ticker]["eval_windows"] += 1
                else:
                    train_model_data[ticker].append(model_data)
                    summary[ticker]["train_windows"] += 1
                    
            except Exception as e:
                log_error(f"Error creating ModelData for {ticker} in {window_name}: {e}")
                summary[ticker]["skipped"] += 1
    
    # Log summary
    log_info("ModelData creation summary:")
    for ticker, stats in summary.items():
        log_info(f"  - {ticker}: {stats['train_windows']} training windows, {stats['eval_windows']} evaluation windows, {stats['skipped']} skipped")
    
    return {
        'train': train_model_data,
        'eval': eval_model_data,
        'summary': summary
    }

def main():
    """Main function to train price prediction models."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train price prediction models')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to train models for')
    parser.add_argument('--check-only', action='store_true', help='Only check data, do not train models')
    parser.add_argument('--model-type', choices=['simple', 'cnn', 'complex'], default='simple',
                        help='Model type to use (simple=LSTM only, cnn=CNN+LSTM, complex=CNN+LSTM with regularization)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Initialize the process
    log_step_start(CONFIG)
    Process.start_process(CONFIG)
    
    # Display processor configuration
    log_info(f"System has {multiprocessing.cpu_count()} CPU cores available")
    log_info(f"Using up to {CONFIG[MAX_WORKERS]} worker processes")
    
    # Get tickers from args or config
    tickers = args.tickers if args.tickers else CONFIG[TICKERS]
    log_info(f"Selected tickers: {tickers}")
    
    # Adjust model architecture based on model type argument
    if args.model_type == 'simple':
        # Simple LSTM model
        CONFIG[MODEL_PARAMS].update({
            'cnn_layers': 0,                # No CNN layers
            'lstm_layers': 2,                # Two LSTM layers
            'l2_reg': 0.0,                   # No L2 regularization
            'use_batch_norm': False,         # No batch normalization
            'recurrent_dropout_rate': 0.0    # No recurrent dropout
        })
        log_info("Using simple LSTM model architecture")
    elif args.model_type == 'cnn':
        # CNN + LSTM model
        CONFIG[MODEL_PARAMS].update({
            'cnn_layers': 2,
            'lstm_layers': 2,                # Two LSTM layers
            'l2_reg': 0.0,                   # No L2 regularization
            'use_batch_norm': False,         # No batch normalization
            'recurrent_dropout_rate': 0.0    # No recurrent dropout
        })
        log_info("Using CNN+LSTM model architecture")
    elif args.model_type == 'complex':
        # CNN + LSTM with regularization
        CONFIG[MODEL_PARAMS].update({
            'cnn_layers': 2,
            'lstm_layers': 2,                # Two LSTM layers
            'l2_reg': 0.001,                # Add L2 regularization
            'use_batch_norm': True,          # Use batch normalization
            'recurrent_dropout_rate': 0.2,   # Add recurrent dropout
            'dropout_rate': 0.3              # Increase dropout rate
        })
        log_info("Using complex CNN+LSTM model architecture with regularization")
    
    log_info(f"Model parameters: {CONFIG[MODEL_PARAMS]}")
    
    # Load scaled historical windows
    log_section("Loading Data")
    log_info(f"Loading scaled historical windows from {CONFIG[HISTORICAL_INPUT_DIR]}")
    
    historical_windows = read_parquet_files_from_directory(CONFIG[HISTORICAL_INPUT_DIR])
    
    if not historical_windows:
        log_error(f"No historical windows found in {CONFIG[HISTORICAL_INPUT_DIR]}")
        return
    
    historical_load_time = time.time()
    log_success(f"Loaded {len(historical_windows)} historical windows in {historical_load_time - start_time:.2f} seconds")
    
    # Load PPC data from merged data
    log_info(f"Loading merged data with PPC columns from {CONFIG[PPC_INPUT_DIR]}")
    merged_data = read_parquet_files_from_directory(CONFIG[PPC_INPUT_DIR])
    
    if not merged_data:
        log_error(f"No merged data found in {CONFIG[PPC_INPUT_DIR]}")
        return
    
    # Get the merged DataFrame (should be a single file)
    if 'merged_data' in merged_data:
        ppc_data = merged_data['merged_data']
    else:
        # If not found by name, take the first one
        ppc_data = next(iter(merged_data.values()))
    
    merged_load_time = time.time()
    log_success(f"Loaded merged data with shape {ppc_data.shape} in {merged_load_time - historical_load_time:.2f} seconds")
    
    # Get evaluation rows from config
    evaluation_rows = CONFIG[EVALUATION_ROWS]
    log_info(f"Using last {evaluation_rows} rows for evaluation with a window size of {CONFIG[WINDOW_SIZE]} days")
    
    # Create ModelData objects
    model_data_objects = create_model_data_objects(
        historical_windows, 
        ppc_data, 
        tickers, 
        evaluation_rows
    )
    
    # Check-only mode: only check data, do not train models
    if args.check_only:
        log_info("Check-only mode enabled. Will not train models.")
        
        # If specific tickers provided, check existing model info
        for ticker in tickers:
            ticker_output_dir = os.path.join(CONFIG[OUTPUT_DIR], ticker)
            
            # Check if the model directory exists
            if os.path.exists(ticker_output_dir):
                log_info(f"Found model directory for {ticker}: {ticker_output_dir}")
                
                # Check if there's a model_info.json file
                model_info_path = os.path.join(ticker_output_dir, "model_info.json")
                if os.path.exists(model_info_path):
                    try:
                        with open(model_info_path, "r") as f:
                            model_info = json.load(f)
                        
                        log_info(f"Model information for {ticker}:")
                        log_info(f"  - Model directory: {model_info.get('model_dir')}")
                        
                        # Generate model summary
                        model_dir = model_info.get('model_dir')
                        if model_dir and os.path.exists(model_dir):
                            log_info(f"Generating model summary for {ticker}...")
                            
                            try:
                                summary_info = ModelStorageManager.generate_model_summary(model_dir)
                                log_info(f"Summary created successfully:")
                                log_info(f"  - Total runs: {summary_info.get('total_runs')}")
                                log_info(f"  - Valid runs: {summary_info.get('valid_runs')}")
                                log_info(f"  - Best run: {summary_info.get('best_run_number')}")
                                log_info(f"  - Summary path: {summary_info.get('summary_path')}")
                            except Exception as e:
                                log_error(f"Error generating model summary for {ticker}: {e}")
                    except Exception as e:
                        log_error(f"Error reading model info for {ticker}: {e}")
                else:
                    log_warning(f"No model information found for {ticker}")
            else:
                log_warning(f"No model directory found for {ticker}")
                
        log_step_complete(start_time, {
            'check_only': True,
            'tickers_checked': tickers
        })
        return
    
    # Train models for each ticker
    log_section("Training Models")
    
    results = []
    for ticker in tickers:
        train_data_list = model_data_objects['train'].get(ticker, [])
        eval_data_list = model_data_objects['eval'].get(ticker, [])
        
        # Use the new ModelTrainingManager instead of train_model_for_ticker
        from model.model_training_manager import ModelTrainingManager
        
        # Create a training manager for this ticker
        training_manager = ModelTrainingManager(
            ticker=ticker,
            output_dir=CONFIG[MODEL_OUTPUT_DIR],
            epochs=CONFIG[EPOCHS],
            batch_size=CONFIG[BATCH_SIZE],
            early_stopping_patience=CONFIG[EARLY_STOPPING_PATIENCE],
            # Use the logger from this script
            logger=None  # We'll use the logger's functions directly
        )
        
        # Train using model parameters
        ticker_result = training_manager.train_from_parameters(
            model_params=CONFIG[MODEL_PARAMS],
            train_data_list=train_data_list,
            eval_data_list=eval_data_list,
            training_params={
                'batch_size': CONFIG[BATCH_SIZE],
                'reduce_lr_patience': CONFIG[REDUCE_LR_PATIENCE]
            }
        )
        
        results.append(ticker_result)
    
    # Save training summary
    training_summary = {
        "tickers_trained": len(results),
        "successful_trainings": sum(1 for r in results if r.get('success', False)),
        "failed_trainings": sum(1 for r in results if not r.get('success', False)),
        "model_type": args.model_type,
        "ticker_details": {r['ticker']: {
            "success": r['success'],
            "train_windows": r['train_windows'],
            "eval_windows": r['eval_windows'],
            "metrics": r.get('metrics', {}),
            "training_time": r.get('training_time', 0)
        } for r in results},
        "model_parameters": CONFIG[MODEL_PARAMS],
        "training_configuration": {
            "epochs": CONFIG[EPOCHS],
            "batch_size": CONFIG[BATCH_SIZE],
            "early_stopping_patience": CONFIG[EARLY_STOPPING_PATIENCE],
            "evaluation_rows": evaluation_rows,
            "window_size": CONFIG[WINDOW_SIZE]
        }
    }
    
    # Save execution metadata
    end_time = time.time()
    
    time_markers = {
        "historical_load": historical_load_time,
        "merged_load": merged_load_time,
        "end": end_time
    }
    
    Process.save_execution_metadata(
        config=CONFIG,
        filename='training_summary.json',
        metadata=training_summary,
        start_time=start_time,
        time_markers=time_markers
    )
    
    log_step_complete(start_time)
    
    # Group results by status
    successful_results = [result for result in results if result['success']]
    failed_results = [result for result in results if not result['success']]
    
    # Print summary
    log_section("TRAINING SUMMARY")
    log_info(f"Total tickers processed: {len(results)}")
    log_info(f"Successfully trained models: {len(successful_results)}")
    log_info(f"Failed: {len(failed_results)}")
    
    # If there are failures, print them
    if failed_results:
        log_section("FAILED TICKERS")
        for result in failed_results:
            log_error(f"{result['ticker']}: {result['error']}")
    
    # Generate model summaries for each successfully trained ticker
    if successful_results and not args.check_only:
        log_section("GENERATING MODEL SUMMARIES")
        summary_results = []
        
        for result in successful_results:
            ticker = result['ticker']
            model_dir = result['model_dir']
            
            try:
                log_info(f"Generating summary for {ticker}...")
                
                # Create a training manager for this ticker
                training_manager = ModelTrainingManager(
                    ticker=ticker,
                    output_dir=CONFIG[MODEL_OUTPUT_DIR]
                )
                
                # Generate summary using the training manager
                summary = training_manager.generate_summary(model_dir)
                summary_results.append({
                    'ticker': ticker,
                    'best_run': summary.get('best_run_number'),
                    'best_mse': summary.get('best_mse'),
                    'total_runs': summary.get('total_runs'),
                    'summary_path': summary.get('summary_path')
                })
                
                log_success(f"Summary generated for {ticker}: Best MSE={summary.get('best_mse', 'N/A')}")
            except Exception as e:
                log_error(f"Error generating summary for {ticker}: {str(e)}")
        
        # Print overall summary results
        if summary_results:
            log_section("MODEL SUMMARY RESULTS")
            # Create a DataFrame for nice display
            summary_df = pd.DataFrame(summary_results)
            if not summary_df.empty:
                summary_df = summary_df.sort_values('best_mse')
                log_info("\nModel Performance Summary (sorted by MSE):")
                log_info(summary_df.to_string(index=False))
            
    log_section("COMPLETED")

if __name__ == "__main__":
    main() 