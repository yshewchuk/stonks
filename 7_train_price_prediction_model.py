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
import argparse
import concurrent.futures
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

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

# Local constants for this script
HISTORICAL_INPUT_DIR = 'historical_input_dir'
PPC_INPUT_DIR = 'ppc_input_dir'
EPOCHS = 'epochs'
BATCH_SIZE = 'batch_size'
EARLY_STOPPING_PATIENCE = 'early_stopping_patience'
MODEL_PARAMS = 'model_params'

# Get default model parameters from ModelBuilder
default_model_params = ModelBuilder.get_default_model_params()

# Configuration
CONFIG = CONFIG | TRAINING_EVALUATION_CONFIG | TIME_WINDOW_CONFIG | {
    HISTORICAL_INPUT_DIR: "data/6_scaled_data",
    PPC_INPUT_DIR: "data/3_merged_data",
    OUTPUT_DIR: "data/7_models",
    DESCRIPTION: "Trained price prediction models",
    STEP_NAME: "Train Price Prediction Models",
    # Training configuration
    EPOCHS: 1000,
    BATCH_SIZE: 32,
    EARLY_STOPPING_PATIENCE: 30,
    # Start with default model parameters and override specific settings
    MODEL_PARAMS: {
        **default_model_params,  # Use all defaults from ModelBuilder
        
        # Override specific parameters
        'n_steps': TIME_WINDOW_CONFIG[WINDOW_SIZE],  # Use window size from config
        'n_units': 128,                  # Larger network than default
        'dropout_rate': 0.3,             # Higher dropout for better regularization
        'learning_rate': 0.001,          # Different learning rate
        
        # Basic parameters that will be overridden by --model-type argument
        'cnn_filters': [],               # No CNN by default, override with --model-type
        'cnn_kernel_sizes': [],          # No CNN by default, override with --model-type
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

def train_model_for_ticker(ticker, train_data_list, eval_data_list, output_dir):
    """
    Train a price prediction model for a specific ticker.
    
    Args:
        ticker (str): Ticker symbol
        train_data_list (list): List of ModelData objects for training
        eval_data_list (list): List of ModelData objects for evaluation
        output_dir (str): Directory to save model and results
        
    Returns:
        dict: Results of training including model metrics
    """
    log_section(f"Training Model for {ticker}")
    log_info(f"Training data: {len(train_data_list)} windows")
    log_info(f"Evaluation data: {len(eval_data_list)} windows")
    
    # Create output directory for this ticker
    ticker_output_dir = os.path.join(output_dir, ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    
    # Check if we have enough data to train
    if not train_data_list:
        log_error(f"No training data available for ticker {ticker}")
        return {
            'success': False,
            'ticker': ticker,
            'error': 'No training data available',
            'train_windows': 0,
            'eval_windows': len(eval_data_list)
        }
    
    if not eval_data_list:
        log_warning(f"No evaluation data available for ticker {ticker}. Will train without evaluation.")
    
    try:
        # Get feature count from the first window's historical data
        feature_count = train_data_list[0].historical_data.shape[1]
        log_info(f"Feature count: {feature_count}")
        
        # Update model_params with feature count
        model_params = CONFIG[MODEL_PARAMS].copy()
        model_params['n_features_total'] = feature_count
        
        # Build the model using ModelBuilder
        log_info(f"Building model with parameters: {model_params}")
        model = ModelBuilder.build_price_prediction_model(model_params)
                
        # Initialize the model to ensure all flags are properly set
        # This creates input tensors and builds the model graph
        log_info("Initializing model to ensure all internal flags are set")
        dummy_input = np.zeros((1, model_params['n_steps'], model_params['n_features_total']))
        _ = model(dummy_input)
        
        log_info(f"Model compilation status: {'Compiled' if hasattr(model, '_is_compiled') and model._is_compiled else 'Not compiled'}")
        log_info(f"Model has {len(model.trainable_weights)} trainable weights")
        
        # Create the training agent with the pre-built model
        agent = PricePredictionTrainingAgent(
            ticker=ticker,
            model=model
        )
        
        # Train the model
        training_start = time.time()
        history = agent.train_model(
            train_data_list=train_data_list,
            eval_data_list=eval_data_list if eval_data_list else None,
            epochs=CONFIG[EPOCHS],
            batch_size=CONFIG[BATCH_SIZE],
            early_stopping_patience=CONFIG[EARLY_STOPPING_PATIENCE]
        )
        training_time = time.time() - training_start
        
        # Evaluate the model
        evaluation_df = None
        metrics = {}
        
        if eval_data_list:
            evaluation_start = time.time()
            evaluation_df = agent.evaluate_model(
                eval_data_list=eval_data_list,
                batch_size=CONFIG[BATCH_SIZE]
            )
            evaluation_time = time.time() - evaluation_start
            
            # Calculate evaluation metrics if we have results
            if evaluation_df is not None and not evaluation_df.empty:
                # Group columns by expectation vs prediction
                expected_cols = [col for col in evaluation_df.columns if col.startswith('Expected_')]
                predicted_cols = [col for col in evaluation_df.columns if col.startswith('Predicted_')]
                
                # Calculate MSE for each PPC
                for expected_col, predicted_col in zip(expected_cols, predicted_cols):
                    feature_name = expected_col.replace('Expected_', '')
                    metrics[f"MSE_{feature_name}"] = float(((evaluation_df[expected_col] - evaluation_df[predicted_col])**2).mean())
                
                # Calculate overall MSE
                expected_values = evaluation_df[expected_cols].values
                predicted_values = evaluation_df[predicted_cols].values
                metrics["Overall_MSE"] = float(((expected_values - predicted_values)**2).mean())
        else:
            evaluation_time = 0
        
        # Save model
        save_start = time.time()
        model_path = os.path.join(ticker_output_dir, "model.keras")
        agent.model.save(model_path)
        
        # Save model parameters
        with open(os.path.join(ticker_output_dir, "model_params.json"), "w") as f:
            json.dump(model_params, f, indent=2)
        
        # Save training history
        if history:
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(ticker_output_dir, "training_history.csv"))
        
        # Save evaluation results
        if evaluation_df is not None and not evaluation_df.empty:
            evaluation_df.to_csv(os.path.join(ticker_output_dir, "evaluation_results.csv"))
            
            # Save metrics
            with open(os.path.join(ticker_output_dir, "evaluation_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
        save_time = time.time() - save_start
        
        log_success(f"Successfully trained and saved model for {ticker}")
        
        if metrics:
            log_info(f"Overall MSE: {metrics.get('Overall_MSE', 'N/A')}")
        
        # Return results
        return {
            'success': True,
            'ticker': ticker,
            'training_time': training_time,
            'evaluation_time': evaluation_time,
            'save_time': save_time,
            'metrics': metrics,
            'train_windows': len(train_data_list),
            'eval_windows': len(eval_data_list),
            'feature_count': feature_count
        }
        
    except Exception as e:
        log_error(f"Error training model for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e),
            'train_windows': len(train_data_list),
            'eval_windows': len(eval_data_list)
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
            'cnn_filters': [],                # No CNN layers
            'cnn_kernel_sizes': [],          # No CNN layers
            'lstm_layers': 2,                # Two LSTM layers
            'l2_reg': 0.0,                   # No L2 regularization
            'use_batch_norm': False,         # No batch normalization
            'recurrent_dropout_rate': 0.0    # No recurrent dropout
        })
        log_info("Using simple LSTM model architecture")
    elif args.model_type == 'cnn':
        # CNN + LSTM model
        CONFIG[MODEL_PARAMS].update({
            'cnn_filters': [64, 128],        # Two CNN layers
            'cnn_kernel_sizes': [3, 3],      # With 3x3 kernels
            'lstm_layers': 2,                # Two LSTM layers
            'l2_reg': 0.0,                   # No L2 regularization
            'use_batch_norm': False,         # No batch normalization
            'recurrent_dropout_rate': 0.0    # No recurrent dropout
        })
        log_info("Using CNN+LSTM model architecture")
    elif args.model_type == 'complex':
        # CNN + LSTM with regularization
        CONFIG[MODEL_PARAMS].update({
            'cnn_filters': [64, 128],        # Two CNN layers
            'cnn_kernel_sizes': [3, 3],      # With 3x3 kernels
            'lstm_layers': 2,                # Two LSTM layers
            'l2_reg': 0.0001,                # Add L2 regularization
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
    
    # Check if we're only checking data
    if args.check_only:
        log_info("Data check completed. Use --tickers to specify which tickers to train.")
        return
    
    # Train models for each ticker
    log_section("Training Models")
    
    results = []
    for ticker in tickers:
        train_data_list = model_data_objects['train'].get(ticker, [])
        eval_data_list = model_data_objects['eval'].get(ticker, [])
        
        ticker_result = train_model_for_ticker(
            ticker, 
            train_data_list, 
            eval_data_list, 
            CONFIG[OUTPUT_DIR]
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
    
    # Print final summary
    log_section("Training Summary")
    for result in results:
        ticker = result.get('ticker', 'Unknown')
        status = "SUCCESS" if result.get('success', False) else "FAILED"
        mse = result.get('metrics', {}).get('Overall_MSE', 'N/A')
        train_windows = result.get('train_windows', 0)
        eval_windows = result.get('eval_windows', 0)
        log_info(f"  - {ticker}: {status}, MSE: {mse}, Trained on {train_windows} windows, Evaluated on {eval_windows} windows")

if __name__ == "__main__":
    main() 