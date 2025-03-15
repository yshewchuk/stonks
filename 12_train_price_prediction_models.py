#!/usr/bin/env python3
"""
Script to train ticker-specific price prediction models.

This script:
1. Loads training and evaluation data from ETL pipeline outputs
2. Creates ModelData objects for each ticker and window
3. Trains price prediction models for each ticker
4. Evaluates model performance and saves results

The script uses the updated PricePredictionTrainingAgent that correctly
aligns historical data features with complete data targets based on dates.
"""

import os
import json
import time
import datetime
import argparse
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from pathlib import Path

from config import CONFIG, TICKERS
from model.model_data import ModelData
from utils.dataframe import read_parquet_files_from_directory
from model.price_prediction_training_agent import PricePredictionTrainingAgent

# Paths configuration
TRAINING_HISTORICAL_PATH = "data/6_training_scale_data"
TRAINING_COMPLETE_PATH = "data/3_training_performance"
EVALUATION_HISTORICAL_PATH = "data/11_evaluation_scaled"
EVALUATION_COMPLETE_PATH = "data/8_evaluation_performance"
OUTPUT_DIR = "models/price_prediction"

# Training configuration
WINDOW_SIZE = 60  # Number of days in each window
EPOCHS = 1000
BATCH_SIZE = 32
EARLY_STOPPING = True
MODEL_PARAMS = {
    'n_steps': WINDOW_SIZE,
    'n_units': 128,
    'learning_rate': 0.001,
    'dropout_rate': 0.2
}

def check_data_directories():
    """
    Check if all required data directories exist.
    
    Returns:
        bool: True if all directories exist, False otherwise
    """
    required_dirs = [
        TRAINING_HISTORICAL_PATH,
        TRAINING_COMPLETE_PATH,
        EVALUATION_HISTORICAL_PATH,
        EVALUATION_COMPLETE_PATH
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Error: Required directory '{directory}' does not exist.")
            return False
            
        # Check if directory has parquet files
        if not any(f.endswith('.parquet') for f in os.listdir(directory)):
            print(f"Warning: Directory '{directory}' does not contain any parquet files.")
            return False
    
    return True

def load_data(historical_path, complete_path, description=""):
    """
    Load historical (scaled) and complete (unscaled) data from specified paths.
    
    Args:
        historical_path (str): Path to directory containing historical (scaled) data
        complete_path (str): Path to directory containing complete (unscaled) data
        description (str): Description for logging (e.g., "training" or "evaluation")
        
    Returns:
        tuple: (historical_data, complete_data) dictionaries of DataFrames
    """
    print(f"\n{'='*80}")
    print(f"Loading {description} data")
    print(f"{'='*80}")
    
    print(f"Loading historical data from {historical_path}...")
    historical_data = read_parquet_files_from_directory(historical_path)
    
    if not historical_data:
        raise ValueError(f"No historical data found in {historical_path}")
    
    print(f"Successfully loaded {len(historical_data)} historical data files:")
    for name, df in historical_data.items():
        date_range = f"{df.index.min().date()} to {df.index.max().date()}" if not df.empty else "empty"
        print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns, date range: {date_range}")
    
    print(f"\nLoading complete data from {complete_path}...")
    complete_data = read_parquet_files_from_directory(complete_path)
    
    if not complete_data:
        raise ValueError(f"No complete data found in {complete_path}")
    
    # Organize the complete data by ticker if possible
    organized_complete_data = {}
    
    for name, df in complete_data.items():
        if df.empty:
            continue
            
        date_range = f"{df.index.min().date()} to {df.index.max().date()}"
        
        # Check if the dataframe has multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            # If it has a multi-level index, the first level might be tickers
            tickers = df.columns.get_level_values(0).unique().tolist()
            print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns, date range: {date_range}")
            print(f"    Multi-level columns with tickers: {tickers}")
            
            # Split the dataframe by ticker
            for ticker in tickers:
                try:
                    ticker_df = df.xs(ticker, level=0, axis=1)
                    organized_complete_data[ticker] = ticker_df
                except:
                    print(f"    Warning: Could not extract ticker {ticker} from {name}")
        else:
            # Single-level columns - try to determine if this is for a specific ticker
            # Check if the filename contains a ticker name
            ticker_match = None
            for ticker in CONFIG['tickers']:
                if ticker in name:
                    ticker_match = ticker
                    break
            
            if ticker_match:
                print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns, date range: {date_range}")
                print(f"    Single-level columns, matched to ticker: {ticker_match}")
                organized_complete_data[ticker_match] = df
            else:
                # Keep the original name as key if we can't determine the ticker
                print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns, date range: {date_range}")
                print(f"    Single-level columns, ticker unknown")
                organized_complete_data[name] = df
    
    # If we couldn't organize by ticker, just return the original data
    if not organized_complete_data:
        print("\nWarning: Could not organize complete data by ticker. Using original data structure.")
        return historical_data, complete_data
    
    print(f"\nSuccessfully organized complete data for {len(organized_complete_data)} items.")
    return historical_data, organized_complete_data

def create_model_data_objects(historical_data, complete_data, tickers, description=""):
    """
    Create ModelData objects from historical and complete data for each ticker and window.
    
    Args:
        historical_data (dict): Dictionary of historical (scaled) DataFrames
        complete_data (dict): Dictionary of complete (unscaled) DataFrames
        tickers (list): List of tickers to create ModelData objects for
        description (str): Description for logging
        
    Returns:
        dict: Dictionary mapping tickers to lists of ModelData objects
    """
    print(f"\n{'='*80}")
    print(f"Creating {description} ModelData objects")
    print(f"{'='*80}")
    
    model_data_dict = {ticker: [] for ticker in tickers}
    summary = {ticker: {"windows": 0, "skipped": 0} for ticker in tickers}
    
    # For each window in historical data
    for window_name, historical_df in historical_data.items():
        # Get start and end dates from the window
        if historical_df.empty:
            print(f"Warning: Empty historical DataFrame for {window_name}. Skipping.")
            continue
            
        start_date = historical_df.index.min()
        end_date = historical_df.index.max()
        
        # For each ticker, find the corresponding complete data
        for ticker in tickers:
            # Try to find the ticker in the complete data
            ticker_complete_data = None
            
            # Check if the complete data is organized by ticker
            if ticker in complete_data:
                # Complete data is a dict with ticker keys
                ticker_df = complete_data[ticker]
                
                # Check if it covers the date range we need
                if not ticker_df.empty and start_date >= ticker_df.index.min() and end_date <= ticker_df.index.max():
                    # Filter to only the date range we need
                    date_mask = (ticker_df.index >= start_date) & (ticker_df.index <= end_date)
                    ticker_complete_data = ticker_df.loc[date_mask]
            
            # If we didn't find ticker-specific data, look for it in the other complete data files
            if ticker_complete_data is None:
                for complete_name, complete_df in complete_data.items():
                    if isinstance(complete_name, str) and ticker in complete_name:
                        # This complete data file might contain the ticker we're looking for
                        if not complete_df.empty and start_date >= complete_df.index.min() and end_date <= complete_df.index.max():
                            # Filter to only the date range we need
                            date_mask = (complete_df.index >= start_date) & (complete_df.index <= end_date)
                            ticker_complete_data = complete_df.loc[date_mask]
                            break
            
            if ticker_complete_data is None or ticker_complete_data.empty:
                print(f"  - Warning: No matching complete data found for ticker {ticker} in window {window_name}. Skipping.")
                summary[ticker]["skipped"] += 1
                continue
            
            # Create ModelData object and add to dictionary
            try:
                # Verify that PPC columns exist in the complete data
                ppc_cols = [col for col in ticker_complete_data.columns if 'PPC' in col]
                if not ppc_cols:
                    print(f"  - Warning: No PPC columns found for ticker {ticker}. Skipping.")
                    summary[ticker]["skipped"] += 1
                    continue
                    
                model_data = ModelData(
                    historical_df, 
                    ticker_complete_data, 
                    [ticker], 
                    start_date=start_date, 
                    end_date=end_date
                )
                model_data_dict[ticker].append(model_data)
                summary[ticker]["windows"] += 1
            except Exception as e:
                print(f"  - Error creating ModelData for {ticker} in {window_name}: {e}")
                summary[ticker]["skipped"] += 1
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{description} ModelData creation summary:")
    for ticker, stats in summary.items():
        print(f"  - {ticker}: Created {stats['windows']} windows, skipped {stats['skipped']} windows")
    print(f"{'='*80}")
    
    return model_data_dict

def train_model_for_ticker(ticker, train_data_list, eval_data_list, output_dir):
    """
    Train a price prediction model for a specific ticker.
    
    Args:
        ticker (str): Ticker symbol
        train_data_list (list): List of ModelData objects for training
        eval_data_list (list): List of ModelData objects for evaluation
        output_dir (str): Directory to save model and results
        
    Returns:
        tuple: (model, history, evaluation_df) - trained model, training history, and evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Training model for ticker: {ticker}")
    print(f"{'='*80}")
    print(f"Training data: {len(train_data_list)} windows")
    print(f"Evaluation data: {len(eval_data_list)} windows")
    
    # Check if we have enough data to train
    if not train_data_list:
        print(f"Error: No training data available for ticker {ticker}")
        return None, None, None
        
    if not eval_data_list:
        print(f"Warning: No evaluation data available for ticker {ticker}. Will train without evaluation.")
        
    feature_count = train_data_list[0].historical_data.shape[1]
    print(f"Feature count: {feature_count}")
    
    # Check for existing model
    ticker_output_dir = os.path.join(output_dir, ticker)
    model_path = os.path.join(ticker_output_dir, "model")
    if os.path.exists(model_path):
        print(f"Warning: Model already exists at {model_path}. Will create a backup and retrain.")
        # Create a backup
        backup_dir = os.path.join(ticker_output_dir, f"backup_{int(time.time())}")
        os.makedirs(backup_dir, exist_ok=True)
        if os.path.exists(os.path.join(ticker_output_dir, "model")):
            os.rename(
                os.path.join(ticker_output_dir, "model"),
                os.path.join(backup_dir, "model")
            )
        # Copy other files
        for f in os.listdir(ticker_output_dir):
            if os.path.isfile(os.path.join(ticker_output_dir, f)):
                os.rename(
                    os.path.join(ticker_output_dir, f),
                    os.path.join(backup_dir, f)
                )
        print(f"Backup created at {backup_dir}")
    
    # Create and train the model
    try:
        agent = PricePredictionTrainingAgent(
            ticker=ticker,
            feature_count=feature_count
        )
        
        # Train the model
        history = agent.train_model(
            train_data_list=train_data_list,
            eval_data_list=eval_data_list if eval_data_list else train_data_list,  # Use train data for validation if no eval data
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Evaluate the model
        evaluation_df = None
        if eval_data_list:
            evaluation_df = agent.evaluate_model(
                eval_data_list=eval_data_list,
                batch_size=BATCH_SIZE
            )
        
        # Save the model and results
        os.makedirs(ticker_output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(ticker_output_dir, "model.keras")
        agent.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save model parameters
        with open(os.path.join(ticker_output_dir, "model_params.json"), "w") as f:
            json.dump(MODEL_PARAMS, f, indent=2)
        
        # Save training history
        if history:
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(ticker_output_dir, "training_history.csv"))
            
            # Plot training history and save
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'])
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'])
                    plt.legend(['Train', 'Validation'], loc='upper right')
                else:
                    plt.legend(['Train'], loc='upper right')
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['mse'])
                if 'val_mse' in history.history:
                    plt.plot(history.history['val_mse'])
                    plt.legend(['Train', 'Validation'], loc='upper right')
                else:
                    plt.legend(['Train'], loc='upper right')
                plt.title('Model MSE')
                plt.ylabel('MSE')
                plt.xlabel('Epoch')
                
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_output_dir, "training_history.png"))
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create training history plot: {e}")
        
        # Save evaluation results
        if evaluation_df is not None and not evaluation_df.empty:
            evaluation_df.to_csv(os.path.join(ticker_output_dir, "evaluation_results.csv"))
            
            # Calculate evaluation metrics
            metrics = {}
            # Group columns by expectation vs prediction
            expected_cols = [col for col in evaluation_df.columns if col.startswith('Expected_')]
            predicted_cols = [col for col in evaluation_df.columns if col.startswith('Predicted_')]
            
            # Calculate MSE for each PPC
            for expected_col, predicted_col in zip(expected_cols, predicted_cols):
                feature_name = expected_col.replace('Expected_', '')
                metrics[f"MSE_{feature_name}"] = ((evaluation_df[expected_col] - evaluation_df[predicted_col])**2).mean()
            
            # Calculate overall MSE
            expected_values = evaluation_df[expected_cols].values
            predicted_values = evaluation_df[predicted_cols].values
            metrics["Overall_MSE"] = ((expected_values - predicted_values)**2).mean()
            
            # Save metrics
            with open(os.path.join(ticker_output_dir, "evaluation_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Evaluation results saved to {ticker_output_dir}")
            print(f"Overall MSE: {metrics['Overall_MSE']:.6f}")
        
        return agent.model, history, evaluation_df
    
    except Exception as e:
        print(f"Error while training model for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main function to run the model training process."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Train ticker-specific price prediction models')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to train models for')
    parser.add_argument('--check-only', action='store_true', help='Only check if data exists, do not train models')
    parser.add_argument('--force', action='store_true', help='Force training even if TensorFlow import fails')
    args = parser.parse_args()
    
    # Check if required data directories exist
    if not check_data_directories():
        print("Error: Required data directories are missing or empty.")
        return
    
    if args.check_only:
        print("Data directories check completed successfully. Use --tickers to specify which tickers to train.")
        return
    
    # Get tickers from args or config
    tickers = args.tickers if args.tickers else CONFIG['tickers']
    print(f"Training models for tickers: {tickers}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Load training data
        train_historical_data, train_complete_data = load_data(
            TRAINING_HISTORICAL_PATH, 
            TRAINING_COMPLETE_PATH,
            description="training"
        )
        
        # Load evaluation data
        eval_historical_data, eval_complete_data = load_data(
            EVALUATION_HISTORICAL_PATH, 
            EVALUATION_COMPLETE_PATH,
            description="evaluation"
        )
        
        # Create ModelData objects for training
        train_model_data_dict = create_model_data_objects(
            train_historical_data, 
            train_complete_data, 
            tickers,
            description="training"
        )
        
        # Create ModelData objects for evaluation
        eval_model_data_dict = create_model_data_objects(
            eval_historical_data, 
            eval_complete_data, 
            tickers,
            description="evaluation"
        )
        
        # Train models for each ticker
        ticker_results = {}
        for ticker in tickers:
            train_data_list = train_model_data_dict.get(ticker, [])
            eval_data_list = eval_model_data_dict.get(ticker, [])
            
            if not train_data_list:
                print(f"Warning: No training data available for ticker {ticker}. Skipping.")
                continue
                
            if not eval_data_list:
                print(f"Warning: No evaluation data available for ticker {ticker}. Will train without evaluation.")
            
            model, history, evaluation_df = train_model_for_ticker(
                ticker, 
                train_data_list, 
                eval_data_list, 
                OUTPUT_DIR
            )
            
            ticker_results[ticker] = {
                "model": model is not None,
                "train_windows": len(train_data_list),
                "eval_windows": len(eval_data_list),
                "training_time_seconds": int(time.time() - start_time)
            }
        
        # Save summary of all models
        summary = {
            "training_time": str(datetime.timedelta(seconds=int(time.time() - start_time))),
            "model_parameters": MODEL_PARAMS,
            "tickers_trained": list(ticker_results.keys()),
            "ticker_details": {ticker: {
                "train_windows": details["train_windows"],
                "eval_windows": details["eval_windows"],
                "trained_successfully": details["model"],
                "training_time_seconds": details["training_time_seconds"]
            } for ticker, details in ticker_results.items()}
        }
        
        with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Training complete! Time elapsed: {datetime.timedelta(seconds=int(time.time() - start_time))}")
        print(f"Models saved to: {OUTPUT_DIR}")
        print(f"{'='*80}")
        
        # Print summary
        print("\nTraining summary:")
        for ticker, details in ticker_results.items():
            status = "SUCCESS" if details["model"] else "FAILED"
            print(f"  - {ticker}: {status}, trained on {details['train_windows']} windows, evaluated on {details['eval_windows']} windows")
    
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 