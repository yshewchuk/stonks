#!/usr/bin/env python3
"""
Script to determine optimal model configurations using genetic algorithms.

This script:
1. Loads scaled historical time windows from step 6
2. Loads merged data with PPC columns from step 3
3. Splits the data into training and evaluation sets
4. Uses genetic algorithms to find optimal hyperparameters
5. Saves the best model configuration for each ticker

Uses concurrent processing for evaluating multiple model configurations in parallel.
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
from model.genetic import GeneticOptimizer, Specimen, SpecimenStorageManager
from model.model_identifier import ModelIdentifier

# Local constants for this script
HISTORICAL_INPUT_DIR = 'historical_input_dir'
PPC_INPUT_DIR = 'ppc_input_dir'
EPOCHS = 'epochs'
EARLY_STOPPING_PATIENCE = 'early_stopping_patience'
POPULATION_SIZE = 'population_size'
GENERATIONS = 'generations'
MUTATION_RATE = 'mutation_rate'
CROSSOVER_RATE = 'crossover_rate'
ELITISM_COUNT = 'elitism_count'
TOURNAMENT_SIZE = 'tournament_size'
GENETIC_OUTPUT_DIR = 'genetic_output_dir'
PARALLEL_PROCESSES = 'parallel_processes'

# Configuration
CONFIG = CONFIG | TRAINING_EVALUATION_CONFIG | TIME_WINDOW_CONFIG | {
    HISTORICAL_INPUT_DIR: "data/6_scaled_data",
    PPC_INPUT_DIR: "data/3_merged_data",
    OUTPUT_DIR: "data/7_optimal_models",
    GENETIC_OUTPUT_DIR: "data/genetic_optimization",
    DESCRIPTION: "Optimal model configurations",
    STEP_NAME: "Determine Optimal Model Configurations",

    PARALLEL_PROCESSES: 2,

    # Training configuration
    EPOCHS: 1000,  # Fewer epochs for faster genetic evolution
    EARLY_STOPPING_PATIENCE: 30,
    
    # Genetic algorithm configuration
    POPULATION_SIZE: 100,
    GENERATIONS: 100,
    MUTATION_RATE: 0.1,
    CROSSOVER_RATE: 0.7,
    ELITISM_COUNT: 5,
    TOURNAMENT_SIZE: 9
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
                if using_tuple_columns:
                    # Old style with tuple column names
                    ticker_exists = ticker in historical_df.columns.get_level_values(0).unique()
                    
                    # Check if PPC columns exist for this ticker
                    ticker_ppc_cols = [col for col in window_ppc_data.columns 
                                      if col[0] == ticker and 'PPC' in col[1]]
                    
                    if not ticker_exists or not ticker_ppc_cols:
                        summary[ticker]["skipped"] += 1
                        continue
                else:
                    # New style with simple columns - all columns are feature columns
                    if historical_df.empty or window_ppc_data.empty:
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

def optimize_model_for_ticker(ticker, train_data_list, eval_data_list, config):
    """
    Find optimal model configuration for a specific ticker using genetic algorithms.
    
    Args:
        ticker (str): Ticker symbol
        train_data_list (list): List of ModelData objects for training
        eval_data_list (list): List of ModelData objects for evaluation
        config (dict): Configuration dictionary
        
    Returns:
        dict: Results including the best model configuration
    """
    log_section(f"Optimizing Model for {ticker}")
    log_info(f"Training data: {len(train_data_list)} windows")
    log_info(f"Evaluation data: {len(eval_data_list)} windows")
    
    # Create output directory for this ticker
    ticker_output_dir = os.path.join(config[OUTPUT_DIR], ticker)
    os.makedirs(ticker_output_dir, exist_ok=True)
    
    # Create genetic output directory
    genetic_output_dir = os.path.join(config[GENETIC_OUTPUT_DIR], ticker)
    os.makedirs(genetic_output_dir, exist_ok=True)
    
    # Check if we have enough data to train
    if not train_data_list:
        log_error(f"No training data available for ticker {ticker}")
        return {
            'success': False,
            'ticker': ticker,
            'error': 'No training data available',
            'train_windows': 0,
            'eval_windows': len(eval_data_list) if eval_data_list else 0
        }
    
    if not eval_data_list:
        log_warning(f"No evaluation data available for ticker {ticker}. Will train without evaluation.")
    
    try:
        # Create experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{ticker}_evolution_{timestamp}"
        
        # Initialize the genetic optimizer
        optimizer = GeneticOptimizer(
            ticker=ticker,
            train_data=train_data_list,
            eval_data=eval_data_list if eval_data_list else [],
            output_dir=genetic_output_dir,
            experiment_name=experiment_name,
            population_size=config[POPULATION_SIZE],
            generations=config[GENERATIONS],
            mutation_rate=config[MUTATION_RATE],
            crossover_rate=config[CROSSOVER_RATE],
            elitism_count=config[ELITISM_COUNT],
            tournament_size=config[TOURNAMENT_SIZE],
            epochs=config[EPOCHS],
            early_stopping_patience=config[EARLY_STOPPING_PATIENCE],
            max_workers=CONFIG[PARALLEL_PROCESSES]
        )
        
        # Run the optimization process
        log_info(f"Starting evolution process for {ticker}")
        optimization_start = time.time()
        best_specimen = optimizer.evolve()
        optimization_time = time.time() - optimization_start
        
        if best_specimen and best_specimen.is_evaluated:
            log_success(f"Optimization completed for {ticker}")
            log_info(f"Best MSE: {best_specimen.best_mse:.6f}")
            log_info(f"Best model ID: {best_specimen.model_id}")
            
            # Create a configuration file with the best parameters
            best_config = {
                "ticker": ticker,
                "specimen_id": best_specimen.specimen_id,
                "model_id": best_specimen.model_id,
                "model_parameters": best_specimen.model_parameters,
                "training_parameters": best_specimen.training_parameters,
                "feature_indexes": list(best_specimen.feature_indexes),
                "best_mse": best_specimen.best_mse,
                "optimization_time": optimization_time,
                "training_time": best_specimen.training_time,
                "generations": config[GENERATIONS],
                "population_size": config[POPULATION_SIZE],
                "created_at": datetime.now().isoformat()
            }
            
            # Save the best configuration
            best_config_path = os.path.join(ticker_output_dir, "optimal_configuration.json")
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            log_info(f"Saved optimal configuration to {best_config_path}")
            
            # If there's a best model, create a symlink or copy to the ticker output directory
            best_model_path = best_specimen.training_results.get('best_model_path')
            if best_model_path and os.path.exists(best_model_path):
                try:
                    ticker_model_path = os.path.join(ticker_output_dir, "optimal_model.keras")
                    if os.path.exists(ticker_model_path):
                        os.remove(ticker_model_path)
                    try:
                        os.symlink(best_model_path, ticker_model_path)
                        log_info(f"Created symlink to best model at {ticker_model_path}")
                    except:
                        # Fall back to copy if symlink fails
                        shutil.copy2(best_model_path, ticker_model_path)
                        log_info(f"Copied best model to {ticker_model_path}")
                except Exception as e:
                    log_warning(f"Could not create model link in ticker directory: {e}")
            
            # Save a summary of the optimization process
            summary = optimizer.storage_manager.create_experiment_summary()
            summary_path = os.path.join(ticker_output_dir, "optimization_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            log_info(f"Saved optimization summary to {summary_path}")
            
            # Return results
            return {
                'success': True,
                'ticker': ticker,
                'optimization_time': optimization_time,
                'best_specimen_id': best_specimen.specimen_id,
                'best_model_id': best_specimen.model_id,
                'best_mse': best_specimen.best_mse,
                'train_windows': len(train_data_list),
                'eval_windows': len(eval_data_list) if eval_data_list else 0
            }
        else:
            log_error(f"Optimization failed for {ticker}: No valid best specimen found")
            return {
                'success': False,
                'ticker': ticker,
                'error': 'No valid best specimen found',
                'train_windows': len(train_data_list),
                'eval_windows': len(eval_data_list) if eval_data_list else 0
            }
    except Exception as e:
        log_error(f"Error optimizing model for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e),
            'train_windows': len(train_data_list),
            'eval_windows': len(eval_data_list) if eval_data_list else 0
        }

def main():
    """Main function to determine optimal model configurations."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Determine optimal model configurations')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to optimize')
    parser.add_argument('--generations', type=int, help='Number of generations for evolution')
    parser.add_argument('--population', type=int, help='Population size per generation')
    parser.add_argument('--parallel', type=int, help='Number of tickers to optimize in parallel')
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
    
    # Update configuration if specified in args
    if args.generations:
        CONFIG[GENERATIONS] = args.generations
    if args.population:
        CONFIG[POPULATION_SIZE] = args.population
    
    # Determine parallelism
    parallel_tickers = args.parallel if args.parallel else min(len(tickers), CONFIG[MAX_WORKERS])
    
    log_info(f"Genetic algorithm configuration:")
    log_info(f"  Population size: {CONFIG[POPULATION_SIZE]}")
    log_info(f"  Generations: {CONFIG[GENERATIONS]}")
    log_info(f"  Mutation rate: {CONFIG[MUTATION_RATE]}")
    log_info(f"  Crossover rate: {CONFIG[CROSSOVER_RATE]}")
    log_info(f"  Elitism count: {CONFIG[ELITISM_COUNT]}")
    log_info(f"  Tournament size: {CONFIG[TOURNAMENT_SIZE]}")
    log_info(f"  Optimizing {len(tickers)} tickers with {parallel_tickers} in parallel")
    
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
    
    # Optimize models for each ticker (in parallel if specified)
    log_section("Optimizing Models")
    
    results = []
    if parallel_tickers > 1:
        # Parallel optimization across tickers
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_tickers) as executor:
            futures = []
            
            for ticker in tickers:
                train_data_list = model_data_objects['train'].get(ticker, [])
                eval_data_list = model_data_objects['eval'].get(ticker, [])
                
                future = executor.submit(
                    optimize_model_for_ticker,
                    ticker,
                    train_data_list,
                    eval_data_list,
                    CONFIG
                )
                futures.append(future)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        log_success(f"Completed optimization for {result['ticker']} with MSE={result['best_mse']:.6f}")
                    else:
                        log_error(f"Failed optimization for {result['ticker']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    log_error(f"Error in optimization task: {e}")
    else:
        # Sequential optimization
        for ticker in tickers:
            train_data_list = model_data_objects['train'].get(ticker, [])
            eval_data_list = model_data_objects['eval'].get(ticker, [])
            
            result = optimize_model_for_ticker(
                ticker,
                train_data_list,
                eval_data_list,
                CONFIG
            )
            
            results.append(result)
    
    # Save optimization summary
    optimization_summary = {
        "tickers_optimized": len(results),
        "successful_optimizations": sum(1 for r in results if r.get('success', False)),
        "failed_optimizations": sum(1 for r in results if not r.get('success', False)),
        "genetic_parameters": {
            "population_size": CONFIG[POPULATION_SIZE],
            "generations": CONFIG[GENERATIONS],
            "mutation_rate": CONFIG[MUTATION_RATE],
            "crossover_rate": CONFIG[CROSSOVER_RATE],
            "elitism_count": CONFIG[ELITISM_COUNT],
            "tournament_size": CONFIG[TOURNAMENT_SIZE]
        },
        "ticker_details": {r['ticker']: {
            "success": r['success'],
            "best_mse": r.get('best_mse', None),
            "train_windows": r['train_windows'],
            "eval_windows": r['eval_windows'],
            "optimization_time": r.get('optimization_time', 0)
        } for r in results},
        "training_configuration": {
            "epochs": CONFIG[EPOCHS],
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
        filename='optimization_summary.json',
        metadata=optimization_summary,
        start_time=start_time,
        time_markers=time_markers
    )
    
    log_step_complete(start_time)
    
    # Group results by status
    successful_results = [result for result in results if result['success']]
    failed_results = [result for result in results if not result['success']]
    
    # Print summary
    log_section("OPTIMIZATION SUMMARY")
    log_info(f"Total tickers processed: {len(results)}")
    log_info(f"Successfully optimized: {len(successful_results)}")
    log_info(f"Failed: {len(failed_results)}")
    
    # If there are failures, print them
    if failed_results:
        log_section("FAILED TICKERS")
        for result in failed_results:
            log_error(f"{result['ticker']}: {result['error']}")
    
    # Print overall results
    if successful_results:
        log_section("OPTIMIZATION RESULTS")
        # Create a DataFrame for nice display
        summary_df = pd.DataFrame([
            {
                'Ticker': r['ticker'],
                'Best MSE': r['best_mse'],
                'Specimen ID': r['best_specimen_id'],
                'Optimization Time (s)': r['optimization_time']
            }
            for r in successful_results
        ])
        
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Best MSE')
            log_info("\nModel Optimization Results (sorted by MSE):")
            log_info(summary_df.to_string(index=False))
            
    log_section("COMPLETED")

if __name__ == "__main__":
    main() 