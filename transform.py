"""
ETL - Transform Script: Loads raw stock data, transforms it, prepares training data sequences,
and saves both transformed CSVs and prepared data as .npy files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import time

from config import RAW_DATA_DIR, RAW_DATA_USED_COLUMNS, TRANSFORMED_DATA_DIR, TICKERS, FEATURE_COLUMNS_STOCK_DATA, N_STEPS, PREPARED_DATA_DIR, PREPARED_TRAIN_DATA_X_FILE, PREPARED_TRAIN_DATES_FILE, PREPARED_EVAL_DATA_X_FILE, PREPARED_EVAL_DATES_FILE
from model.model_data import ModelData
from model.moving_average import MovingAverage
from model.portfolio import Portfolio
from model.rolling_hi_lo import RollingHiLo
from model.simulation import Simulation
from model.ticker_history import TickerHistory
from utils.dataframe import print_dataframe_debugging_info


def transform_stock_data(raw_stock_data, tickers=TICKERS, feature_columns=FEATURE_COLUMNS_STOCK_DATA, n_steps=N_STEPS):
    for ticker in tickers: 
        append_technical_indicators(raw_stock_data[ticker])
        scale_features(raw_stock_data[ticker])

    return raw_stock_data

def split_train_eval_data(transformed_data, eval_ratio=0.2):
    """Splits transformed data into training and evaluation sets based on date."""
    train_data = {}
    eval_data = {}
    for ticker, df in transformed_data.items():
        split_index = int(len(df) * (1 - eval_ratio)) # Index to split data
        train_data[ticker] = df.iloc[:split_index]
        eval_data[ticker] = df.iloc[split_index:]
    return train_data, eval_data

def prepare_training_data_for_model(train_combined_df, tickers, n_steps, feature_columns_stock_data):
    """
    Prepares training data for the neural network model, including portfolio state features.
    Moved from train.py to transform_data.py
    """
    start_time_features = time.time()

    X = []
    dates_list = sorted(list(train_combined_df.index.unique())) # Get sorted unique dates from the training data


    for i in range(n_steps, len(dates_list)):
        timestep_features = []
        current_date_window = dates_list[i - n_steps:i] # Get date window

        # Initialize portfolio state for each timestep - in a real setup, portfolio state would persist across timesteps
        portfolio_state = {  # Simplified portfolio state, could be extended
            'portfolio_value': 100000.0, # Initial portfolio value for each sequence
            'initial_portfolio_value': 100000.0,
            'positions': {ticker: 0 for ticker in TICKERS} # No initial positions at the start of each sequence
        }

        portfolio_features_for_timestep = []
        portfolio_cash_percentage = portfolio_state['portfolio_value'] / portfolio_state['initial_portfolio_value'] if portfolio_state['initial_portfolio_value'] > 0 else 0.0
        portfolio_features_for_timestep.append(portfolio_cash_percentage)  # Cash percentage
        for ticker in tickers: # Position percentages (initially zero)
            position_percentage = 0.0 # Initially no positions at the start of each training sequence
            portfolio_features_for_timestep.append(position_percentage)


        for date_in_window in current_date_window:
            date_features = []
            for ticker in tickers:
                # Access transformed data using MultiIndex: (ticker, column_name)
                ticker_date_features_df = train_combined_df.loc[date_in_window, ticker] # Get DataFrame slice for date and ticker
                ticker_date_features = ticker_date_features_df[feature_columns_stock_data].values # Select feature columns and get numpy array
                date_features.extend(ticker_date_features) # Extend date_features with stock features for this ticker

            date_features.extend(portfolio_features_for_timestep) # Append portfolio state features *to each date's features*
            timestep_features.append(date_features) # Append features for this date (all tickers + portfolio state)


        X.append(np.array(timestep_features)) # Append timestep features to X


    return np.array(X), dates_list[n_steps:] # Return dates list corresponding to the *start* of each sequence


def save_transformed_and_prepared_data(train_data, eval_data, transformed_data_dir=TRANSFORMED_DATA_DIR, prepared_data_dir=PREPARED_DATA_DIR):
    """
    Saves transformed training/eval datasets to CSVs with proper MultiIndex header,
    and also saves prepared training/eval data (X, dates) as .npy files.
    """
    print(f"Saving transformed data to: {transformed_data_dir}...")
    print(f"Saving prepared data to: {prepared_data_dir}...")

    # --- Save transformed data to CSVs ---
    train_combined_df = pd.concat(train_data, axis=1, keys=train_data.keys())
    train_file_path = os.path.join(transformed_data_dir, 'train_data.csv')
    train_combined_df.to_csv(train_file_path, header=True, index_label=['Date'])
    print(f"  Transformed training data saved to: {train_file_path}")

    eval_combined_df = pd.concat(eval_data, axis=1, keys=eval_data.keys())
    eval_file_path = os.path.join(transformed_data_dir, 'eval_data.csv')
    eval_combined_df.to_csv(eval_file_path, header=True, index_label=['Date'])
    print(f"  Transformed evaluation data saved to: {eval_file_path}")


    # --- Prepare training data using prepare_training_data_for_model ---
    print("\nPreparing training data for model...")
    X_train, train_dates_list = prepare_training_data_for_model(train_combined_df, TICKERS, N_STEPS, FEATURE_COLUMNS_STOCK_DATA)
    if X_train is not None:
        print(f"  Prepared training data shape: X_train={X_train.shape}")
    else:
        print("❌ No training data prepared.")

    # --- Save prepared training data as .npy files ---
    if not os.path.exists(prepared_data_dir):
        os.makedirs(prepared_data_dir) # Create prepared data directory if it doesn't exist

    if X_train is not None:
        np.save(PREPARED_TRAIN_DATA_X_FILE, X_train)
        np.save(PREPARED_TRAIN_DATES_FILE, train_dates_list) # Save dates list as well, if needed
        print(f"  Prepared training data (X) saved to: {PREPARED_TRAIN_DATA_X_FILE}")
        print(f"  Prepared training dates saved to: {PREPARED_TRAIN_DATES_FILE}")
    else:
        print("❌ Prepared training data not saved due to preparation failure.")


    # --- Prepare evaluation data using prepare_training_data_for_model ---
    print("\nPreparing evaluation data for model...")
    X_eval, eval_dates_list = prepare_training_data_for_model(eval_combined_df, TICKERS, N_STEPS, FEATURE_COLUMNS_STOCK_DATA)
    if X_eval is not None:
        print(f"  Prepared evaluation data shape: X_eval={X_eval.shape}")
    else:
        print("❌ No evaluation data prepared.")


    # --- Save prepared evaluation data as .npy files ---
    if X_eval is not None:
        np.save(PREPARED_EVAL_DATA_X_FILE, X_eval)
        np.save(PREPARED_EVAL_DATES_FILE, eval_dates_list) # Save dates list
        print(f"  Prepared evaluation data (X) saved to: {PREPARED_EVAL_DATA_X_FILE}")
        print(f"  Prepared evaluation dates saved to: {PREPARED_EVAL_DATES_FILE}")
    else:
        print("❌ Prepared evaluation data not saved due to preparation failure.")


    print("\n🎉 Stock Data Transformation and Preparation Complete!")
    print(f"Transformed data saved to: {transformed_data_dir}")
    print(f"Prepared data saved to: {prepared_data_dir}")



def load_and_transform_stock_data(raw_data_dir, transformed_data_dir, tickers, feature_columns, n_steps=N_STEPS):
    """Loads, transforms, and splits stock data into training and evaluation sets."""
    print("\n🔄 Starting Stock Data Transformation (Transform Script)...")


    model = ModelData(TICKERS, RAW_DATA_DIR)
    model.save_to_csv('model_data/stock.csv')
    simulations = model.create_simulations(365, 10000)
    
    os.makedirs('simulations', exist_ok=True)
    count = 0
    for sim in simulations:
        count += 1
        sim.start(60).to_csv(f'simulations/{count}.csv')
        for i in range(1, 100):
            sim.step(60, [{ 'ticker': 'AAPL', 'quantity': 1, 'order_type': 'buy'}]).to_csv(f'simulations/{count}_{i}.csv')

    return

'''
if __name__ == '__main__':
    print("\n🎉 Starting Stock Data Transformation and Preparation (Transform Script)...")

    train_data, eval_data = load_and_transform_stock_data(RAW_DATA_DIR, TRANSFORMED_DATA_DIR, tickers=TICKERS, feature_columns=FEATURE_COLUMNS_STOCK_DATA, n_steps=N_STEPS) # Load and transform
    if train_data and eval_data:
        save_transformed_and_prepared_data(train_data, eval_data) # Call the new combined saving function
    else:
        print("❌ Transformation process failed. Prepared data not saved.")

    print("Transformed and prepared data saved to directories specified in config.py") # Update message
    print("✅ ETL - Transform and Prepare script completed successfully!") # Final success message
'''

# Example Usage (for testing and demonstration)
if __name__ == '__main__':
    # Define tickers and starting cash
    tickers = ['AAPL', 'GOOG', 'MSFT']
    starting_cash = 100000

    # Create a Portfolio instance
    portfolio = Portfolio(starting_cash, tickers)

    # Simulate buying stocks
    date1 = pd.to_datetime('2024-01-02')
    portfolio.buy(date1, 'AAPL', price=170, quantity=10)
    date2 = pd.to_datetime('2024-01-03')
    portfolio.buy(date2, 'GOOG', price=2700, quantity=2)

    # Simulate selling stocks
    date3 = pd.to_datetime('2024-01-05')
    portfolio.sell(date3, 'AAPL', price=180, quantity=5)


    # Get transactions for date3
    transactions_date3 = portfolio.get_transactions_on_date(date3)
    print(f"\n--- Transactions on {date3.date()} ---")
    for txn in transactions_date3:
        print(txn)

    print(f"\n--- Full Transaction Log ---")
    for txn in portfolio.transaction_log:
        print(txn) # Transaction objects will be printed using their __repr__ method

    # Calculate portfolio value with example stock prices
    current_stock_prices = {'AAPL': 185, 'GOOG': 3000,  'MSFT': 310}
    date4 = pd.to_datetime('2024-01-07')
    portfolio.update_valuations(date4, current_stock_prices)
    print(f"\n--- Portfolio Value on {date4.date()} ---")
    print(f"Portfolio Value: ${portfolio.latest_value():.2f}")

    # Test chronological order enforcement - should raise ValueError
    try:
        portfolio.buy(pd.to_datetime('2024-01-04'), 'MSFT', price=300, quantity=10) # Date before last transaction (2024-01-07 from update_valuations, but conceptually 2024-01-05 from sell)
    except ValueError as e:
        print(f"\n--- Chronological Order Test (Expected Error) ---")
        print(f"Error caught: {e}")
    else:
        print("\n--- Chronological Order Test (Error NOT Caught - PROBLEM!) ---")
        print("Expected ValueError was not raised when adding out-of-order transaction.")