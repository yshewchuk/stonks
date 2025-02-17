"""
ETL - Load Script: Loads transformed training data and trains the neural network model.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import random
from datetime import datetime, timedelta, date


# Configuration
TRANSFORMED_DATA_DIR = 'transformed_stock_data'  # Directory where transformed data CSVs are stored
MODEL_NAME = "multi_stock_trader_nn"
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
N_STEPS = 60  # Lookback for time-series features
FEATURE_COLUMNS_STOCK_DATA = ['Open', 'High', 'Low', 'Close', 'Volume', # Stock data features (must match transform script)
                   'MA7', 'MA30', 'MA60',
                   'High7D', 'Low7D', 'High30D', 'Low30D', 'High60D', 'Low60D']
FEATURE_COLUMNS_PORTFOLIO_STATE = ['portfolio_cash_percentage'] # Portfolio state features (placeholder)
FEATURE_COLUMNS = FEATURE_COLUMNS_STOCK_DATA + FEATURE_COLUMNS_PORTFOLIO_STATE # Combined features
N_FEATURES_PER_TICKER = len(FEATURE_COLUMNS) # Calculate total features per ticker
N_FEATURES_TOTAL = N_FEATURES_PER_TICKER * len(TICKERS) # Calculate total features for all tickers
N_UNITS = 128
N_OUTPUTS_PER_TICKER = 2  # Buy percentage, Sell percentage
N_OUTPUT_TOTAL = N_OUTPUTS_PER_TICKER * len(TICKERS) # Total output units
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001


OUTPUT_NAMES = []
for ticker in TICKERS:
    OUTPUT_NAMES.extend([f'{ticker}_buy_percent', f'{ticker}_sell_percent'])



def load_transformed_train_data(transformed_data_dir=TRANSFORMED_DATA_DIR):
    """Loads the transformed training data from CSV."""
    train_file_path = os.path.join(transformed_data_dir, 'train_data.csv')
    if not os.path.exists(train_file_path):
        print(f"❌ Training data file not found: {train_file_path}. Please run transform_data.py first.")
        return None
    print(f"Loading transformed training data from: {train_file_path}...")
    train_combined_df = pd.read_csv(train_file_path, header=[0, 1], index_col=0, parse_dates=True) # MultiIndex header
    return train_combined_df


def prepare_training_data_for_model(train_combined_df, tickers, n_steps, feature_columns_stock_data):
    """
    Prepares training data for the neural network model, including portfolio state features.
    """
    start_time_features = time.time()

    X = []
    dates_list = sorted(list(train_combined_df.index.unique())) # Get sorted unique dates from the training data


    for i in range(n_steps, len(dates_list)):
        timestep_features = []
        current_date_window = dates_list[i - n_steps:i] # Get date window
        print(f'Current date window {current_date_window}')

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


    return np.array(X), None # No labels in this unsupervised/reward-based approach, return X and None for y

def build_model_multi_output(n_steps, n_features_total, n_units, n_output_total):
    """Builds a model with multiple LSTM layers and Dense output for buy/sell percentages."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_steps, n_features_total)),
        tf.keras.layers.LSTM(units=n_units, return_sequences=True),
        tf.keras.layers.LSTM(units=n_units),
        tf.keras.layers.Dense(units=n_units//2, activation='relu'),
        tf.keras.layers.Dense(units=n_output_total, activation='sigmoid') # Sigmoid for percentage outputs
    ])
    return model

def train_model_reward_based_multi_output(model, X_train, all_ticker_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, tickers=TICKERS, n_features_per_ticker=N_FEATURES_PER_TICKER, n_features_total=N_FEATURES_TOTAL, n_outputs_per_ticker=N_OUTPUTS_PER_TICKER):
    """Trains the multi-output LSTM model using a reward-based approach - Corrected Input Shape."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    cumulative_rewards_history = []

    print("\nStarting Reward-Based Training (Multi-Output Model - Corrected Input)...\n")

    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} - Training in Batches across all tickers ---")
        epoch_rewards = []
        portfolio_values_epoch = []
        start_time = time.time()

        for i in range(0, len(X_train), batch_size): # Iterate through training data in batches
            batch_indices = range(i, min(i + batch_size, len(X_train))) # Get batch indices
            current_batch_tickers = tickers  # All tickers for each batch step
            batch_rewards = []
            batch_portfolio_values = []
            total_loss = 0  # Initialize batch loss
            portfolio_value = 100000.0 # Reset portfolio at start of batch (or epoch, adjust as needed)
            positions = {ticker: 0 for ticker in tickers} # Reset positions

            with tf.GradientTape() as tape: # Gradient tape for backprop
                for batch_index_offset, global_index in enumerate(batch_indices): # Iterate within batch
                    portfolio_value_day_start = portfolio_value # Track portfolio start value for daily reward
                    date_for_day = train_combined_df.index[global_index + N_STEPS] # Get date for timestep

                    # --- Corrected: Get *all features* for the current timestep (for all tickers) ---
                    current_features_all_tickers = X_train[global_index:global_index+1] # Shape (1, N_STEPS, N_FEATURES_TOTAL)
                    current_features_all_tickers_np = np.array(current_features_all_tickers).astype(np.float32) # To numpy and float32

                    # --- Debugging - Print shape of current_features_all_tickers before model input ---
                    print(f"Shape of current_features_all_tickers before model input: {current_features_all_tickers_np.shape}") # ADDED DEBUG PRINT

                    output_percentages = model(current_features_all_tickers_np) # Model predicts buy/sell for *all tickers at once*
                    output_percentages_np = output_percentages.numpy()[0] # Get numpy array and remove batch dimension
                    ticker_percentages = output_percentages_np.reshape((len(tickers), n_outputs_per_ticker)) # Reshape to tickers x outputs

                    # --- Execute trades and calculate portfolio change for *all tickers* for the day ---
                    for ticker_index, ticker in enumerate(current_batch_tickers):
                        buy_percent = ticker_percentages[ticker_index, 0]
                        sell_percent = ticker_percentages[ticker_index, 1]

                        price_today = all_ticker_data[ticker].loc[date_for_day, 'Open'] # Open price for simulation
                        cash_to_buy = portfolio_value * buy_percent
                        shares_to_buy = cash_to_buy / price_today if price_today > 0 else 0
                        shares_to_sell = positions[ticker] * sell_percent

                        buy_trade_value = shares_to_buy * price_today
                        if portfolio_value >= buy_trade_value:
                            positions[ticker] += shares_to_buy
                            portfolio_value -= buy_trade_value

                        sell_trade_value = shares_to_sell * price_today
                        if positions[ticker] >= shares_to_sell:
                            positions[ticker] -= shares_to_sell
                            portfolio_value += sell_trade_value

                    # --- Update portfolio value based on end-of-day 'Close' prices for all tickers ---
                    for held_ticker in tickers:
                        if positions[held_ticker] > 0:
                            portfolio_value += positions[held_ticker] * (all_ticker_data[held_ticker].loc[date_for_day, 'Close'] - all_ticker_data[held_ticker].loc[train_combined_df.index[global_index -1 + N_STEPS] if global_index > 0 else train_combined_df.index[N_STEPS] , 'Close'] if global_index > 0 else 0 )


                    # --- Calculate daily reward (portfolio change for the day across all tickers) ---
                    day_reward = portfolio_value - portfolio_value_day_start
                    total_reward += day_reward
                    batch_rewards.append(day_reward) # Reward for the day

                    # --- Simplified Reward-Based Model Update (MSE Loss based on daily reward) ---
                    with tf.GradientTape() as tape_inner: # Inner tape for loss calculation
                        output_probs_tape = model(current_features_all_tickers_np) # Get model output for *all features for the day*
                        target_probs = np.zeros_like(output_probs_tape.numpy()) # Initialize target probs

                        if day_reward > 0:
                            target_probs = np.ones_like(output_probs_tape.numpy()) * 0.9 # Encourage positive reward actions
                        else:
                            target_probs = np.zeros_like(output_probs_tape.numpy()) # Discourage negative reward actions

                        loss = tf.keras.losses.MeanSquaredError()(target_probs, output_probs_tape) # MSE loss

                    gradients = tape_inner.gradient(loss, model.trainable_variables) # Calculate gradients
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Apply gradients


                    batch_portfolio_values.append(portfolio_value) # Portfolio value after all trades for the day


            avg_batch_reward = np.mean(batch_rewards) # Avg batch reward
            avg_final_portfolio_value = np.mean(batch_portfolio_values) # Avg final portfolio value in batch

            epoch_rewards.extend(batch_rewards) # Collect batch rewards for epoch
            portfolio_values_epoch.extend(batch_portfolio_values) # Collect batch portfolio values for epoch

        avg_epoch_reward = np.mean(epoch_rewards) # Avg epoch reward
        avg_final_portfolio_value_epoch = np.mean(portfolio_values_epoch) # Avg final portfolio value for epoch

        rewards_history.append(avg_epoch_reward) # Store epoch average reward

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"  Epoch {epoch+1} completed in {epoch_time:.2f} seconds - Avg Reward: {avg_epoch_reward:.4f}, Avg Final Portfolio Value: ${avg_final_portfolio_value_epoch:.2f}")
        print(f"  Example Portfolio Value Trajectory (First few episodes in epoch): {[portfolio_values_epoch[:min(5, len(portfolio_values_epoch))] ]}...\n")


    print("\nReward-Based Training (Multi-Output Model - Corrected Input) Finished!\n")
    return model, rewards_history

if __name__ == '__main__':
    print("🚀 Starting Multi-Stock Reward-Based Neural Network Model Training (Load Script)...")

    print("Loading transformed training data from CSV...")
    train_combined_df = load_transformed_train_data()
    if train_combined_df is None: # Exit if training data loading failed
        exit()

    print("\nPreparing data for model training...")
    X_train, _ = prepare_training_data_for_model(train_combined_df, TICKERS, N_STEPS, FEATURE_COLUMNS_STOCK_DATA) # Prepare training sequences

    if X_train is None or len(X_train) == 0:
        print("❌ No training data prepared. Check data and feature preparation steps.")
        exit()
    print(f"Prepared training data shape: X_train={X_train.shape}")

    print("\nBuilding Multi-Output Model...")
    model_build_start_time = time.time()
    model = build_model_multi_output(N_STEPS, N_FEATURES_TOTAL, N_UNITS, N_OUTPUT_TOTAL) # Build model
    model_build_end_time = time.time()
    print(f"Model building time: {model_build_end_time - model_build_start_time:.2f} seconds")
    model.summary()

    # Create a dummy all_ticker_data dictionary for train_model_reward_based_multi_output,
    # as it still expects this format for price lookup during trading simulation.
    # In a real-world load script, you might need to re-load the *original* data (or relevant price data)
    # if the reward calculation requires it, instead of just using the transformed training data.
    dummy_all_ticker_data = {} # Create a dummy dictionary.  In a real scenario, you might reload price data here.
    for ticker in TICKERS:
        dummy_all_ticker_data[ticker] = pd.DataFrame(index=train_combined_df.index) # Use the DatetimeIndex directly

    print("\nStarting Reward-Based Training (Multi-Output Model)...")
    training_start_time = time.time()
    trained_model, rewards_history = train_model_reward_based_multi_output(model, X_train, dummy_all_ticker_data) # Train the model
    training_end_time = time.time()
    print(f"Total training time: {training_end_time - training_start_time:.2f} seconds")


    if trained_model:
        print("\n🎉 Reward-Based Multi-Output Training Completed (Load Script)!")
        print(f"Final Cumulative Rewards History: {rewards_history}")
        print(f"Model is trained but not saved in this script.") # Add model saving if needed
        print(f"Next steps: Evaluate and refine the model using evaluation data (eval_data.csv).")
    else:
        print("\n❌ Reward-Based Multi-Output Training Failed or No Models Trained (Load Script).")