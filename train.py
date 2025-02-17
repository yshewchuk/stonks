"""
ETL - Load Script: Loads prepared training data and trains the neural network model.
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
PREPARED_DATA_DIR = 'prepared_stock_data' # Directory for prepared data
PREPARED_TRAIN_DATA_X_FILE = os.path.join(PREPARED_DATA_DIR, 'train_data_X.npy')
PREPARED_TRAIN_DATES_FILE = os.path.join(PREPARED_DATA_DIR, 'train_dates.npy')
PREPARED_EVAL_DATA_X_FILE = os.path.join(PREPARED_DATA_DIR, 'eval_data_X.npy')
PREPARED_EVAL_DATES_FILE = os.path.join(PREPARED_DATA_DIR, 'eval_dates.npy')
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


def load_prepared_train_data():
    """Loads the prepared training data (X and dates) from .npy files."""
    print(f"Loading prepared training data from: {PREPARED_TRAIN_DATA_X_FILE} and {PREPARED_TRAIN_DATES_FILE}...")
    if not os.path.exists(PREPARED_TRAIN_DATA_X_FILE) or not os.path.exists(PREPARED_TRAIN_DATES_FILE):
        print(f"❌ Prepared training data files not found. Please run transform_data.py first.")
        return None, None # Return None for both X and dates if files are missing

    X_train = np.load(PREPARED_TRAIN_DATA_X_FILE)
    train_dates_list = np.load(PREPARED_TRAIN_DATES_FILE, allow_pickle=True) # Load dates list

    return X_train, train_dates_list


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


def calculate_reward_multi_stock_portfolio(output_percentages, current_ticker_data, portfolio_value, positions, ticker):
    """Calculates reward based on portfolio change for a single ticker trade."""
    rewards = [] # Initialize rewards list (can hold rewards at each step if needed)
    buy_percent = output_percentages[0, 0].numpy() # Extract buy percentage for the ticker
    sell_percent = output_percentages[0, 1].numpy() # Extract sell percentage for the ticker

    price_today = current_ticker_data['Open'].iloc[0] # Use 'Open' price for trading simulation

    cash_to_buy = portfolio_value * buy_percent
    shares_to_buy = cash_to_buy / price_today if price_today > 0 else 0
    shares_to_sell = positions[ticker] * sell_percent

    buy_trade_value = shares_to_buy * price_today
    if portfolio_value >= buy_trade_value: # Check if enough cash to buy
        positions[ticker] += shares_to_buy
        portfolio_value -= buy_trade_value

    sell_trade_value = shares_to_sell * price_today
    if positions[ticker] >= shares_to_sell: # Check if enough shares to sell
        positions[ticker] -= shares_to_sell
        portfolio_value += sell_trade_value

    # --- Update portfolio value based on end-of-day 'Close' price for the current ticker ---
    current_close_price = current_ticker_data['Close'].iloc[0]
    portfolio_value += positions[ticker] * current_close_price # Update portfolio value based on current holding and close price


    # --- Calculate reward based on portfolio value change for this ticker's trade ---
    reward = portfolio_value - 100000.0  # Simple reward: final portfolio value (adjust baseline as needed)
    rewards.append(reward) # Store the reward

    return portfolio_value, rewards # Return updated portfolio value and rewards


def train_model_reward_based_multi_output(model, X_train, all_ticker_data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, tickers=TICKERS, n_features_per_ticker=N_FEATURES_PER_TICKER):
    """Trains the multi-output LSTM model using a reward-based approach with batching and ticker-specific feature slicing."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    rewards_history = []

    print("\nStarting Reward-Based Training (Multi-Output Model)...\n")

    for epoch in range(epochs):
        print(f"--- Epoch {epoch+1}/{epochs} - Training in Batches across all tickers ---")
        epoch_rewards = []
        portfolio_values_epoch = []
        start_time = time.time()

        for i in range(0, len(X_train), batch_size): # Iterate through training data in batches
            batch_indices = range(i, min(i + batch_size, len(X_train))) # Get indices for the current batch
            current_batch_tickers = tickers  # Use all tickers for each batch step in multi-output training
            batch_rewards = []
            batch_portfolio_values = []
            total_loss = 0 # Initialize total loss for the batch
            portfolio_value = 100000.0 # Reset portfolio value for each batch - or epoch, adjust as needed
            positions = {ticker: 0 for ticker in tickers} # Reset positions

            with tf.GradientTape() as tape: # Gradient tape for automatic differentiation
                for batch_index_offset, global_index in enumerate(batch_indices): # Iterate within the batch
                    portfolio_value_day_start = portfolio_value # Track portfolio value at start of day
                    date_for_day = train_combined_df.index[global_index + N_STEPS] # Get date for current timestep

                    for ticker_index, ticker in enumerate(current_batch_tickers): # Iterate through tickers *within each timestep*

                        # Prepare features for the *current ticker only* for multi-output model
                        start_feature_index = ticker_index * n_features_per_ticker # Start index of ticker's features
                        end_feature_index = start_feature_index + n_features_per_ticker # End index of ticker's features
                        current_features_ticker = X_train[global_index:global_index+1, :, start_feature_index:end_feature_index] # Shape (1, N_STEPS, N_FEATURES_PER_TICKER)

                        current_features_ticker_np = np.array(current_features_ticker).astype(np.float32) # Ensure float32
                        # --- Debugging - Print shape of current_features_ticker just before model input ---
                        print(f"Shape of current_features_ticker before model input: {current_features_ticker_np.shape}") # ADDED DEBUG PRINT

                        output_percentages = model(current_features_ticker_np) # Model predicts buy/sell for *current ticker*

                        # Get price data for reward calculation for *current ticker*
                        current_ticker_data = all_ticker_data[ticker].loc[train_combined_df.index[global_index + N_STEPS : global_index + N_STEPS + 1]] # Get price data for *current ticker* and *current date*

                        # Calculate reward and portfolio value for *current ticker trade*
                        portfolio_value, rewards = calculate_reward_multi_stock_portfolio(
                            output_percentages, current_ticker_data, portfolio_value, positions, ticker
                        )
                        total_reward = sum(rewards) # Accumulate rewards - adjust as needed (e.g., sum, mean)


                    # --- Simplified Reward-Based Model Update (using MSE Loss - applied after all tickers for the day) ---
                    # After processing all tickers for the day, calculate loss and gradients.
                    # This approach simplifies the reward and loss association to the portfolio change for the *day*.
                    with tf.GradientTape() as tape_inner: # Inner tape for loss calculation
                        output_probs_tape = model(X_train[global_index:global_index+1]) # Get model output for *all features for the day*
                        target_probs = np.zeros_like(output_probs_tape.numpy()) # Initialize target probabilities

                        day_reward = portfolio_value - portfolio_value_day_start # Reward for the *entire day's trading*

                        if day_reward > 0:
                            target_probs = np.ones_like(output_probs_tape.numpy()) * 0.9 # Encourage actions for positive daily reward
                        else:
                            target_probs = np.zeros_like(output_probs_tape.numpy()) # Discourage for negative daily reward

                        loss = tf.keras.losses.MeanSquaredError()(target_probs, output_probs_tape) # MSE loss


                    gradients = tape_inner.gradient(loss, model.trainable_variables) # Gradients for this day's loss
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Apply gradients


                    batch_rewards.append(total_reward) # Reward for the entire day's trading across all tickers
                    batch_portfolio_values.append(portfolio_value) # Portfolio value after all trades for the day



            avg_batch_reward = np.mean(batch_rewards) # Average reward over the batch
            avg_final_portfolio_value = np.mean(batch_portfolio_values) # Average portfolio value over batch

            epoch_rewards.extend(batch_rewards) # Collect rewards for epoch tracking
            portfolio_values_epoch.extend(batch_portfolio_values)

        avg_epoch_reward = np.mean(epoch_rewards) # Average reward for the epoch
        avg_final_portfolio_value_epoch = np.mean(portfolio_values_epoch) # Average final portfolio value for epoch

        rewards_history.append(avg_epoch_reward) # Store epoch reward in history

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"  Epoch {epoch+1} completed in {epoch_time:.2f} seconds - Avg Reward: {avg_epoch_reward:.4f}, Avg Final Portfolio Value: ${avg_final_portfolio_value_epoch:.2f}")
        print(f"  Example Portfolio Value Trajectory (First few episodes in epoch): {[portfolio_values_epoch[:min(5, len(portfolio_values_epoch))] ]}...\n")


    print("\nReward-Based Training (Multi-Output Model) Finished!\n")
    return model, rewards_history



if __name__ == '__main__':
    print("🚀 Starting Multi-Stock Reward-Based Neural Network Model Training (Load Script)...")

    print("Loading prepared training data...")
    X_train, _ = load_prepared_train_data()
    if X_train is None: # Exit if prepared training data loading failed
        exit()

    if X_train is None or len(X_train) == 0:
        print("❌ No prepared training data loaded. Check data loading steps.")
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