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
from model.training_agent import TrainingAgent
import time
import itertools

from config import RAW_DATA_DIR, RAW_DATA_USED_COLUMNS, TRANSFORMED_DATA_DIR, TICKERS, FEATURE_COLUMNS_STOCK_DATA, N_STEPS, PREPARED_DATA_DIR, PREPARED_TRAIN_DATA_X_FILE, PREPARED_TRAIN_DATES_FILE, PREPARED_EVAL_DATA_X_FILE, PREPARED_EVAL_DATES_FILE
from model.data_manager import DataManager
from model.model_data import ModelData
from model.moving_average import MovingAverage
from model.portfolio import Portfolio
from model.price_prediction_training_agent import PricePredictionTrainingAgent
from model.rolling_hi_lo import RollingHiLo
from model.simulation import Simulation
from model.simulation_state import SimulationState
from model.ticker_history import TickerHistory
from model.transaction import Transaction
from utils.dataframe import print_dataframe_debugging_info
from utils.obj import print_public_interface 


def load_and_transform_stock_data(raw_data_dir, transformed_data_dir, tickers, feature_columns, n_steps=N_STEPS):
    """Loads, transforms, and splits stock data into training and evaluation sets."""
    print("\n🔄 Starting Stock Data Transformation (Transform Script)...")

    data_manager = DataManager(TICKERS, RAW_DATA_DIR)
    data_manager.save_to_csv('model_data/stock.csv')
    windows = data_manager.create_price_prediction_windows()

    agent = PricePredictionTrainingAgent(list(itertools.islice(windows, 2500)), 'AAPL') # Using LSTM model
    agent.train_model(20) # Train for 2 epochs, window size 5

    agent.evaluate_model(list(itertools.islice(windows, 30, 230))).to_csv('simulations/predictions.csv')

    os.makedirs('saved_model')
    agent.model.save('saved_model/model.keras')

    return

'''
# Example Usage (using the TrainingAgent class from your code):
if __name__ == '__main__':
    print_public_interface(Simulation)
    print_public_interface(SimulationState)
    print_public_interface(Transaction)
    print_public_interface(Portfolio)
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
# Example Usage (for testing)
if __name__ == '__main__':
    # --- Sample Data and Portfolio Setup (as in previous example) ---
    dates = pd.to_datetime(['2012-08-01', '2012-08-02', '2012-08-03', '2012-08-06', '2012-08-07', '2012-08-08', '2012-08-09', '2012-08-10'] * 10) # Extended dates for more steps
    dates.sort_values() # Ensure dates are sorted
    dates = dates.unique() # Get unique dates if duplicates were introduced
    tickers = ['AAPL', 'MSFT', 'GOOG'] # Added GOOG to increase output size
    columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']], names=['Ticker', 'OHLCV'])
    data_values = np.random.rand(len(dates), len(tickers) * 7)
    sample_data = pd.DataFrame(data_values, index=dates, columns=columns)
    sample_data.index.name = 'Date'

    initial_cash = 100000
    sample_portfolio = Portfolio(initial_cash, tickers)

    # --- Create Multiple Simulations ---
    num_simulations = 2
    simulations_list = []
    for i in range(num_simulations):
        sim_data = sample_data.copy() # Independent copy for each simulation
        sim_portfolio = sample_portfolio.copy() # Independent portfolio for each simulation
        sim = Simulation(sim_data, sim_portfolio)
        simulations_list.append(sim)

    # --- Prepare Agent Parameters Dynamically ---
    sample_simulation = simulations_list[0] # Use the first simulation to dynamically determine parameters
    n_tickers = len(sample_simulation.portfolio.tickers())
    n_features_per_ticker = 0
    for col in sample_simulation.data.columns:
        if col[0] == list(sample_simulation.portfolio.tickers())[0]:
            n_features_per_ticker += 1
    n_ticker_features_total = n_features_per_ticker * n_tickers
    n_portfolio_state_features = 2
    n_features_total = n_ticker_features_total + n_portfolio_state_features
    n_output_total = 2 * n_tickers

    agent_params = {
        'n_steps': 5, # Example window size
        'n_units': 32,
        'n_features_total': n_features_total, # Dynamically calculated total features
        'n_output_total': n_output_total  # Dynamically calculated total outputs
    }

    # --- Initialize and Train Training Agent ---
    agent = TrainingAgent(simulations=simulations_list, model_params=agent_params) # Using LSTM model
    agent.train_model(window_size=5) # Train for 2 epochs, window size 5

    print("\n--- LSTM Training Agent Run Completed ---")
'''