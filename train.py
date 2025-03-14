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
from data_sources.ticker_history import TickerHistory
from model.transaction import Transaction
from utils.dataframe import print_dataframe_debugging_info
from utils.obj import print_public_interface

def load_model_data_from_disk_bulk(base_dir='price_prediction/data_with_lags'):
    """
    Loads ModelData DTO instances from disk from a directory structure.

    Assumes ModelData DTOs were saved in directories under base_dir,
    where each directory name is the end date of the ModelData instance (YYYY-MM-DD).
    Inside each date directory, the files are prefixed with 'data' (e.g., data_historical_data.csv).

    Args:
        base_dir (str): Base directory containing date-named subdirectories.
                          Defaults to 'price_prediction/data_with_lags'.

    Returns:
        list of ModelData: A list of loaded ModelData objects.
                             Returns an empty list if no ModelData objects are found or loading fails.
    """
    loaded_model_data_list = []
    date_directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    if not date_directories:
        print(f"Warning: No date-named subdirectories found in '{base_dir}'. No ModelData to load.")
        return []

    loaded = 0
    for date_dir_name in date_directories:
        filepath_prefix = os.path.join(base_dir, date_dir_name, 'data')
        loaded_dto = ModelData.load_from_disk(filepath_prefix)
        if loaded_dto:
            loaded_model_data_list.append(loaded_dto)
            loaded = loaded + 1
            if loaded % 20 == 0:
                print(f"✅ Loaded ModelData from: {filepath_prefix}, Date: {date_dir_name}")
        else:
            print(f"❌ Failed to load ModelData from: {filepath_prefix}. Skipping directory {date_dir_name}.")

    if not loaded_model_data_list:
        print(f"Warning: No ModelData objects could be loaded from '{base_dir}'.")
    else:
        print(f"✅ Successfully loaded {len(loaded_model_data_list)} ModelData objects from '{base_dir}'.")

    return loaded_model_data_list

# Example Usage (add to the end of model/data_manager.py or in a separate script)
if __name__ == '__main__':
    # ... (Your existing example code from before) ...

    # --- Example: Bulk Load ModelData DTOs from Disk ---
    print("\n--- Example: Bulk Load ModelData DTOs from Disk ---")
    loaded_model_data_list = load_model_data_from_disk_bulk()

    if loaded_model_data_list:
        print(f"First loaded ModelData DTO - Tickers: {loaded_model_data_list[0].tickers}, Start Date: {loaded_model_data_list[0].start_date.date()}, End Date: {loaded_model_data_list[0].end_date.date()}")
        print(f"Number of loaded ModelData DTOs: {len(loaded_model_data_list)}")
        # You can now iterate through loaded_model_data_list and use the ModelData objects
        # for training or simulation.
    else:
        print("❌ No ModelData DTOs loaded from disk.")

    training_windows = list(itertools.islice(loaded_model_data_list, 2530))
    eval_windows = list(itertools.islice(loaded_model_data_list, 60, 530))

    feature_count = len(training_windows[0].historical_data.columns)

    agent = PricePredictionTrainingAgent('AAPL', feature_count) # Using LSTM model

    agent.train_model(training_windows, eval_windows, 1000)

    os.makedirs('saved_model', exist_ok=True)
    agent.model.save('saved_model/model6.keras')

    agent.evaluate_model(eval_windows).to_csv('simulations/predictions.csv')
