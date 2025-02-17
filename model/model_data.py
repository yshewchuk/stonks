import os

import numpy as np
import pandas as pd

from model.moving_average import MovingAverage 
from model.portfolio import Portfolio 
from model.rolling_hi_lo import RollingHiLo 
from model.simulation import Simulation 
from model.ticker_history import TickerHistory 

RAW_DATA_USED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']  # Columns used from raw dataset

class ModelData:
    """
    Represents processed stock market data ready for model training and simulation.

    This class handles loading historical stock data for multiple tickers,
    calculating technical indicators (Moving Averages, Rolling Hi-Lo),
    aggregating the data into a single DataFrame, and creating independent
    simulation datasets with associated Portfolio objects for backtesting or training.
    """

    def __init__(self, tickers, raw_data_dir):
        """
        Initializes ModelData object.

        Loads historical data for given tickers, calculates performance indicators,
        and aggregates the data.

        Args:
            tickers (list of str): List of stock ticker symbols (e.g., ['AAPL', 'GOOG']).
            raw_data_dir (str): Path to the directory containing raw data CSV files.
                                 CSV files should be named as '{ticker}.csv'.
        """
        if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
            raise ValueError("Tickers must be a list of strings.")
        if not os.path.isdir(raw_data_dir):
            raise ValueError(f"Raw data directory '{raw_data_dir}' is not a valid directory.")

        self.tickers = list(tickers) # Store tickers internally as a list
        self.raw_data_dir = raw_data_dir
        self.stock_data = {}  # Dictionary to hold processed DataFrame for each ticker
        self.data = None # Combined DataFrame holding data for all tickers
        self.date_min = None # Earliest date in the combined dataset
        self.date_max = None # Latest date in the combined dataset

        self._load_and_process_data() # Call internal method to load and process data during initialization

    def _load_and_process_data(self):
        """
        Internal method to load raw data for each ticker and process it.

        Loads data using TickerHistory, calculates performance indicators,
        and stores the processed DataFrame in self.stock_data.
        Finally, it concatenates all ticker data into self.data and sets date range.
        """
        processed_stock_data = {} # Temporary dict to hold processed dataframes

        for ticker in self.tickers:
            print(f'Reading data file for ticker: {ticker}')
            file_path = os.path.join(self.raw_data_dir, f'{ticker}.csv')
            history = TickerHistory(ticker) # Assuming TickerHistory class exists and is properly initialized
            if not history.load_from_csv(file_path):
                print(f"Warning: Could not load data from CSV for ticker {ticker} at {file_path}. Skipping ticker.")
                continue # Skip to the next ticker if loading fails

            processed_df = self._standardize_and_calculate_performance(history.dataframe())
            if processed_df is not None:
                processed_stock_data[ticker] = processed_df
            else:
                print(f"Warning: Processing failed for ticker {ticker}. Skipping ticker.")

        if not processed_stock_data:
            print("Error: No valid stock data loaded for any ticker. ModelData initialization failed.")
            self.data = None # Indicate data loading failure
            return

        self.stock_data = processed_stock_data # Assign processed data to instance variable
        self.data = pd.concat(self.stock_data, axis='columns', join='inner', keys=self.stock_data.keys(), names=['Ticker']).dropna() # Use keys and names for clarity

        self.date_min = self.data.index.min()
        self.date_max = self.data.index.max()
        print(f"✅ Model data aggregated for tickers: {self.tickers}. Date range: {self.date_min.date()} to {self.date_max.date()}")


    def _standardize_and_calculate_performance(self, df):
        """
        Internal method to standardize data and calculate performance indicators for a single ticker DataFrame.

        Currently, it drops unnecessary columns and extends the DataFrame with
        Moving Averages and Rolling Hi-Lo range indicators.

        Args:
            df (pd.DataFrame): DataFrame for a single ticker, loaded from CSV (TickerHistory).

        Returns:
            pd.DataFrame: DataFrame with added performance indicators, or None if input DataFrame is invalid.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("Warning: Input DataFrame for performance calculation is invalid or empty.")
            return None # Handle invalid DataFrame gracefully

        # 1. Select Used Columns
        try:
            df = df[RAW_DATA_USED_COLUMNS] # Select only the columns we need, error if not present
        except KeyError as e:
            print(f"Error: Raw data is missing required columns: {e}. Required columns: {RAW_DATA_USED_COLUMNS}")
            return None

        # 2. Calculate Performance Indicators - Extend DataFrame with Moving Averages and Rolling Hi-Lo
        try:
            MovingAverage(5).extend(df)
            MovingAverage(20).extend(df)
            MovingAverage(50).extend(df)
            RollingHiLo(5).extend(df)
            RollingHiLo(20).extend(df)
            RollingHiLo(50).extend(df)
        except ValueError as e: # Catch potential errors from indicator calculations
            print(f"Error during performance indicator calculation: {e}")
            return None

        return df

    def save_to_csv(self, file_path):
        """
        Saves the aggregated model data to a CSV file.

        Args:
            file_path (str): Full path to the CSV file where the data will be saved.
                             Directory path will be created if it does not exist.
        """
        if self.data is None:
            print("Warning: No data available to save. Data might not have been loaded or processed successfully.")
            return False

        os.makedirs(os.path.dirname(file_path), exist_ok=True) # Ensure directory exists
        self.data.to_csv(file_path)
        print(f"✅ Model data saved to CSV file: {file_path}")
        return True


    def create_simulations(self, simulation_window_days, portfolio_value):
        """
        Creates a list of Simulation objects from the model data.

        Each Simulation object represents an independent simulation starting at each date
        in the model data (within valid date range) and extending for 'simulation_window_days'.
        Each simulation is initialized with a new Portfolio object.

        Args:
            simulation_window_days (int): Number of days for each simulation window.
            portfolio_value (float or int): Starting portfolio value for each simulation.

        Returns:
            list of Simulation: A list of Simulation objects, or an empty list if no simulations could be created.
        """
        if not isinstance(simulation_window_days, int) or simulation_window_days <= 0:
            raise ValueError("simulation_window_days must be a positive integer.")
        if not isinstance(portfolio_value, (int, float)) or portfolio_value < 0:
            raise ValueError("portfolio_value must be a non-negative number.")
        if self.data is None or self.data.empty:
            print("Warning: No data available to create simulations. Model data is empty or not loaded.")
            return [] # Return empty list if no data


        simulations = []
        valid_simulation_dates = 0 # Counter for successfully created simulations

        unique_dates = sorted(self.data.index.unique()) # Get sorted unique dates from index for iteration

        for date in unique_dates:
            end_date = date + pd.Timedelta(days=simulation_window_days) # Use pandas Timedelta for date calculations
            if end_date <= self.date_max:
                portfolio = Portfolio(portfolio_value, self.tickers) # Create new Portfolio for each simulation
                simulation_data = self.data.loc[date : end_date] # Extract simulation data window
                simulations.append(Simulation(simulation_data, portfolio)) # Assuming Simulation class exists and takes data and portfolio
                valid_simulation_dates += 1

        if not simulations:
            print("Warning: No simulations could be created within the available date range and simulation window.")
        else:
            print(f"✅ Created {valid_simulation_dates} simulations, each with a {simulation_window_days}-day window.")

        return simulations


# Example Usage (for testing and demonstration - assuming data files exist in 'raw_data' directory)
if __name__ == '__main__':
    # Example tickers and raw data directory (you might need to adjust raw_data_dir to point to your actual data)
    example_tickers = ['AAPL', 'GOOG'] # Make sure you have AAPL.csv and GOOG.csv in raw_data_dir
    example_raw_data_dir = 'raw_data'  # Replace with the actual path to your raw data directory

    try:
        # Create ModelData instance - this will load and process data
        model_data = ModelData(example_tickers, example_raw_data_dir)

        if model_data.data is not None: # Check if data loading was successful
            print("\n--- Model Data Summary ---")
            print(f"Tickers in model data: {model_data.tickers}")
            print(f"Data date range: {model_data.date_min.date()} to {model_data.date_max.date()}")
            print(f"Shape of aggregated data: {model_data.data.shape}")

            # Save model data to CSV (optional)
            output_csv_path = 'transformed_data/model_data.csv' # Example output path
            if model_data.save_to_csv(output_csv_path):
                print(f"Model data saved to: {output_csv_path}")

            # Create simulations
            simulation_window = 30 # Example simulation window in days
            initial_portfolio_value = 100000 # Example initial portfolio value
            simulations = model_data.create_simulations(simulation_window, initial_portfolio_value)
            print(f"\nNumber of simulations created: {len(simulations)}")

        else:
            print("\n❌ ModelData could not be initialized. Check warnings and errors above.")


    except ValueError as e:
        print(f"Error during ModelData initialization or processing: {e}")