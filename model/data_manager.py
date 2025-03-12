import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model.date_features import DateFeatures
from model.moving_average import MovingAverage
from model.portfolio import Portfolio
from model.rolling_hi_lo import RollingHiLo
from model.max_percent_change_per_day import DailyPercentChange
from model.simulation import Simulation
from model.ticker_history import TickerHistory
from model.model_data import ModelData # Import the ModelData DTO
from model.percent_price_change_probability import PercentPriceChangeProbability
from model.relative_strength import RSI
from model.macd import MACD
from model.lagged_features import LaggedFeatures

RAW_DATA_USED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']  # Columns used from raw dataset
PRICE_COLUMN_TAGS = ['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo'] # List of price column tags

# --- NEW: Allowlist for historical_data features ---
NON_STOCK_FEATURES = ['DayOfWeek', 'MonthOfYear']
HISTORICAL_DATA_FEATURE_ALLOWLIST = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA5', 'MA20', 'MA50',
    'Hi5', 'Lo5',
    'Hi20', 'Lo20',
    'Hi50', 'Lo50',
    'RSI5', 'RSI20', 'RSI50',
    'MoACD_Fast12_Slow26', 'MoACD_Signal_9', 'MoACD_Hist_12_26_9' # Include MACD features in allowlist
    # Exclude: DailyPercentChange (MaxPI30, MaxPD30, MaxDPI30) and PercentPriceChangeProbability (PPC_* columns) - these are future-looking
]

LAG_PERIODS = [1, 2, 3] # Define lag periods in days

class DataManager:
    """
    Manages stock market data, preparing it for model training and simulation.

    Loads historical stock data, calculates indicators, applies scaling,
    and generates ModelData DTO instances (data windows) for use in models and simulations.
    """

    def __init__(self, tickers, raw_data_dir):
        """
        Initializes DataManager object.

        Loads historical data, calculates indicators, and applies global volume scaling.

        Args:
            tickers (list of str): List of stock ticker symbols.
            raw_data_dir (str): Path to the directory containing raw data CSV files.
        """
        if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
            raise ValueError("Tickers must be a list of strings.")
        if not os.path.isdir(raw_data_dir):
            raise ValueError(f"Raw data directory '{raw_data_dir}' is not a valid directory.")

        self.tickers = list(tickers)
        self.raw_data_dir = raw_data_dir
        self.stock_data = {}
        self.data = None
        self.date_min = None
        self.date_max = None
        self.volume_scalers = {} # Store scalers for volume columns
        self.rsi_scalers = {} # Store scalers for RSI features - NEW
        self.macd_line_scalers = {} # Scalers for MACD Line - NEW
        self.macd_signal_scalers = {} # Scalers for MACD Signal - NEW
        self.macd_hist_scalers = {} # Scalers for MACD Histogram - NEW


        self._load_and_process_data()  # Load and process data during initialization

    def _load_and_process_data(self):
        """
        Loads raw data, calculates indicators, and aggregates data for all tickers.
        Ensures the index of self.data is UNIQUE and SORTED once during initialization.
        GLOBAL VOLUME scaling and WINDOW-BASED PRICE scaling are handled within _generate_processed_model_data.
        """
        processed_stock_data = {}

        for ticker in self.tickers:
            print(f'Reading data file for ticker: {ticker}')
            file_path = os.path.join(self.raw_data_dir, f'{ticker}.csv')
            history = TickerHistory(ticker)
            if not history.load_from_csv(file_path):
                print(f"Warning: Could not load data from CSV for ticker {ticker} at {file_path}. Skipping ticker.")
                continue

            processed_df = self._standardize_and_calculate_performance(history.dataframe(), ticker) # Pass ticker here
            if processed_df is not None:
                processed_stock_data[ticker] = processed_df
            else:
                print(f"Warning: Processing failed for ticker {ticker}. Skipping ticker.")

        if not processed_stock_data:
            print("Error: No valid stock data loaded for any ticker. DataManager initialization failed.")
            self.data = None
            return

        self.stock_data = processed_stock_data
        print(f"Before pd.concat, processed_stock_data keys: {list(self.stock_data.keys())}") # ADDED: Print keys
        for ticker, df in self.stock_data.items(): # ADDED: Print shape and date range of each df
            if df is not None and not df.empty:
                print(f"  Ticker: {ticker}, Shape: {df.shape}, Date Range: {df.index.min()} to {df.index.max()}")
                # Explicitly convert index to DateTimeIndex BEFORE concat
                processed_stock_data[ticker].index = pd.to_datetime(processed_stock_data[ticker].index) # Convert index here
            else:
                print(f"  Ticker: {ticker}, DataFrame is None or Empty.")


        # CHANGE join='inner' to join='outer' and REMOVE dropna() temporarily
        self.data = pd.concat(processed_stock_data, axis='columns', join='outer', keys=processed_stock_data.keys(), names=['Ticker']) # Changed to 'outer' join, dropna REMOVED
        print(f"After pd.concat, shape of self.data: {self.data.shape}") # ADDED: Shape after concat
        print(f"Sample of self.data index after concat (before dropna/dedup):\n{self.data.index[:5]}") # ADDED: Sample index
        print(f"Sample of self.data head after concat (before dropna/dedup):\n{self.data.head().to_string()}") # ADDED: Sample data
        self.data = self.data.dropna()

        DateFeatures().extend(self.data)

        # --- ENSURE UNIQUE AND SORTED INDEX ---
        if not self.data.index.is_unique: # Check for index uniqueness
            print("Warning: Data index is not unique. Deduplicating index by keeping first entries.")
            self.data = self.data[~self.data.index.duplicated(keep='first')] # Keep first occurrence of duplicate indices

        self.data.sort_index(inplace=True) # Sort the index IN-PLACE after ensuring uniqueness

        self.date_min = self.data.index.min()
        self.date_max = self.data.index.max()

        print(f"✅ DataManager data aggregated (indicators calculated, scaling NOT yet applied), index UNIQUE and SORTED for tickers: {self.tickers}. Date range: {self.date_min.date()} to {self.date_max.date()}")

    def _standardize_and_calculate_performance(self, df, ticker): # Added ticker argument
        """
        Internal method to standardize data and calculate performance indicators
        (including Percent Price Change Probabilities) for a single ticker DataFrame.

        Extends the DataFrame with Moving Averages, Rolling Hi-Lo range indicators,
        and Percent Price Change Probability columns.
        Also adds lagged features for allowed historical data columns.
        Note: Volume scaling is handled GLOBALLY in _scale_volume_data_globally method.
              Price scaling is WINDOW-BASED and will be applied later by DataManager.

        Args:
            df (pd.DataFrame): DataFrame for a single ticker, loaded from CSV (TickerHistory).
            ticker (str): The ticker symbol for which the DataFrame is being processed. # Added ticker argument

        Returns:
            pd.DataFrame: DataFrame with added performance indicators and lagged features,
                          or None if input DataFrame is invalid.
        """
        print(f"Processing ticker: {ticker}") # ADDED: Start processing message
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("Warning: Input DataFrame for performance calculation is invalid or empty.")
            return None

        # 1. Select Used Columns
        try:
            print(f"  Before column selection, columns: {df.columns.tolist()}") # ADDED: Columns before selection
            df = df[RAW_DATA_USED_COLUMNS]
            print(f"  After column selection, columns: {df.columns.tolist()}, shape: {df.shape}") # ADDED: Columns and shape after selection
        except KeyError as e:
            print(f"Error: Raw data is missing required columns: {e}. Required columns: {RAW_DATA_USED_COLUMNS}")
            return None

        # 2. Calculate Performance Indicators - Extend DataFrame with Moving Averages, Rolling Hi-Lo, and Percent Price Change Probabilities
        try:
            MovingAverage(5).extend(df)
            MovingAverage(20).extend(df)
            MovingAverage(50).extend(df)
            RSI(5).extend(df)
            RSI(20).extend(df)
            RSI(50).extend(df)
            RollingHiLo(5).extend(df)
            RollingHiLo(20).extend(df)
            RollingHiLo(50).extend(df)
            MACD().extend(df) # Calculate MACD indicators

            DailyPercentChange(30).extend(df)

            # --- Calculate Percent Price Change Probabilities ---
            probability_calculators = [
                PercentPriceChangeProbability(1, 3, max_percent_change=-0.07),
                PercentPriceChangeProbability(1, 3, min_percent_change=-0.07, max_percent_change=-0.03),
                PercentPriceChangeProbability(1, 3, min_percent_change=-0.03, max_percent_change=-0.01),
                PercentPriceChangeProbability(1, 3, min_percent_change=-0.01, max_percent_change=0.01),
                PercentPriceChangeProbability(1, 3, min_percent_change=0.01, max_percent_change=0.03),
                PercentPriceChangeProbability(1, 3, min_percent_change=0.03, max_percent_change=0.07),
                PercentPriceChangeProbability(1, 3, min_percent_change=0.07),

                PercentPriceChangeProbability(4, 10, max_percent_change=-0.10),
                PercentPriceChangeProbability(4, 10, min_percent_change=-0.10, max_percent_change=-0.04),
                PercentPriceChangeProbability(4, 10, min_percent_change=-0.04, max_percent_change=-0.02),
                PercentPriceChangeProbability(4, 10, min_percent_change=-0.02, max_percent_change=0.02),
                PercentPriceChangeProbability(4, 10, min_percent_change=0.02, max_percent_change=0.04),
                PercentPriceChangeProbability(4, 10, min_percent_change=0.04, max_percent_change=0.10),
                PercentPriceChangeProbability(4, 10, min_percent_change=0.10),

                PercentPriceChangeProbability(11, 30, max_percent_change=-0.12),
                PercentPriceChangeProbability(11, 30, min_percent_change=-0.12, max_percent_change=-0.05),
                PercentPriceChangeProbability(11, 30, min_percent_change=-0.05, max_percent_change=-0.03),
                PercentPriceChangeProbability(11, 30, min_percent_change=-0.03, max_percent_change=0.03),
                PercentPriceChangeProbability(11, 30, min_percent_change=0.03, max_percent_change=0.05),
                PercentPriceChangeProbability(11, 30, min_percent_change=0.05, max_percent_change=0.12),
                PercentPriceChangeProbability(11, 30, min_percent_change=0.12)
            ]
            for calculator in probability_calculators:
                df = calculator.extend(df)

        except ValueError as e:  # Catch potential errors from indicator calculations
            print(f"Error during performance indicator calculation: {e}")
            return None

        # 3. Add Lagged Features - Using LaggedFeatures Class
        lag_feature_calculator = LaggedFeatures(LAG_PERIODS, HISTORICAL_DATA_FEATURE_ALLOWLIST) # Instantiate LaggedFeatures
        df = lag_feature_calculator.extend(df) # Extend DataFrame with lagged features


        print(f"  Successfully processed ticker: {ticker}, final shape: {df.shape}") # ADDED: Success message and final shape
        return df

    def _generate_processed_model_data(self, start_date, data_window_size, scaling_window_size):
        """
        Internal method to generate a processed ModelData DTO for a given start date and window sizes.

        This method extracts data for the specified data window, performs volume scaling
        using the scaling window, applies window-based price scaling, and creates a ModelData DTO.

        Args:
            start_date (pd.Timestamp): The starting date for both data and scaling windows.
            data_window_size (int): Size of the data window in days.
            scaling_window_size (int): Size of the scaling window in days.

        Returns:
            ModelData: A ModelData DTO instance containing scaled historical data and unscaled complete data.
                        Returns None if data window is empty or scaling fails.
        """
        if not isinstance(start_date, pd.Timestamp):
            raise ValueError("start_date must be a pandas Timestamp.")
        if not isinstance(data_window_size, int) or data_window_size <= 0:
            raise ValueError("data_window_size must be a positive integer.")
        if not isinstance(scaling_window_size, int) or scaling_window_size <= 0 or scaling_window_size > data_window_size:
            raise ValueError("scaling_window_size must be a positive integer that is less than or equal to the data window size.")

        # --- Generate Date Ranges ---
        unique_dates = self.data.index # Use pre-sorted unique index

        start_date_index = unique_dates.get_loc(start_date) # Find the index of the start_date in the unique dates

        data_window_end_index = start_date_index + data_window_size
        if data_window_end_index > len(unique_dates):
            print(f"Warning: Data window exceeds available dates from start date {start_date.date()}. Not enough data for window of size {data_window_size}.")
            return None # Not enough data for the data window

        data_window_dates = unique_dates[start_date_index:data_window_end_index] # Extract data window dates

        if not len(data_window_dates): # Check if data_window_dates is empty after extraction
            print("Warning: No data window dates generated. Check date range and window sizes.")
            return None

        unscaled_window_df = self.data.loc[data_window_dates].copy() # Copy here to avoid modifying original data
        if unscaled_window_df.empty:
            print("Warning: Extracted data window DataFrame is empty after date range extraction.")
            return None

        # --- **NEW: Filter columns for historical_data based on allowlist** ---
        historical_data_columns = []
        for col in unscaled_window_df.columns:
            if col[0] in NON_STOCK_FEATURES or col[1].split('_Lag')[0] in HISTORICAL_DATA_FEATURE_ALLOWLIST: # Check original column tag before lagging
                historical_data_columns.append(col)

        # --- Scaling (Apply scaling WITHIN this data window using the scaling_window dates) ---
        scaled_window_df = unscaled_window_df[historical_data_columns].copy() # Start with a copy for scaling

        scaling_window_df = unscaled_window_df.iloc[:scaling_window_size] # <---- **REFINED:** Scaling window from *unscaled* data
        scaled_window_df = self._scale_non_price_window(scaled_window_df, scaling_window_df) # Scale non-price data (Volume, RSI, MACD)
        scaled_window_df = self._scale_price_window(scaled_window_df) # Apply window-based price scaling

        # --- Drop rows with NaN introduced by lagging, AFTER scaling ---
        scaled_window_df.dropna(inplace=True)
        unscaled_window_df = unscaled_window_df.loc[scaled_window_df.index] # Keep only non-NaN rows in unscaled_window_df as well

        # --- Create and return ModelData DTO with both scaled and unscaled DataFrames ---
        model_data_dto = ModelData(scaled_window_df, unscaled_window_df, self.tickers) # Pass actual start and end dates after dropping NaNs
        return model_data_dto

    def _scale_non_price_window(self, window_df, scaling_window_df):
        """
        Scales non-price-related data within a given data window (DataFrame) such as volume, RSI, MACD,
        and other values that aren't specific to the price of the stock.
        Scales columns based on prefixes: 'Volume', 'RSI', 'MoACD'.

        Args:
            window_df (pd.DataFrame): A DataFrame representing a data window for a single ticker.
            scaling_window_df (pd.DataFrame): DataFrame for the scaling window (subset of data window).

        Returns:
            pd.DataFrame: The scaled data window DataFrame.
        """

        # --- **VERIFICATION: Ensure scaling_window_dates are within data_window_dates timeframe** ---
        data_window_date_set = set(window_df.index)
        scaling_window_date_set = set(scaling_window_df.index)

        if not scaling_window_date_set.issubset(data_window_date_set): # Check if scaling_window_dates is a subset of data_window_dates
            print("Warning: Scaling window dates are NOT fully contained within data window dates.")
            raise ValueError("scaling_window_df must be a subset of window_df.")

        prefixes_to_scale = ['Volume', 'RSI', 'MoACD'] # Prefixes for columns to scale
        scalers = {} # Dictionary to store scalers, keyed by (ticker, prefix)


        for ticker in self.tickers:
            for prefix in prefixes_to_scale:
                scalers[(ticker, prefix)] = MinMaxScaler() # Initialize scaler for each ticker and prefix

            for col in window_df.columns:
                col_ticker, col_name = col # Unpack MultiIndex

                if col_ticker == ticker: # Only process for the current ticker
                    for prefix in prefixes_to_scale:
                        if col_name.startswith(prefix): # Check if column name starts with the prefix
                            if prefix == 'Volume':
                                scaling_data = scaling_window_df[col].values.reshape(-1, 1) # Use scaling_window_df for Volume
                                if not scaling_data.size: # Handle empty scaling data
                                    print(f"Warning: No scaling data for {col} in scaling window for ticker {ticker}. Skipping scaling for this column in this window.")
                                    continue # Skip to the next column if no scaling data
                                scaler = scalers[(ticker, prefix)] # Get pre-initialized scaler
                                scaler.fit(scaling_data) # Fit scaler with scaling window data only for Volume
                                data_in_window = window_df[col].values.reshape(-1, 1)
                                window_df[col] = scaler.transform(data_in_window).flatten()
                            elif prefix == 'RSI' or prefix == 'MoACD':
                                scaler = scalers[(ticker, prefix)] # Get pre-initialized scaler
                                window_df[col] = scaler.fit_transform(window_df[col].values.reshape(-1, 1)).flatten() # Fit AND transform RSI and MoACD in window directly


        return window_df

    def _scale_price_window(self, window_df):
        """
        Scales price-related data within a given data window (DataFrame)
        relative to the 'Open' price of the first day in the window.
        Volume data in the window is assumed to be already globally scaled.

        Args:
            window_df (pd.DataFrame): A DataFrame representing a data window for a single ticker.

        Returns:
            pd.DataFrame: The scaled data window DataFrame.
        """
        for ticker in self.tickers:
            first_open_price_in_window = window_df.iloc[0][(ticker, 'Open')]
            if first_open_price_in_window == 0:
                first_open_price_in_window = 1.0 # Avoid division by zero

            for col in self.data.columns:
                if col[0] == ticker and any(tag in col[1].split('_Lag')[0] for tag in PRICE_COLUMN_TAGS): # Check original tag before lagging
                    window_df[col] = (window_df[col] - first_open_price_in_window) / first_open_price_in_window

        return window_df

    def create_price_prediction_windows(self, window_size=60, step_size=1):
        """
        Generates ModelData DTO instances (data windows) for price prediction model training.

        Each window is of 'window_size' days. ModelData DTOs will contain both:
        1. window-based price scaled 'historical_data' (for model input) - SCALING DONE BY DATAMANAGER
        2. complete, unscaled, processed 'complete_data' (for reward calculation, etc.)

        Volume, RSI, and MACD data in windows are also window-based scaled within `_scale_non_price_window`.

        Args:
            window_size (int): The size of each data window in days (e.g., 60 days).
            step_size (int): Step size for sliding window (e.g., 1 for daily windows).

        Yields:
            ModelData: ModelData DTO instance representing a data window with both scaled and unscaled data,
                         or None if no valid windows can be created.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("step_size must be a positive integer.")
        if self.data is None or self.data.empty:
            print("Warning: No data available to create prediction windows. Model data is empty or not loaded.")
            return  # Return empty generator

        start_index = 0
        while start_index + window_size <= len(self.data):
            start_date = self.data.index[start_index] # Get the start date for the current window

            model_data_dto = self._generate_processed_model_data(start_date, window_size, window_size) # Call _generate_processed_model_data with start_date and window sizes (SAME for data and scaling)

            if model_data_dto: # Check if DTO was successfully created (not None)
                yield model_data_dto # Yield the ModelData DTO
            else:
                print(f"Warning: Failed to generate ModelData DTO for window starting at", start_date.date(), ". Skipping window.")

            start_index += step_size


    def create_simulations(self, simulation_window_days, portfolio_value):
        """
        Creates a list of Simulation objects from ModelData DTO instances.

        Each Simulation object is created with a ModelData DTO representing an independent simulation window.
        ModelData DTOs now contain both:
        1. window-based price scaled 'historical_data' (for model input - though simulations might not use this directly) - SCALING DONE BY DATAMANAGER
        2. complete, unscaled, processed 'complete_data' (for simulation logic, reward calculation, etc.)

        Simulation data windows will still have GLOBALLY volume-scaled data (done by DataManager initialization)
        and WINDOW-based price scaled data (scaling applied by DataManager before DTO creation).

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
            return []  # Return empty list if no data

        simulations = []
        valid_simulation_dates = 0  # Counter for successfully created simulations

        unique_dates = self.data.index  # Get sorted unique dates for iteration

        for date in unique_dates: # 'date' here is the START date of the simulation window
            end_date = date + pd.Timedelta(days=simulation_window_days)  # Calculate end_date - not directly used in _generate_processed_model_data anymore, window size is used instead
            if end_date <= self.date_max:
                portfolio = Portfolio(portfolio_value, self.tickers)  # Create new Portfolio for each simulation

                model_data_dto = self._generate_processed_model_data(date, simulation_window_days, simulation_window_days) # Call _generate_processed_model_data with start_date and window sizes (SAME for data and scaling)

                if model_data_dto: # Check if DTO was successfully created (not None)
                    yield Simulation(model_data_dto, portfolio)  # Pass ModelData DTO to Simulation
                    valid_simulation_dates += 1
                else:
                    print("Warning: Failed to generate ModelData DTO for simulation starting at", date.date(), ". Skipping simulation.")

        if not simulations:
            print("Warning: No simulations could be created within the available date range and simulation window.")
        else:
            print(f"✅ Created {valid_simulation_dates} simulations, each with a {simulation_window_days}-day window.")


    def save_to_csv(self, file_path):
        """Saves the aggregated model data to a CSV file."""
        if self.data is None:
            print("Warning: No data available to save.")
            return False
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.data.to_csv(file_path)
        print(f"✅ DataManager data saved to CSV file: {file_path}")
        return True


# Example Usage (for testing and demonstration)
if __name__ == '__main__':
    example_tickers = ['AAPL', 'GOOG']
    example_raw_data_dir = 'raw_data'

    try:
        data_manager = DataManager(example_tickers, example_raw_data_dir) # Create DataManager instance (renamed class)

        if data_manager.data is not None:
            print("\n--- DataManager Summary ---")
            print(f"Tickers in data: {data_manager.tickers}")
            print(f"Data date range: {data_manager.date_min.date()} to {data_manager.date_max.date()}")
            print(f"Shape of aggregated data (globally volume-scaled, price NOT window-scaled): {data_manager.data.shape}")
            print("\n--- Sample of Data (Globally Volume Scaled, Price NOT yet window-scaled) ---")
            sample_output_unscaled_price = data_manager.data.sample(5).sort_index()
            print(sample_output_unscaled_price.to_string())

            # --- Generate Price Prediction Windows (now returns ModelData DTOs with both scaled and unscaled data) ---
            window_size = 60
            price_prediction_windows = data_manager.create_price_prediction_windows(window_size=window_size, step_size=30)
            sample_window_dto = next(price_prediction_windows) # Get the first ModelData DTO
            sample_scaled_window_df = sample_window_dto.get_historical_data() # Get SCALED DataFrame from DTO using getter
            sample_unscaled_window_df = sample_window_dto.get_complete_data() # Get UNSCALED DataFrame using getter
            window_start_date = sample_window_dto.get_start_date() # Get start date using getter
            window_end_date = sample_window_dto.get_end_date() # Get end date using getter

            print(f"\n--- Sample Price Prediction Window (Window-Based Scaled - from ModelData DTO, Size: {window_size} days) ---")
            print(sample_scaled_window_df.to_string()) # Print SCALED data sample
            print(f"\nScaled Data Columns:\n{sample_scaled_window_df.columns.to_list()}") # Print scaled data columns to verify lagged features

            print(f"\n--- Sample of UNscaled Complete Data (from ModelData DTO, Date Range: {window_start_date.date()} to {window_end_date.date()}) ---")
            print(sample_unscaled_window_df.sample(5).sort_index().to_string()) # Print UNSCALED data sample
            print(f"\nUnscaled Data Columns:\n{sample_unscaled_window_df.columns.to_list()}") # Print unscaled data columns
            print(f"\nModelData DTO Start Date: {window_start_date.date()}, End Date: {window_end_date.date()}")


            # Save model data to CSV (optional - saves globally volume-scaled data)
            output_csv_path = 'transformed_data/data_manager_output.csv' # Changed filename to data_manager_output
            if data_manager.save_to_csv(output_csv_path):
                print(f"DataManager data (globally volume-scaled, price NOT window-scaled in main data) saved to: {output_csv_path}")


            # Example of creating simulations (simulations will now also use ModelData DTOs)
            simulation_window = 30
            initial_portfolio_value = 100000
            simulations = data_manager.create_simulations(simulation_window, initial_portfolio_value)
            print(f"\nNumber of simulations created: {sum(1 for _ in simulations)}")

            # Example: Accessing scaled data from a ModelData DTO within simulations (if needed - Simulation class needs updates)
            for sim in simulations:
                scaled_sim_data = sim.model_data.get_historical_data()
                unscaled_sim_data = sim.model_data.get_complete_data()
                sim_start_date = sim.model_data.get_start_date()
                sim_end_date = sim.model_data.get_end_date()

                if scaled_sim_data is not None:
                    print(f"\nSample SCALED data from first day of a simulation window (ModelData DTO, Date: {sim_start_date.date()}):\n{scaled_sim_data.iloc[[0]].to_string()}") # Print sample of SCALED data
                    print(f"\nSample UNSCALED data from first day of simulation window (ModelData DTO, Date: {sim_start_date.date()}):\n{unscaled_sim_data.iloc[[0]].to_string()}") # Print sample of UNSCALED data
                    print(f"Simulation Window Date Range: {sim_start_date.date()} to {sim_end_date.date()}")
                    break # Just print for the first simulation as an example


        else:
            print("\n❌ DataManager could not be initialized. Check warnings and errors above.")

    except ValueError as e:
        print(f"Error during DataManager initialization or processing: {e}")