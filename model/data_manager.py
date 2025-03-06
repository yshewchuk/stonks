import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model.moving_average import MovingAverage
from model.portfolio import Portfolio
from model.rolling_hi_lo import RollingHiLo
from model.max_percent_change_per_day import DailyPercentChange
from model.simulation import Simulation
from model.ticker_history import TickerHistory
from model.model_data import ModelData # Import the ModelData DTO
from model.percent_price_change_probability import PercentPriceChangeProbability

RAW_DATA_USED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']  # Columns used from raw dataset
PRICE_COLUMN_TAGS = ['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo'] # List of price column tags

# --- NEW: Allowlist for historical_data features ---
HISTORICAL_DATA_FEATURE_ALLOWLIST = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA5', 'MA20', 'MA50',
    'Hi5', 'Lo5',
    'Hi20', 'Lo20',
    'Hi50', 'Lo50'
    # Exclude: DailyPercentChange (MaxPI30, MaxPD30, MaxDPI30) and PercentPriceChangeProbability (PPC_* columns) - these are future-looking
]

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

            processed_df = self._standardize_and_calculate_performance(history.dataframe())
            if processed_df is not None:
                processed_stock_data[ticker] = processed_df
            else:
                print(f"Warning: Processing failed for ticker {ticker}. Skipping ticker.")

        if not processed_stock_data:
            print("Error: No valid stock data loaded for any ticker. DataManager initialization failed.")
            self.data = None
            return

        self.stock_data = processed_stock_data
        self.data = pd.concat(self.stock_data, axis='columns', join='inner', keys=self.stock_data.keys(), names=['Ticker']).dropna()

        # --- ENSURE UNIQUE AND SORTED INDEX ---
        if not self.data.index.is_unique: # Check for index uniqueness
            print("Warning: Data index is not unique. Deduplicating index by keeping first entries.")
            self.data = self.data[~self.data.index.duplicated(keep='first')] # Keep first occurrence of duplicate indices
            # Alternative deduplication strategies if needed:
            # self.data = self.data.groupby(level=0).first() # Take the first entry for each date
            # self.data = self.data.groupby(level=0).mean()  # Take the mean for each date if averaging is more appropriate

        self.data.sort_index(inplace=True) # Sort the index IN-PLACE after ensuring uniqueness

        self.date_min = self.data.index.min()
        self.date_max = self.data.index.max()

        print(f"✅ DataManager data aggregated (indicators calculated, scaling NOT yet applied), index UNIQUE and SORTED for tickers: {self.tickers}. Date range: {self.date_min.date()} to {self.date_max.date()}")

    def _standardize_and_calculate_performance(self, df):
        """
        Internal method to standardize data and calculate performance indicators
        (including Percent Price Change Probabilities) for a single ticker DataFrame.

        Extends the DataFrame with Moving Averages, Rolling Hi-Lo range indicators,
        and Percent Price Change Probability columns.
        Note: Volume scaling is handled GLOBALLY in _scale_volume_data_globally method.
              Price scaling is WINDOW-BASED and will be applied later by DataManager.

        Args:
            df (pd.DataFrame): DataFrame for a single ticker, loaded from CSV (TickerHistory).

        Returns:
            pd.DataFrame: DataFrame with added performance indicators, or None if input DataFrame is invalid.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("Warning: Input DataFrame for performance calculation is invalid or empty.")
            return None

        # 1. Select Used Columns
        try:
            df = df[RAW_DATA_USED_COLUMNS]
        except KeyError as e:
            print(f"Error: Raw data is missing required columns: {e}. Required columns: {RAW_DATA_USED_COLUMNS}")
            return None

        # 2. Calculate Performance Indicators - Extend DataFrame with Moving Averages, Rolling Hi-Lo, and Percent Price Change Probabilities
        try:
            MovingAverage(5).extend(df)
            MovingAverage(20).extend(df)
            MovingAverage(50).extend(df)
            RollingHiLo(5).extend(df)
            RollingHiLo(20).extend(df)
            RollingHiLo(50).extend(df)
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
        
        unscaled_window_df = self.data.loc[data_window_dates]
        if unscaled_window_df.empty:
            print("Warning: Extracted data window DataFrame is empty after date range extraction.")
            return None

        # --- **NEW: Filter columns for historical_data based on allowlist** ---
        historical_data_columns = []
        for col in unscaled_window_df.columns:
            if col[1] in HISTORICAL_DATA_FEATURE_ALLOWLIST: # Check if column tag is in allowlist
                historical_data_columns.append(col)

        # --- Volume Scaling (Apply volume scaling WITHIN this data window using the scaling_window dates) ---
        scaled_window_df = unscaled_window_df[historical_data_columns].copy() # Start with a copy for scaling

        scaling_window_df = unscaled_window_df.iloc[:scaling_window_size] # <---- **REFINED:** Scaling window from *unscaled* data
        # --- Price Scaling (Apply window-based price scaling AFTER volume scaling) ---
        scaled_window_df = self._scale_volume_window(scaled_window_df, scaling_window_df)
        scaled_window_df = self._scale_price_window(scaled_window_df) # Apply window-based price scaling

        # --- Create and return ModelData DTO with both scaled and unscaled DataFrames ---
        model_data_dto = ModelData(scaled_window_df, unscaled_window_df, self.tickers)
        return model_data_dto

    def _scale_volume_window(self, window_df, scaling_window_df):
        """
        Scales price-related data within a given data window (DataFrame)
        relative to the 'Open' price of the first day in the window.
        Volume data in the window is assumed to be already globally scaled.

        Args:
            window_df (pd.DataFrame): A DataFrame representing a data window for a single ticker.

        Returns:
            pd.DataFrame: The scaled data window DataFrame.
        """

        # --- **VERIFICATION: Ensure scaling_window_dates are within data_window_dates timeframe** ---
        data_window_date_set = set(window_df.index)
        scaling_window_date_set = set(scaling_window_df.index)

        if not scaling_window_date_set.issubset(data_window_date_set): # Check if scaling_window_dates is a subset of data_window_dates
            print("Warning: Scaling window dates are NOT fully contained within data window dates.")
            raise ValueError("scaling_window_df must be a subset of window_df.")

        for ticker in self.tickers:
            # Fit volume scaler on the provided scaling_window dates
            scaling_volume_data = scaling_window_df[(ticker, 'Volume')].values.reshape(-1, 1)
            if not scaling_volume_data.size: # Handle case where scaling window has no data for volume scaling
                print(f"Warning: No volume data in scaling window for ticker {ticker}. Skipping volume scaling for this ticker in this window.")
                continue # Skip to next ticker - volume will remain unscaled in this window for this ticker

            scaler = MinMaxScaler()
            scaler.fit(scaling_volume_data)

            # Apply the scaler to the volume data within the CURRENT data window (data_window_dates)
            volume_data_in_window = window_df[(ticker, 'Volume')].values.reshape(-1, 1)
            window_df[(ticker, 'Volume')] = scaler.transform(volume_data_in_window).flatten() # Flatten to avoid column shape issues

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
                if col[0] == ticker and any(tag in col[1] for tag in PRICE_COLUMN_TAGS):
                    window_df[col] = (window_df[col] - first_open_price_in_window) / first_open_price_in_window

        return window_df

    def create_price_prediction_windows(self, window_size=60, step_size=1):
        """
        Generates ModelData DTO instances (data windows) for price prediction model training.

        Each window is of 'window_size' days. ModelData DTOs will contain both:
        1. window-based price scaled 'historical_data' (for model input) - SCALING DONE BY DATAMANAGER
        2. complete, unscaled, processed 'complete_data' (for reward calculation, etc.)

        Volume data in windows is GLOBALLY scaled (done during DataManager initialization).

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

            print(f"\n--- Sample of UNscaled Complete Data (from ModelData DTO, Date Range: {window_start_date.date()} to {window_end_date.date()}) ---")
            print(sample_unscaled_window_df.sample(5).sort_index().to_string()) # Print UNSCALED data sample
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