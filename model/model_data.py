# In model/model_data.py, inside the ModelData class:

import pandas as pd
import os
import json

class ModelData:
    """
    Data Transfer Object (DTO) representing a data window
    prepared for model input (price prediction or simulation).

    Holds both scaled (historical_data - model input) and unscaled
    (complete_data - for reward calculation, order execution, etc.) DataFrames,
    along with tickers and date range for the window.
    """

    def __init__(self, historical_data_df, complete_data_df, tickers, start_date=None, end_date=None): # Added start_date and end_date to constructor
        """
        Initializes ModelData DTO with pre-scaled historical data and unscaled complete data windows.

        Args:
            historical_data_df (pd.DataFrame): DataFrame representing a window of window-based price SCALED data.
                                                This is the data intended for model input.
            complete_data_df (pd.DataFrame): DataFrame representing a window of UNscaled, but fully processed data.
                                                This DataFrame contains all indicators, probabilities, and original prices, for other uses like reward calculation.
            tickers (list of str): List of stock tickers in the data window.
            start_date (pd.Timestamp, optional): Start date of the data window. Defaults to None, will be inferred from complete_data_df index.
            end_date (pd.Timestamp, optional): End date of the data window. Defaults to None, will be inferred from complete_data_df index.
        """
        if not isinstance(historical_data_df, pd.DataFrame) or historical_data_df.empty:
            raise ValueError("Input historical_data_df must be a non-empty Pandas DataFrame.")
        if not isinstance(complete_data_df, pd.DataFrame) or complete_data_df.empty:
            raise ValueError("Input complete_data_df must be a non-empty Pandas DataFrame.")
        if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
            raise ValueError("Tickers must be a list of strings.")

        self.__tickers = tickers
        self.__historical_data = historical_data_df # Store the ALREADY SCALED historical data directly

        self.__complete_data = complete_data_df  # Store the complete, UNscaled data directly

        if start_date is None:
            self.__start_date = self.__historical_data.index.min() # Determine start and end dates from complete_data index
        else:
            self.__start_date = pd.to_datetime(start_date)
        if end_date is None:
            self.__end_date = self.__historical_data.index.max()
        else:
            self.__end_date = pd.to_datetime(end_date)


    @property
    def tickers(self):
        """Returns the list of tickers."""
        return self.__tickers

    @property
    def historical_data(self):
        """Returns the window-based price scaled DataFrame (for model input)."""
        return self.__historical_data

    @property
    def complete_data(self):
        """Returns the complete, UNscaled but processed DataFrame (for reward calculation, etc.)."""
        return self.__complete_data

    @property
    def start_date(self):
        """Returns the start date of the data window."""
        return self.__start_date

    @property
    def end_date(self):
        """Returns the end date of the data window."""
        return self.__end_date

    def save_to_disk(self, filepath_prefix):
        """
        Saves the ModelData DTO to disk as CSV files.

        Args:
            filepath_prefix (str): Prefix for the file paths.
                                   Will be used to create files like:
                                   - {filepath_prefix}_historical_data.csv
                                   - {filepath_prefix}_complete_data.csv
                                   - {filepath_prefix}_tickers.json
        """
        if not os.path.exists(os.path.dirname(filepath_prefix)):
            os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)

        historical_data_filepath = f"{filepath_prefix}_historical_data.csv"
        complete_data_filepath = f"{filepath_prefix}_complete_data.csv"
        tickers_filepath = f"{filepath_prefix}_tickers.json"

        try:
            self.historical_data.to_csv(historical_data_filepath, index=True)
            print(f"✅ ModelData historical_data saved to: {historical_data_filepath}")
            self.complete_data.to_csv(complete_data_filepath, index=True)
            print(f"✅ ModelData complete_data saved to: {complete_data_filepath}")
            with open(tickers_filepath, 'w') as f:
                json.dump(self.tickers, f)
            print(f"✅ ModelData tickers saved to: {tickers_filepath}")

        except Exception as e:
            print(f"❌ Error saving ModelData to disk: {e}")
            return False
        return True

    @classmethod
    def load_from_disk(cls, filepath_prefix):
        """
        Loads a ModelData DTO from disk from CSV files, skipping the first two header rows.
        """
        historical_data_filepath = f"{filepath_prefix}_historical_data.csv"
        complete_data_filepath = f"{filepath_prefix}_complete_data.csv"
        tickers_filepath = f"{filepath_prefix}_tickers.json"

        try:
            historical_data_df = pd.read_csv(historical_data_filepath, index_col=0, header=[0,1,2], parse_dates=True) # Skip first 2 rows, no header row
            complete_data_df = pd.read_csv(complete_data_filepath, index_col=0, header=[0,1,2], parse_dates=True) # Skip first 2 rows, no header row
            with open(tickers_filepath, 'r') as f:
                tickers = json.load(f)

            return cls(historical_data_df, complete_data_df, tickers) # Use the constructor, passing tickers

        except FileNotFoundError:
            print(f"❌ File not found while loading ModelData from disk. Missing files for: {filepath_prefix}")
            return None
        except Exception as e:
            print(f"❌ Error loading ModelData from disk: {e}, Fileprefix: {filepath_prefix}") # Include file prefix in error message
            return None

# In model/data_manager.py, update the example usage (if __name__ == '__main__': section):

# Example Usage (for testing and demonstration)
if __name__ == '__main__':
    # ... (rest of your existing example code in DataManager.py) ...

    # --- Example: Save and Load ModelData DTO ---
    print("\n--- Example: Save and Load ModelData DTO ---")
    sample_window_dto_to_save = next(price_prediction_windows) # Get another ModelData DTO (or reuse the previous one)
    save_filepath_prefix = 'model_data_dto/sample_dto' # Define where to save

    if sample_window_dto_to_save.save_to_disk(save_filepath_prefix):
        print(f"✅ ModelData DTO saved with prefix: {save_filepath_prefix}")

        loaded_dto = ModelData.load_from_disk(save_filepath_prefix)
        if loaded_dto:
            print(f"✅ ModelData DTO loaded successfully from prefix: {save_filepath_prefix}")
            print(f"Loaded DTO tickers: {loaded_dto.tickers}")
            print(f"Loaded DTO historical_data shape: {loaded_dto.historical_data.shape}")
            print(f"Loaded DTO complete_data shape: {loaded_dto.complete_data.shape}")
            print(f"Loaded DTO start date: {loaded_dto.start_date.date()}, end date: {loaded_dto.end_date.date()}")

            # Verify loaded data (optional - compare a few values)
            original_historical_sample = sample_window_dto_to_save.historical_data.iloc[0, 0]
            loaded_historical_sample = loaded_dto.historical_data.iloc[0, 0]
            print(f"\nSample value - Original historical data: {original_historical_sample}, Loaded historical data: {loaded_historical_sample}")

        else:
            print(f"❌ Failed to load ModelData DTO from prefix: {save_filepath_prefix}")
    else:
        print(f"❌ Failed to save ModelData DTO with prefix: {save_filepath_prefix}")