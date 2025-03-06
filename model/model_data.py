# In model/model_data.py, inside the ModelData class:

import pandas as pd

class ModelData:
    """
    Data Transfer Object (DTO) representing a data window
    prepared for model input (price prediction or simulation).

    Holds both scaled (historical_data - model input) and unscaled
    (complete_data - for reward calculation, order execution, etc.) DataFrames,
    along with tickers and date range for the window.
    """

    def __init__(self, historical_data_df, complete_data_df, tickers): # Modified constructor to accept historical_data_df and complete_data_df
        """
        Initializes ModelData DTO with pre-scaled historical data and unscaled complete data windows.

        Args:
            historical_data_df (pd.DataFrame): DataFrame representing a window of window-based price SCALED data.
                                                This is the data intended for model input.
            complete_data_df (pd.DataFrame): DataFrame representing a window of UNscaled, but fully processed data.
                                              This DataFrame contains all indicators, probabilities, and original prices, for other uses like reward calculation.
            tickers (list of str): List of stock tickers in the data window.
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
        self.__start_date = self.complete_data.index.min() # Determine start and end dates from complete_data index
        self.__end_date = self.complete_data.index.max()

    # def _scale_window(self, window_df): # REMOVED - Scaling is now done by DataManager, ModelData DTO is NOT responsible for scaling
    #     # ... (removed scaling logic) ...

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