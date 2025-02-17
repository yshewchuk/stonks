import pandas as pd
import numpy as np

from utils.dataframe import verify_dataframe_structure

REQUIRED_COLUMNS = {
    'Close': float
}

class RollingHiLo:
    """
    Calculates and adds a High and Low columns to a Pandas DataFrame.

    Verifies input DataFrame structure (Date index, Close column) and handles
    rolling calculations for time-series data, inherently accounting for gaps
    in the DatetimeIndex like weekends.
    """

    def __init__(self, window):
        """
        Initializes MovingAverage calculator.

        Args:
            window (int): The time window (in days/periods) for moving average calculation.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        self.window = window

    def extend(self, df):
        """
        Extends the input DataFrame with Hi and Lo column of the 'Close' prices.

        Verifies:
            - DataFrame has 'Date' index of type pd.DatetimeIndex.
            - DataFrame has a 'Close' column that is numeric.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date' index and 'Close' column.

        Returns:
            pd.DataFrame: DataFrame with an added Moving Average column ('MA{window}').

        Raises:
            ValueError: If input DataFrame structure is invalid (missing index, wrong index type, missing 'Close' column, non-numeric 'Close' column).
        """
        # --- Input Verification ---
        if not verify_dataframe_structure(df, REQUIRED_COLUMNS, ignore_extra_columns=True, expected_index_name='Date', expected_index_dtype=np.datetime64):
            print(f"❌ Missing required column or index")
            raise Exception("Invalid input")

        # --- Extend DataFrame ---
        df[f'Hi{self.window}'] = df['Close'].rolling(window=self.window).max().shift(1) # Add Hi column to DataFrame
        df[f'Lo{self.window}'] = df['Close'].rolling(window=self.window).min().shift(1) # Add Lo column to DataFrame

        return df