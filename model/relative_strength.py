import pandas as pd
import numpy as np

from utils.dataframe import verify_dataframe_structure

REQUIRED_COLUMNS_RSI = {
    'Close': float
}

class RSI:
    """
    Calculates and adds the Relative Strength Index (RSI) column to a Pandas DataFrame.

    Verifies input DataFrame structure (Date index, Close column) and handles
    rolling calculations for time-series data.
    """

    def __init__(self, window):
        """
        Initializes RSI calculator.

        Args:
            window (int): The time window (periods) for RSI calculation (e.g., 14).
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        self.window = window

    def extend(self, df):
        """
        Extends the input DataFrame with an RSI column ('RSI{window}').

        Verifies:
            - DataFrame has 'Date' index of type pd.DatetimeIndex.
            - DataFrame has a 'Close' column that is numeric.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date' index and 'Close' column.

        Returns:
            pd.DataFrame: DataFrame with an added RSI column ('RSI{window}').

        Raises:
            ValueError: If input DataFrame structure is invalid.
        """
        # --- Input Verification ---
        if not verify_dataframe_structure(df, REQUIRED_COLUMNS_RSI, ignore_extra_columns=True, expected_index_name='Date', expected_index_dtype=np.datetime64):
            print(f"❌ RSI: Missing required column or index")
            raise Exception("Invalid input for RSI calculation")

        # --- RSI Calculation ---
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df[f'RSI{self.window}'] = rsi

        return df