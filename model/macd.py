import pandas as pd
import numpy as np

from utils.dataframe import verify_dataframe_structure

REQUIRED_COLUMNS_MACD = {
    'Close': float
}

class MACD:
    """
    Calculates and adds MACD, MACD Signal, and MACD Histogram columns to a Pandas DataFrame.

    Verifies input DataFrame structure (Date index, Close column) and handles
    calculations for time-series data.
    """

    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Initializes MACD calculator.

        Args:
            fast_period (int): Period for the fast EMA (typically 12).
            slow_period (int): Period for the slow EMA (typically 26).
            signal_period (int): Period for the signal EMA (typically 9).
        """
        if not isinstance(fast_period, int) or fast_period <= 0:
            raise ValueError("Fast period must be a positive integer.")
        if not isinstance(slow_period, int) or slow_period <= 0:
            raise ValueError("Slow period must be a positive integer.")
        if not isinstance(signal_period, int) or signal_period <= 0:
            raise ValueError("Signal period must be a positive integer.")
        if fast_period >= slow_period:
            raise ValueError("Fast period must be shorter than slow period.")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def extend(self, df):
        """
        Extends the input DataFrame with MACD, Signal, and Histogram columns.

        Verifies:
            - DataFrame has 'Date' index of type pd.DatetimeIndex.
            - DataFrame has a 'Close' column that is numeric.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date' index and 'Close' column.

        Returns:
            pd.DataFrame: DataFrame with added MACD, Signal, and Histogram columns.

        Raises:
            ValueError: If input DataFrame structure is invalid.
        """
        # --- Input Verification ---
        if not verify_dataframe_structure(df, REQUIRED_COLUMNS_MACD, ignore_extra_columns=True, expected_index_name='Date', expected_index_dtype=np.datetime64):
            print(f"❌ MACD: Missing required column or index")
            raise Exception("Invalid input for MACD calculation")

        # --- MACD Calculation ---
        ema_fast = df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        df[f'MoACD_Fast{self.fast_period}_Slow{self.slow_period}'] = macd_line
        df[f'MoACD_Signal_{self.signal_period}'] = signal_line
        df[f'MoACD_Hist_{self.fast_period}_{self.slow_period}_{self.signal_period}'] = histogram

        return df