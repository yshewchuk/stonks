import pandas as pd
import numpy as np

from utils.dataframe import verify_dataframe_structure
from utils.obj import print_public_interface

REQUIRED_COLUMNS = {
    'Open': float
}

class DailyPercentChange:
    """
    Calculates and adds a max percent increase per day and max percent decrease per day

    Verifies input DataFrame structure (Date index, Open column) and handles
    rolling calculations for time-series data, inherently accounting for gaps
    in the DatetimeIndex like weekends.
    """

    def __init__(self, window):
        """
        Initializes Percent Change calculator.

        Args:
            window (int): The time window (in days/periods) for moving percent change calculation.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        self.window = window

    def __inc(self, series):
        div = series.iloc[0]
        
        curr = series.iloc[1] / div - 1

        for i in range(2, len(series)):
            curr = max(curr, (series.iloc[i] / div) - 1)

        return curr
    
    def __dec(self, series):
        div = series.iloc[0]
        
        curr = series.iloc[1] / div - 1

        for i in range(2, len(series)):
            curr = min(curr, (series.iloc[i] / div) - 1)

        return curr
    
    def __daily_inc(self, series):
        div = series.iloc[0]
        
        curr = series.iloc[1] / div - 1

        for i in range(2, len(series)):
            curr = max(curr, ((series.iloc[i] / div) - 1) / i)

        return curr
    
    def __daily_dec(self, series):
        div = series.iloc[0]
        
        curr = series.iloc[1] / div - 1

        for i in range(2, len(series)):
            curr = min(curr, ((series.iloc[i] / div) - 1) / i)

        return curr

    def extend(self, df):
        """
        Extends the input DataFrame with Max Daily Percent Increase (MaxDPI<window>) and Max Daily Percent Decrease (MaxDPD<window>) column of the 'Open' prices.

        Verifies:
            - DataFrame has 'Open' index of type pd.DatetimeIndex.
            - DataFrame has a 'Open' column that is numeric.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date' index and 'Open' column.

        Returns:
            pd.DataFrame: DataFrame with added DPI and DPD columns

        Raises:
            ValueError: If input DataFrame structure is invalid (missing index, wrong index type, missing 'Open' column, non-numeric 'Close' column).
        """
        # --- Input Verification ---
        if not verify_dataframe_structure(df, REQUIRED_COLUMNS, ignore_extra_columns=True, expected_index_name='Date', expected_index_dtype=np.datetime64):
            print(f"❌ Missing required column or index")
            raise Exception("Invalid input")

        # --- Extend DataFrame ---
        df[f'MaxDPI{self.window}'] = df['Open'].rolling(window=self.window).agg(self.__daily_inc).shift(1 - self.window) # Add Max increase column
        df[f'MaxDPD{self.window}'] = df['Open'].rolling(window=self.window).agg(self.__daily_dec).shift(1 - self.window) # Add Max decrease column

        df[f'MaxPI{self.window}'] = df['Open'].rolling(window=self.window).agg(self.__inc).shift(1 - self.window) # Add Max increase column
        df[f'MaxPD{self.window}'] = df['Open'].rolling(window=self.window).agg(self.__dec).shift(1 - self.window) # Add Max decrease column

        return df