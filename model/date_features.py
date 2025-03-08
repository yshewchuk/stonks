import pandas as pd
import numpy as np

from utils.dataframe import verify_dataframe_structure

class DateFeatures:
    """
    Calculates and adds 'DayOfWeek' and 'MonthOfYear' columns to a Pandas DataFrame.

    Requires a DataFrame with a 'Date' index of type pd.DatetimeIndex.
    """

    def __init__(self):
        """
        Initializes DateFeatures calculator. No parameters needed.
        """
        pass # No initialization parameters needed

    def extend(self, df):
        """
        Extends the input DataFrame with 'DayOfWeek' and 'MonthOfYear' columns
        based on the 'Date' index.

        Verifies:
            - DataFrame has 'Date' index of type pd.DatetimeIndex.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date' index.

        Returns:
            pd.DataFrame: DataFrame with added 'DayOfWeek' and 'MonthOfYear' columns.

        Raises:
            ValueError: If input DataFrame does not have a valid 'Date' index.
        """
        # --- Input Verification ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a 'Date' index of type pd.DatetimeIndex.")
        if df.index.name != 'Date':
            raise ValueError("DataFrame index must be named 'Date'.")

        # --- Extend DataFrame ---
        # Day of Week (0=Monday, 6=Sunday) - Scaled to 0-1
        df['DayOfWeek'] = df.index.dayofweek / 6.0

        # Month of Year (0=January, 11=December) - Scaled to 0-1
        df['MonthOfYear'] = (df.index.month - 1) / 11.0

        return df