import pandas as pd
import numpy as np

class LaggedFeatures:
    """
    Extends a DataFrame with lagged features for specified columns.
    """
    def __init__(self, periods, columns_to_lag):
        """
        Initializes LaggedFeatures calculator.

        Args:
            periods (list of int): List of lag periods (in days).
            columns_to_lag (list of str): List of column names to create lagged features for.
        """
        if not isinstance(periods, list) or not all(isinstance(p, int) and p > 0 for p in periods):
            raise ValueError("Periods must be a list of positive integers.")
        if not isinstance(columns_to_lag, list) or not all(isinstance(col, str) for col in columns_to_lag):
            raise ValueError("columns_to_lag must be a list of strings.")

        self.periods = periods
        self.columns_to_lag = columns_to_lag

    def extend(self, df):
        """
        Extends the DataFrame with lagged features.

        For each column in columns_to_lag and each period in periods,
        creates a new column with lagged values.

        Args:
            df (pd.DataFrame): DataFrame to extend.

        Returns:
            pd.DataFrame: Extended DataFrame with lagged features.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")
            
        # Create a list to store all the lagged DataFrames
        lagged_dfs = []
        
        # For each period and column, create a DataFrame with the lagged values
        for period in self.periods:
            for col in self.columns_to_lag:
                if col in df.columns:  # Check if the column exists in the DataFrame
                    lagged_col_name = f'{col}_Lag{period}'
                    lagged_df = pd.DataFrame({lagged_col_name: df[col].shift(period)})
                    lagged_dfs.append(lagged_df)
        
        # If there are lagged DataFrames, concatenate them with the original DataFrame
        if lagged_dfs:
            # Combine all lagged DataFrames with the original DataFrame
            result_df = pd.concat([df] + lagged_dfs, axis=1)
            return result_df
        else:
            return df 