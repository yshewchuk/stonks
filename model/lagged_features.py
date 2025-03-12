import pandas as pd

class LaggedFeatures: # NEW: LaggedFeatures class
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

        for period in self.periods:
            for col in self.columns_to_lag:
                if col in df.columns: # Check if the column exists in the DataFrame
                    lagged_col_name = f'{col}_Lag{period}'
                    df[lagged_col_name] = df[col].shift(period) # Create lagged column

        return df