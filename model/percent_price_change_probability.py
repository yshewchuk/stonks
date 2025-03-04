import pandas as pd
import numpy as np

from utils.dataframe import verify_dataframe_structure

REQUIRED_COLUMNS = {
    'Open': float
}

class PercentPriceChangeProbability:
    """
    Calculates and adds a column representing the probability of a percentage price change
    in the 'Open' price within a specified future date range, using vectorized operations.

    Verifies input DataFrame structure (Date index, Open column) and handles
    calculations for time-series data.
    """

    def __init__(self, start_days_future, end_days_future, min_percent_change=None, max_percent_change=None):
        """
        Initializes PercentPriceChangeProbability calculator.

        Args:
            start_days_future (int): The starting day of the future window (e.g., 1 for tomorrow). Must be positive.
            end_days_future (int): The ending day of the future window (e.g., 3 for up to 3 days from now). Must be >= start_days_future.
            min_percent_change (float, optional): Minimum percentage change to track (e.g., -0.07 for -7%). Defaults to None.
                                                  If None, it means percentage change <= max_percent_change.
            max_percent_change (float, optional): Maximum percentage change to track (e.g., 0.07 for 7%). Defaults to None.
                                                  If None, it means percentage change >= min_percent_change.

        Raises:
            ValueError: If input arguments are invalid.
        """
        if not isinstance(start_days_future, int) or start_days_future <= 0:
            raise ValueError("start_days_future must be a positive integer.")
        if not isinstance(end_days_future, int) or end_days_future < start_days_future:
            raise ValueError("end_days_future must be an integer and >= start_days_future.")

        self.start_days_future = start_days_future
        self.end_days_future = end_days_future
        self.min_percent_change = min_percent_change
        self.max_percent_change = max_percent_change

        self.__window_size = self.end_days_future + 1
        self.__sample_count = 1 + self.end_days_future - self.start_days_future

        # --- Generate Column Name ---
        col_name_parts = ["PPCProb"] # Percent Price Change Probability
        col_name_parts.append(f"F{self.start_days_future}-{self.end_days_future}D") # Future days range
        if self.min_percent_change is not None:
            col_name_parts.append(f"Min{int(self.min_percent_change)}")
        if self.max_percent_change is not None:
            col_name_parts.append(f"Max{int(self.max_percent_change)}")

        self.__probability_column_name = "_".join(col_name_parts)

        if min_percent_change is None and max_percent_change is None:
            raise ValueError("At least one of min_percent_change or max_percent_change must be specified.")

        if min_percent_change is not None and max_percent_change is not None:
            if min_percent_change >= max_percent_change:
                raise ValueError("min_percent_change must be less than max_percent_change.")

    def __check_percent_change(self, series):
        div = series.iloc[0]
        
        curr = 0

        for i in range(self.start_days_future, self.end_days_future + 1):
            percent_change = 100 * series.iloc[i] / div - 100
            less_than_max = self.max_percent_change is None or percent_change <= self.max_percent_change
            greater_than_min = self.min_percent_change is None or percent_change >= self.min_percent_change
            if less_than_max and greater_than_min:
                curr += 1 / self.__sample_count

        return curr

    def extend(self, df):
        """
        Extends the input DataFrame with a column representing the probability of
        a percentage price change in 'Open' price within the specified future date range,
        using vectorized operations for efficiency.

        Verifies:
            - DataFrame has 'Date' index of type pd.DatetimeIndex.
            - DataFrame has an 'Open' column that is numeric.

        Args:
            df (pd.DataFrame): Input DataFrame with 'Date' index and 'Open' column.

        Returns:
            pd.DataFrame: DataFrame with an added probability column. The column name
                          will be dynamically generated based on the configuration.

        Raises:
            ValueError: If input DataFrame structure is invalid.
        """
        # --- Input Verification ---
        if not verify_dataframe_structure(df, REQUIRED_COLUMNS, ignore_extra_columns=True, expected_index_name='Date', expected_index_dtype=np.datetime64):
            print(f"❌ Missing required column or index")
            raise ValueError("Invalid input DataFrame structure.")

        # --- Extend DataFrame ---
        df[self.__probability_column_name] = df['Open'].rolling(window=self.__window_size).agg(self.__check_percent_change).shift(1 - self.__window_size) # Add Percent Price Change Probability column

        return df