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

        if min_percent_change is None and max_percent_change is None:
            raise ValueError("At least one of min_percent_change or max_percent_change must be specified.")

        if min_percent_change is not None and max_percent_change is not None:
            if min_percent_change >= max_percent_change:
                raise ValueError("min_percent_change must be less than max_percent_change.")


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

        current_open_prices = df['Open'] # Series of current open prices

        future_percent_changes = []
        future_dates_range = range(self.start_days_future, self.end_days_future + 1)

        for days_future in future_dates_range:
            future_open_prices_shifted = current_open_prices.shift(-days_future) # Shift future open prices
            percent_change = (future_open_prices_shifted - current_open_prices) / current_open_prices
            future_percent_changes.append(percent_change)

        # Stack future percent changes into a DataFrame for vectorized condition checking
        future_percent_changes_df = pd.concat(future_percent_changes, axis=1)
        future_percent_changes_df = future_percent_changes_df.reindex(df.index) # Ensure correct index

        condition_matrix = pd.DataFrame(index=df.index) # Matrix to store boolean conditions
        if self.min_percent_change is not None and self.max_percent_change is not None:
            condition_matrix = (future_percent_changes_df > self.min_percent_change) & (future_percent_changes_df <= self.max_percent_change)
        elif self.min_percent_change is not None:
            condition_matrix = (future_percent_changes_df >= self.min_percent_change)
        elif self.max_percent_change is not None:
            condition_matrix = (future_percent_changes_df <= self.max_percent_change)

        # Calculate probability - vectorized sum over the rows (True counts as 1, False as 0)
        probability_values = condition_matrix.sum(axis=1) / len(future_dates_range)
        probability_values = probability_values.fillna(0) # Fill NaN values (from potential division by zero or no future dates)

        # --- Generate Column Name ---
        col_name_parts = ["PPCProb"] # Percent Price Change Probability
        col_name_parts.append(f"F{self.start_days_future}-{self.end_days_future}D") # Future days range
        if self.min_percent_change is not None:
            col_name_parts.append(f"Min{int(self.min_percent_change*100)}")
        if self.max_percent_change is not None:
            col_name_parts.append(f"Max{int(self.max_percent_change*100)}")

        probability_column_name = "_".join(col_name_parts)

        df[probability_column_name] = probability_values.values # Assign numpy array to avoid index issues

        return df 