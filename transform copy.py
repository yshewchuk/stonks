"""
ETL - Transform Script: Transforms raw stock data, calculates indicators,
                       and creates training and evaluation datasets.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Configuration
RAW_DATA_DIR = 'raw_stock_data'  # Directory where raw data CSVs are stored
TRANSFORMED_DATA_DIR = 'transformed_stock_data'  # Directory to save transformed data
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
FEATURE_COLUMNS_STOCK_DATA = ['Open', 'High', 'Low', 'Close', 'Volume',  # Base stock data features
                   'MA7', 'MA30', 'MA60',  # Moving Averages
                   'High7D', 'Low7D', 'High30D', 'Low30D', 'High60D', 'Low60D'] # High/Low indicators
RECENT_YEARS_EVAL = 1  # Number of recent years to use for evaluation data

# Create transformed data directory if it doesn't exist
os.makedirs(TRANSFORMED_DATA_DIR, exist_ok=True)

def load_stock_data(ticker, data_dir=RAW_DATA_DIR, expected_columns=FEATURE_COLUMNS_STOCK_DATA):
    """Loads stock data and fixes datetime parsing with UTC conversion."""
    file_path = os.path.join(data_dir, f'{ticker}.csv')
    df = pd.read_csv(file_path, index_col='Date') # Read Date column as string initially
    df.dropna(inplace=True)

    # --- Debugging Prints ---
    print(f"\n--- Debugging for ticker: {ticker} ---")
    print(f"Initial Type of df.index (before conversion): {type(df.index)}")

    # *** User's fix - Convert to DatetimeIndex, handle timezones by converting to UTC then making timezone-naive ***
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    # Ensure essential columns are present and select them along with feature columns
    columns_to_select = expected_columns # Use expected_columns directly - already contains base columns
    valid_columns = [col for col in columns_to_select if col in df.columns]
    df = df[valid_columns].copy()

    return df


def calculate_technical_indicators(df, moving_averages=[7, 30, 60], hl_days=[7, 30, 60]):
    """Calculates moving averages and high/low indicators, with detailed debugging."""

    print("\n--- Debugging inside calculate_technical_indicators ---")
    print(f"Shape of DataFrame at start of calculate_technical_indicators: {df.shape}")
    print(f"Columns of DataFrame at start of calculate_technical_indicators: {df.columns}")
    print(f"Type of df['Close'] column: {type(df['Close'])}") # Check type of 'Close' column

    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        print("DataFrame columns are MultiIndex. Flattening...")
        df.columns = df.columns.get_level_values(1)
        print(f"Columns after flattening: {df.columns}") # Print columns after flattening
    else:
        print("DataFrame columns are NOT MultiIndex.")


    for ma in moving_averages:
        ma_column_name = f'MA{ma}'
        print(f"\nCalculating MA{ma}...")
        rolling_mean = df['Close'].rolling(window=ma, min_periods=1).mean() # Calculate rolling mean
        print(f"Type of rolling_mean (MA{ma}): {type(rolling_mean)}") # Check type of rolling_mean
        print(f"Shape of rolling_mean (MA{ma}): {rolling_mean.shape}") # Check shape of rolling_mean

        try:
            df[ma_column_name] = rolling_mean # Assign rolling mean to new column
            print(f"✅ Successfully assigned MA{ma} column.")
        except ValueError as ve:
            print(f"❌ ValueError during assignment of MA{ma} column: {ve}") # Print specific ValueError
            print(f"DataFrame info just before error for MA{ma}:\n")
            df.info() # Print DataFrame info at point of error
            raise ve # Re-raise the error


    for hl in hl_days:
        # Similar debugging for High/Low indicators can be added if needed, once MA is fixed.
        df[f'High{hl}D'] = df['High'].rolling(window=hl, min_periods=1).max()
        df[f'Low{hl}D'] = df['Low'].rolling(window=hl, min_periods=1).min()
    return df


def prepare_transformed_data(all_ticker_data, tickers, feature_columns_stock_data, recent_years_eval=RECENT_YEARS_EVAL):
    """
    Transforms data, aligns dates, scales features, and splits into train/eval sets.
    """
    print("Aligning dates across tickers...")
    min_end_date = min(df.index[-1] for df in all_ticker_data.values())
    max_start_date = max(df.index[0] for df in all_ticker_data.values())
    common_dates = None

    for ticker, df in all_ticker_data.items():
        valid_dates = df[(df.index >= max_start_date) & (df.index <= min_end_date)].index
        if common_dates is None:
            common_dates = valid_dates
        else:
            common_dates = common_dates.intersection(valid_dates)

    if common_dates is None or len(common_dates) == 0:
        print("⚠️ No common dates found across tickers after alignment.")
        return None, None, None

    print(f"Common date range established: {common_dates.min().date()} to {common_dates.max().date()}")


    transformed_data = {}
    feature_scaler = StandardScaler() # Scaler for numerical features

    for ticker in tickers:
        df = all_ticker_data[ticker].loc[common_dates].copy() # Use common dates and create copy

        print(f"Scaling features for {ticker}...")
        scaled_features = feature_scaler.fit_transform(df[feature_columns_stock_data]) # Scale stock data features
        scaled_df = pd.DataFrame(scaled_features, index=df.index, columns=feature_columns_stock_data) # Create DataFrame from scaled data
        transformed_data[ticker] = scaled_df # Store scaled DataFrame


    print("Splitting data into training and evaluation sets...")
    eval_end_date = common_dates.max() # Evaluation data up to the latest common date
    eval_start_date = eval_end_date - timedelta(days=365 * recent_years_eval) # Evaluation data is the most recent year

    train_data = {}
    eval_data = {}

    for ticker, df in transformed_data.items():
        train_df = df[df.index < eval_start_date].copy() # Training data is before evaluation start
        eval_df = df[ (df.index >= eval_start_date) & (df.index <= eval_end_date) ].copy() # Evaluation data is within eval range

        train_data[ticker] = train_df
        eval_data[ticker] = eval_df

        print(f"  {ticker}: Training data from {train_df.index.min().date()} to {train_df.index.max().date()}, Evaluation data from {eval_df.index.min().date()} to {eval_df.index.max().date()}")


    return train_data, eval_data, feature_scaler  # Return train data, eval data, and scaler


def save_transformed_data_to_csv(train_data, eval_data, transformed_data_dir=TRANSFORMED_DATA_DIR):
    """Saves transformed training/eval datasets to CSVs with proper MultiIndex header."""
    print(f"Saving transformed data to: {transformed_data_dir}...")

    # --- Debugging - Inspect DataFrames before saving (optional now, can comment out if you want clean output) ---
    # print("\n--- Debugging: train_combined_df before saving ---")
    # train_combined_df = pd.concat(train_data, axis=1, keys=train_data.keys())
    # print("train_combined_df.head():\n", train_combined_df.head())
    # print("\ntrain_combined_df.columns (MultiIndex?):\n", train_combined_df.columns)

    # print("\n--- Debugging: eval_combined_df before saving ---")
    # eval_combined_df = pd.concat(eval_data, axis=1, keys=eval_data.keys())
    # print("eval_combined_df.head():\n", eval_combined_df.head())
    # print("\neval_combined_df.columns (MultiIndex?):\n", eval_combined_df.columns)
    # print("\n--- End Debugging ---\n")


    # Save training data - explicitly set header=True and index_label for MultiIndex
    train_combined_df = pd.concat(train_data, axis=1, keys=train_data.keys()) # Concatenate again (if you commented out debugging section)
    train_file_path = os.path.join(transformed_data_dir, 'train_data.csv')
    train_combined_df.to_csv(train_file_path, header=True, index_label=['Date']) # Added index_label for MultiIndex CSV
    print(f"  Transformed training data saved to: {train_file_path}")

    # Save evaluation data - explicitly set header=True and index_label for MultiIndex
    eval_combined_df = pd.concat(eval_data, axis=1, keys=eval_data.keys()) # Concatenate again (if you commented out debugging section)
    eval_file_path = os.path.join(transformed_data_dir, 'eval_data.csv')
    eval_combined_df.to_csv(eval_file_path, header=True, index_label=['Date']) # Added index_label for MultiIndex CSV
    print(f"  Transformed evaluation data saved to: {eval_file_path}")


if __name__ == '__main__':
    print("🚀 Starting Stock Data Transformation...")

    print("Loading raw stock data...")
    raw_data = {}
    for ticker in TICKERS:
        raw_data[ticker] = load_stock_data(ticker, data_dir=RAW_DATA_DIR) # Load raw data from CSVs

    print("Calculating technical indicators...")
    indicator_data = {}
    for ticker, df in raw_data.items():
        indicator_data[ticker] = calculate_technical_indicators(df) # Calculate MAs and HLs

    print("Preparing transformed data (aligning, scaling, splitting)...")
    train_dataset, eval_dataset, feature_scaler = prepare_transformed_data(indicator_data, TICKERS, FEATURE_COLUMNS_STOCK_DATA, RECENT_YEARS_EVAL)

    if train_dataset and eval_dataset: # Proceed only if data preparation was successful
        print("Saving transformed data to CSV...")
        save_transformed_data_to_csv(train_dataset, eval_dataset)
        print("\n🎉 Stock Data Transformation Complete!")
        print(f"Transformed data saved to: {TRANSFORMED_DATA_DIR}")
    else:
        print("❌ Data transformation failed or insufficient data. Check warnings during transformation.")