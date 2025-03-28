import pandas as pd
import numpy
import os
import concurrent.futures
import multiprocessing
from config import OUTPUT_DIR, MAX_WORKERS
from utils.logger import log_info, log_success, log_error, log_warning, log_progress

def print_dataframe_debugging_info(df, name="DataFrame"):
    """
    Prints helpful debugging information about a Pandas DataFrame.

    Outputs:
        - Shape (rows, columns)
        - Column Names
        - Data type of each column
        - A small sample of data rows (first 5 rows if available)

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        name (str, optional):  A name to identify the DataFrame in the output. Defaults to "DataFrame".
    """
    print(f"\n--- Debugging Info for: {name} ---")

    if not isinstance(df, pd.DataFrame):
        log_warning(f"Input is not a Pandas DataFrame. Input type: {type(df)}")
        return

    print(f"Shape: {df.shape}")
    print(f"Type of df.index: {type(df.index)}")
    print("\nIndex Name:", df.index.name)
    print("\nColumn Names:", list(df.columns))

    print("\nData Types of Columns:")
    print(df.dtypes)

    print("\nSample Data (First 5 Rows):")
    if len(df) > 0:
        print(df.head())
    else:
        print("DataFrame is empty.")

    print("--- End Debugging Info ---\n")
    
    
def verify_dataframe_structure(df, expected_dtypes, ignore_extra_columns=False, expected_index_name=None, expected_index_dtype=None):
    """
    Verifies if a DataFrame's structure (columns, data types, index name, index type) matches expectations
    based on a dictionary of expected data types and optional index properties.

    Args:
        df (pd.DataFrame): The DataFrame to verify.
        expected_dtypes (dict): A dictionary mapping column names (strings) to expected data types
                                 (e.g., str, int, float, pd.Timestamp, or dtype objects).
        ignore_extra_columns (boolean, optional): Whether to skip columns that aren't expected
        expected_index_name (str, optional): Expected name of the DataFrame's index. Defaults to None (index name check skipped if None).
        expected_index_dtype (dtype, optional): Expected data type of the DataFrame's index (e.g., pd.DatetimeIndex, int, str, or dtype objects like np.dtype('datetime64[ns]')).
                                               Defaults to None (index type check skipped if None).
        df_name (str, optional):  A name to identify the DataFrame in the output messages. Defaults to "DataFrame to Verify".

    Returns:
        bool: True if the DataFrame structure matches expectations, False otherwise.
    """
    is_valid_structure = True  # Assume valid structure initially
    expected_columns = list(expected_dtypes.keys()) # Derive expected columns from dtypes dict keys

    if not isinstance(df, pd.DataFrame):
        log_error(f"Input is not a Pandas DataFrame. Input type: {type(df)}")
        return False

    # 1. Column Name Verification (same as before)
    df_columns = list(df.columns)
    expected_columns_set = set(expected_columns)
    df_columns_set = set(df_columns)

    missing_columns = list(expected_columns_set - df_columns_set)
    extra_columns = list(df_columns_set - df_columns_set)

    if missing_columns:
        log_error(f"DataFrame is missing expected columns: {missing_columns}")
        return False
    
    if extra_columns and not ignore_extra_columns:
        log_error(f"DataFrame has extra columns not in expected list: {extra_columns}")
        return False

    # 2. Data Type Verification (same as before)
    dtype_mismatches = {}
    for column_name, expected_dtype in expected_dtypes.items():
        actual_dtype = df[column_name].dtype
        if actual_dtype != expected_dtype:
            dtype_mismatches[column_name] = {'expected': expected_dtype, 'actual': actual_dtype}

    if dtype_mismatches:
        log_error("Data type mismatches found in columns:")
        for col, mismatch_info in dtype_mismatches.items():
            print(f"  - Column '{col}': Expected type '{mismatch_info['expected']}', Actual type '{mismatch_info['actual']}'")
        return False

    # 3. Index Name Verification (New)
    if expected_index_name is not None:
        actual_index_name = df.index.name
        if actual_index_name != expected_index_name:
            log_error(f"Index name mismatch. Expected: '{expected_index_name}', Actual: '{actual_index_name}'")
            return False

    # 4. Index Data Type Verification (New)
    if expected_index_dtype is not None:
        actual_index_dtype = df.index.dtype

        if actual_index_dtype != expected_index_dtype and not (pd.api.types.is_datetime64_any_dtype(actual_index_dtype) and pd.api.types.is_datetime64_any_dtype(expected_index_dtype)):
            log_error(f"Index data type mismatch. Expected: '{expected_index_dtype}', Actual: '{actual_index_dtype}'")
            return False

    return True

def _write_parquet_file(args):
    """
    Helper function to write a single DataFrame to a parquet file.
    Used by ProcessPoolExecutor for parallel file writing.
    
    Args:
        args (tuple): Tuple containing (name, df, filepath)
        
    Returns:
        tuple: (name, success, error_message)
    """
    name, df, filepath = args
    try:
        df.to_parquet(filepath, index=True, compression='snappy')
        return (name, True, None)
    except Exception as e:
        return (name, False, str(e))

def write_dataframes_to_parquet(dataframes_dict, config, max_workers=None):
    """
    Writes a dictionary of dataframes to parquet files in the directory
    specified by the OUTPUT_DIR key in the config dictionary.
    Uses multiprocessing for improved performance.
    
    Args:
        dataframes_dict (dict): Dictionary where keys are dataframe names and values are pandas DataFrames
        config (dict): Configuration dictionary containing at least an OUTPUT_DIR key
        max_workers (int, optional): Maximum number of worker processes. If None, uses config[MAX_WORKERS]
        
    Returns:
        bool: True if all dataframes were saved successfully, False otherwise
    """
    if not isinstance(dataframes_dict, dict):
        log_error("dataframes_dict must be a dictionary")
        return False
        
    if not all(isinstance(df, pd.DataFrame) for df in dataframes_dict.values()):
        log_error("All values in dataframes_dict must be pandas DataFrames")
        return False
        
    if OUTPUT_DIR not in config:
        log_error(f"config must contain an {OUTPUT_DIR} key")
        return False
        
    try:
        # Ensure the output directory exists
        output_dir = config[OUTPUT_DIR]
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare arguments for process pool
        tasks = []
        for name, df in dataframes_dict.items():
            filename = f"{name}.parquet"
            filepath = os.path.join(output_dir, filename)
            tasks.append((name, df, filepath))
        
        # Set up progress tracking
        total_files = len(tasks)
        successful_writes = 0
        
        # Determine process pool size - use max_workers if provided, otherwise get from config
        if max_workers is None:
            if MAX_WORKERS in config:
                max_workers = config[MAX_WORKERS]
            else:
                max_workers = max(1, min(int(multiprocessing.cpu_count() * 0.75), 8))
        
        log_info(f"Using {max_workers} processes for writing parquet files")
        
        # Use ProcessPoolExecutor for parallel writing (CPU-bound due to compression)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_write_parquet_file, task): task[0] for task in tasks}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                name, success, error_message = future.result()
                
                if success:
                    successful_writes += 1
                    # Log progress at appropriate intervals
                    log_progress(i + 1, total_files, "Files written", frequency=5)
                else:
                    log_error(f"Error writing '{name}' to parquet: {error_message}")
        
        if successful_writes == total_files:
            log_success(f"Successfully wrote all {total_files} dataframes to parquet files")
            return True
        else:
            log_warning(f"Only {successful_writes} out of {total_files} dataframes were successfully written")
            return successful_writes > 0
        
    except Exception as e:
        log_error(f"Error writing dataframes to parquet: {e}")
        return False

def _read_parquet_file(args):
    """
    Helper function to read a single parquet file.
    Used by ProcessPoolExecutor for parallel file reading.
    
    Args:
        args (tuple): Tuple containing (file, directory)
        
    Returns:
        tuple: (name, dataframe, error_message)
    """
    file, directory = args
    try:
        name = os.path.splitext(file)[0]
        filepath = os.path.join(directory, file)
        df = pd.read_parquet(filepath)
        return (name, df, None)
    except Exception as e:
        return (os.path.splitext(file)[0], None, str(e))

def read_parquet_files_from_directory(directory, max_workers=None):
    """
    Reads all parquet files from a directory and returns them as a dictionary
    where keys are filenames (without extension) and values are pandas DataFrames.
    Uses multiprocessing for improved performance.
    
    Args:
        directory (str): Path to the directory containing parquet files
        max_workers (int, optional): Maximum number of worker processes. If None, uses CONFIG[MAX_WORKERS]
        
    Returns:
        dict: Dictionary of DataFrames with filenames as keys
              Returns empty dict if directory doesn't exist or contains no parquet files
    """
    dataframes = {}
    
    if not os.path.exists(directory):
        log_error(f"Directory not found: {directory}")
        return dataframes
        
    try:
        # Get all parquet files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
        
        if not files:
            log_warning(f"No parquet files found in {directory}")
            return dataframes

        # Determine process pool size - use max_workers if provided, otherwise default
        # Note: We can't directly access CONFIG here since it's not passed in,
        # but scripts importing this function can still get max_workers from CONFIG
        if max_workers is None:
            max_workers = max(1, min(int(multiprocessing.cpu_count() * 0.75), 8))
            
        log_info(f"Using {max_workers} processes for reading parquet files")
        
        # Use ProcessPoolExecutor for parallel reading (CPU-bound due to decompression)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(file, directory) for file in files]
            futures = {executor.submit(_read_parquet_file, task): task[0] for task in tasks}
            
            # Process results as they complete
            current = 0
            for future in concurrent.futures.as_completed(futures):
                name, df, error_message = future.result()
                
                if df is not None:
                    current += 1
                    dataframes[name] = df
                    log_progress(current, len(files), "Files loaded")
                else:
                    log_error(f"Error loading {futures[future]}: {error_message}")
        
        return dataframes
        
    except Exception as e:
        log_error(f"Error reading parquet files from directory: {e}")
        return dataframes

def extract_data_range(df, num_rows, extract_recent=True, min_rows_required=None):
    """
    Extracts or truncates a range of rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to process, assumed to be sorted by index (date)
        num_rows (int): Number of rows to extract or truncate
        extract_recent (bool): If True, extracts the last num_rows rows (evaluation data)
                              If False, truncates the last num_rows rows (training data)
        min_rows_required (int, optional): Minimum rows required in df, defaults to num_rows+1
        
    Returns:
        pd.DataFrame: Processed DataFrame, or None if the DataFrame doesn't have enough rows
    """
    if not isinstance(df, pd.DataFrame):
        print("❌ Error: Input must be a pandas DataFrame")
        return None
        
    # Set default for min_rows_required if not provided
    if min_rows_required is None:
        min_rows_required = num_rows + 1
    
    # Check if DataFrame has enough rows
    if len(df) < min_rows_required:
        print(f"⚠️ Warning: DataFrame has only {len(df)} rows, {min_rows_required} required")
        return None
    
    # Sort DataFrame by index just to be sure
    df = df.sort_index()
    
    if extract_recent:
        # Extract the last num_rows rows (for evaluation data)
        processed_df = df.iloc[-num_rows:].copy()
        print(f"ℹ️ Extracted {len(processed_df)} recent rows from DataFrame with {len(df)} rows")
    else:
        # Remove the last num_rows rows (for training data)
        processed_df = df.iloc[:-num_rows].copy()
        print(f"ℹ️ Truncated {num_rows} recent rows from DataFrame with {len(df)} rows")
    
    return processed_df

def create_time_windows(df, window_size, step_size=None, dropna=True):
    """
    Splits a time series DataFrame into fixed-length time windows.
    
    Args:
        df (pd.DataFrame): DataFrame with a datetime index
        window_size (int): Number of days in each window
        step_size (int, optional): Number of days to slide forward for each new window. 
                                  If None, windows don't overlap (step_size = window_size).
        dropna (bool, optional): Whether to drop windows containing NaN values. Default is True.
        
    Returns:
        list: List of DataFrames representing the time windows
    """
    if not isinstance(df, pd.DataFrame):
        print("❌ Error: Input must be a pandas DataFrame")
        return []
    
    # Ensure index is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print("❌ Error: DataFrame index must be datetime type")
        return []
    
    # Sort by date to ensure correct window creation
    df = df.sort_index()
    
    # If step_size not provided, set it equal to window_size (non-overlapping windows)
    if step_size is None:
        step_size = window_size
    
    # Calculate minimum size needed for creating at least one window
    if len(df) < window_size:
        print(f"❌ Error: DataFrame has {len(df)} rows, need at least {window_size} rows for a window")
        return []
    
    windows = []
    start_idx = 0
    
    # Create windows until we reach the end of the DataFrame
    while start_idx + window_size <= len(df):
        # Extract the window
        window = df.iloc[start_idx:start_idx + window_size].copy()
        
        # Check for NaN values if dropna is True
        if dropna and window.isna().any().any():
            print(f"⚠️ Dropping window at index {start_idx} due to NaN values")
        else:
            # Get the date range for naming the window
            start_date = window.index[0].strftime('%Y-%m-%d')
            end_date = window.index[-1].strftime('%Y-%m-%d')
            
            # Name the window by its date range
            window.name = f"{start_date}_to_{end_date}"
            
            # Add the window to the list
            windows.append(window)
            
            if start_idx % 100 == 0:
                print(f"✅ Created window {window.name}: {len(window)} days")
        
        # Move forward by step_size
        start_idx += step_size
    
    print(f"✅ Created {len(windows)} time windows of {window_size} days each")
    return windows