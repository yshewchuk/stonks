import pandas as pd
import numpy
import os
from config import OUTPUT_DIR

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
        print(f"⚠️ Warning: Input is not a Pandas DataFrame. Input type: {type(df)}")
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
        print(f"❌ Error: Input is not a Pandas DataFrame. Input type: {type(df)}")
        return False

    # 1. Column Name Verification (same as before)
    df_columns = list(df.columns)
    expected_columns_set = set(expected_columns)
    df_columns_set = set(df_columns)

    missing_columns = list(expected_columns_set - df_columns_set)
    extra_columns = list(df_columns_set - df_columns_set)

    if missing_columns:
        print(f"❌ Error: DataFrame is missing expected columns: {missing_columns}")
        return False
    
    if extra_columns and not ignore_extra_columns:
        print(f"❌ Error: DataFrame has extra columns not in expected list: {extra_columns}")
        return False

    # 2. Data Type Verification (same as before)
    dtype_mismatches = {}
    for column_name, expected_dtype in expected_dtypes.items():
        actual_dtype = df[column_name].dtype
        if actual_dtype != expected_dtype:
            dtype_mismatches[column_name] = {'expected': expected_dtype, 'actual': actual_dtype}

    if dtype_mismatches:
        print("❌ Error: Data type mismatches found in columns:")
        for col, mismatch_info in dtype_mismatches.items():
            print(f"  - Column '{col}': Expected type '{mismatch_info['expected']}', Actual type '{mismatch_info['actual']}'")
        return False

    # 3. Index Name Verification (New)
    if expected_index_name is not None:
        actual_index_name = df.index.name
        if actual_index_name != expected_index_name:
            print(f"❌ Error: Index name mismatch. Expected: '{expected_index_name}', Actual: '{actual_index_name}'")
            return False

    # 4. Index Data Type Verification (New)
    if expected_index_dtype is not None:
        actual_index_dtype = df.index.dtype

        if actual_index_dtype != expected_index_dtype and not (pd.api.types.is_datetime64_any_dtype(actual_index_dtype) and pd.api.types.is_datetime64_any_dtype(expected_index_dtype)):
            print(f"❌ Error: Index data type mismatch. Expected: '{expected_index_dtype}', Actual: '{actual_index_dtype}'")
            return False

    return True

def write_dataframes_to_parquet(dataframes_dict, config):
    """
    Writes a dictionary of dataframes to parquet files in the directory
    specified by the OUTPUT_DIR key in the config dictionary.
    
    Args:
        dataframes_dict (dict): Dictionary where keys are dataframe names and values are pandas DataFrames
        config (dict): Configuration dictionary containing at least an OUTPUT_DIR key
        
    Returns:
        bool: True if all dataframes were saved successfully, False otherwise
    """
    if not isinstance(dataframes_dict, dict):
        print("❌ Error: dataframes_dict must be a dictionary")
        return False
        
    if not all(isinstance(df, pd.DataFrame) for df in dataframes_dict.values()):
        print("❌ Error: All values in dataframes_dict must be pandas DataFrames")
        return False
        
    if OUTPUT_DIR not in config:
        print(f"❌ Error: config must contain an {OUTPUT_DIR} key")
        return False
        
    try:
        # Ensure the output directory exists
        output_dir = config[OUTPUT_DIR]
        os.makedirs(output_dir, exist_ok=True)
        
        # Write each dataframe to a parquet file
        for name, df in dataframes_dict.items():
            # Create filename for parquet
            filename = f"{name}.parquet"
            filepath = os.path.join(output_dir, filename)
            
            # Write dataframe to parquet file
            df.to_parquet(filepath, index=True)
            print(f"✅ DataFrame '{name}' saved to {filepath}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error writing dataframes to parquet: {e}")
        return False

def read_parquet_files_from_directory(directory):
    """
    Reads all parquet files from a directory and returns them as a dictionary
    where keys are filenames (without extension) and values are pandas DataFrames.
    
    Args:
        directory (str): Path to the directory containing parquet files
        
    Returns:
        dict: Dictionary of DataFrames with filenames as keys
              Returns empty dict if directory doesn't exist or contains no parquet files
    """
    dataframes = {}
    
    if not os.path.exists(directory):
        print(f"❌ Error: Directory not found: {directory}")
        return dataframes
        
    try:
        # Get all parquet files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
        
        if not files:
            print(f"⚠️ Warning: No parquet files found in {directory}")
            return dataframes
            
        # Read each parquet file into a DataFrame
        for file in files:
            try:
                # Extract filename without extension to use as key
                name = os.path.splitext(file)[0]
                filepath = os.path.join(directory, file)
                
                # Read the parquet file
                df = pd.read_parquet(filepath)
                dataframes[name] = df
                print(f"✅ Loaded DataFrame '{name}' from {filepath}: {len(df)} rows")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        return dataframes
        
    except Exception as e:
        print(f"❌ Error reading parquet files from directory: {e}")
        return dataframes

def truncate_recent_data(df, rows_to_remove, min_rows_required=None):
    """
    Truncates the most recent rows from a DataFrame, optionally checking
    if the DataFrame has enough rows to perform the truncation.
    
    Args:
        df (pd.DataFrame): DataFrame to truncate, assumed to be sorted by index (date)
        rows_to_remove (int): Number of most recent rows to remove
        min_rows_required (int, optional): Minimum rows required, defaults to rows_to_remove+1
        
    Returns:
        pd.DataFrame: Truncated DataFrame, or None if the DataFrame doesn't have enough rows
    """
    if not isinstance(df, pd.DataFrame):
        print("❌ Error: Input must be a pandas DataFrame")
        return None
        
    # Set default for min_rows_required if not provided
    if min_rows_required is None:
        min_rows_required = rows_to_remove + 1
    
    print(f"DEBUG: df length: {len(df)}, rows_to_remove: {rows_to_remove}")
    
    # Check if DataFrame has enough rows
    if len(df) <= rows_to_remove:
        print(f"⚠️ Warning: DataFrame has only {len(df)} rows, more than or equal to {rows_to_remove} to remove")
        return None
    
    if len(df) < min_rows_required:
        print(f"⚠️ Warning: DataFrame has only {len(df)} rows, {min_rows_required} required")
        return None
    
    # Sort DataFrame by index just to be sure
    df = df.sort_index()
    
    # Remove the last 'rows_to_remove' rows
    truncated_df = df.iloc[:-rows_to_remove]
    
    return truncated_df