import pandas as pd
import numpy

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