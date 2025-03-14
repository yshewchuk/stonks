import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from utils.dataframe import write_dataframes_to_parquet, read_parquet_files_from_directory, truncate_recent_data, create_time_windows
from config import OUTPUT_DIR

class TestDataFrameUtils(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test dataframes
        self.df1 = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        }, index=pd.date_range('2023-01-01', periods=5, freq=None))
        
        self.df2 = pd.DataFrame({
            'X': [100, 200, 300, 400, 500],
            'Y': [1000, 2000, 3000, 4000, 5000]
        }, index=pd.date_range('2023-01-01', periods=5, freq=None))
        
        self.test_dataframes = {
            'test1': self.df1,
            'test2': self.df2
        }
        
        self.config = {
            OUTPUT_DIR: self.test_dir
        }
        
        # Create a larger DataFrame for time window testing
        # Generate 100 days of data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create a multi-level column structure similar to our merged data
        # Level 1: Ticker (AAPL, MSFT)
        # Level 2: Feature (Open, Close, Volume)
        columns = pd.MultiIndex.from_product(
            [['AAPL', 'MSFT'], ['Open', 'Close', 'Volume']],
            names=['Ticker', 'Feature']
        )
        
        # Generate random data
        data = np.random.randn(100, 6) * 10 + 100
        # Set volume to positive integers
        data[:, 2] = np.abs(data[:, 2] * 1000).astype(int)
        data[:, 5] = np.abs(data[:, 5] * 1000).astype(int)
        
        # Create the DataFrame
        self.time_series_df = pd.DataFrame(data, index=dates, columns=columns)
        
        # Introduce some NaN values for testing dropna
        self.time_series_df.iloc[25:27, 0:2] = np.nan  # Set AAPL's Open and Close to NaN for days 25-26
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_write_dataframes_to_parquet(self):
        # Test writing dataframes to parquet
        result = write_dataframes_to_parquet(self.test_dataframes, self.config)
        self.assertTrue(result)
        
        # Check that files were created
        for name in self.test_dataframes.keys():
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, f"{name}.parquet")))
    
    def test_write_dataframes_invalid_input(self):
        # Test with invalid input
        invalid_inputs = [
            ("Not a dict", self.config),
            ({"test": "not a dataframe"}, self.config),
            (self.test_dataframes, "not a dict")
        ]
        
        for invalid_input in invalid_inputs:
            result = write_dataframes_to_parquet(invalid_input[0], invalid_input[1])
            self.assertFalse(result)
    
    def test_read_parquet_files_from_directory(self):
        # First write the dataframes to parquet
        write_dataframes_to_parquet(self.test_dataframes, self.config)
        
        # Test reading parquet files
        result = read_parquet_files_from_directory(self.test_dir)
        
        # Check that we got the correct number of dataframes
        self.assertEqual(len(result), len(self.test_dataframes))
        
        # Check that the dataframes have the correct data
        for name, df in self.test_dataframes.items():
            self.assertTrue(name in result)
            # Check values only, not the index frequency
            pd.testing.assert_frame_equal(result[name], df, check_freq=False)
    
    def test_read_nonexistent_directory(self):
        # Test reading from a nonexistent directory
        nonexistent_dir = os.path.join(self.test_dir, "nonexistent")
        result = read_parquet_files_from_directory(nonexistent_dir)
        self.assertEqual(len(result), 0)
    
    def test_truncate_recent_data(self):
        # Create a test dataframe with 10 rows
        test_df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20)
        }, index=pd.date_range('2023-01-01', periods=10, freq=None))
        
        print(f"Test DataFrame shape: {test_df.shape}")
        print(test_df)
        
        # Test basic truncation
        truncated_df = truncate_recent_data(test_df, rows_to_remove=3)
        self.assertEqual(len(truncated_df), 7)
        
        # The values should be the first 7 rows
        pd.testing.assert_frame_equal(truncated_df, test_df.iloc[:7], check_freq=False)
        
        # Test with min_rows_required
        truncated_df = truncate_recent_data(test_df, rows_to_remove=3, min_rows_required=11)
        self.assertIsNone(truncated_df)
        
        truncated_df = truncate_recent_data(test_df, rows_to_remove=3, min_rows_required=7)
        self.assertEqual(len(truncated_df), 7)
        
        # Test removing all rows
        print("Testing with rows_to_remove=10")
        truncated_df = truncate_recent_data(test_df, rows_to_remove=10)
        print(f"Result: {truncated_df}")
        self.assertIsNone(truncated_df)
        
        # Test removing more rows than available
        truncated_df = truncate_recent_data(test_df, rows_to_remove=15)
        self.assertIsNone(truncated_df)
    
    def test_create_time_windows(self):
        """Test the create_time_windows function with various parameters"""
        
        # Test basic functionality with non-overlapping windows
        windows = create_time_windows(self.time_series_df, window_size=10)
        
        # Should create 9 windows of 10 days each (one window is dropped due to NaN values)
        self.assertEqual(len(windows), 9)
        
        # Each window should have 10 days
        for window in windows:
            self.assertEqual(len(window), 10)
        
        # Test window naming
        first_window = windows[0]
        expected_start = self.time_series_df.index[0].strftime('%Y-%m-%d')
        expected_end = self.time_series_df.index[9].strftime('%Y-%m-%d')
        expected_name = f"{expected_start}_to_{expected_end}"
        self.assertEqual(first_window.name, expected_name)
        
        # Test with overlapping windows (step_size < window_size)
        windows = create_time_windows(self.time_series_df, window_size=10, step_size=5)
        
        # Should create fewer windows due to NaN values
        # Calculate expected windows: without NaNs would be (100-10)/5 + 1 = 19
        # But windows containing indices 25-26 will be dropped
        self.assertGreater(len(windows), 10)  # Just verify we have multiple windows
        
        # Check overlapping windows have correct indices
        self.assertEqual(windows[0].index[0], self.time_series_df.index[0])
        self.assertEqual(windows[1].index[0], self.time_series_df.index[5])
    
    def test_create_time_windows_with_nan(self):
        """Test the create_time_windows function's handling of NaN values"""
        
        # Test with dropna=True (default)
        windows = create_time_windows(self.time_series_df, window_size=10)
        
        # Windows containing days 25-26 should be dropped
        # These would be windows starting at indices 20-26
        for window in windows:
            # Check if this window would include the NaN values
            start_date = window.index[0]
            end_date = window.index[-1]
            contains_nan_period = (
                start_date <= self.time_series_df.index[26] and 
                end_date >= self.time_series_df.index[25]
            )
            
            if contains_nan_period:
                self.fail(f"Window {window.name} contains NaN values but was not dropped")
        
        # Test with dropna=False
        windows = create_time_windows(self.time_series_df, window_size=10, dropna=False)
        
        # Should create 10 windows of 10 days each (100 days / 10 days per window)
        self.assertEqual(len(windows), 10)
        
        # Windows containing NaN values should exist
        nan_dates = self.time_series_df.index[25:27]
        windows_with_nan = []
        
        for window in windows:
            if any(date in window.index for date in nan_dates):
                # This window contains dates with NaN values
                if window.isna().any().any():
                    windows_with_nan.append(window)
        
        self.assertGreater(len(windows_with_nan), 0, 
                          "No windows with NaN values found when dropna=False")
    
    def test_create_time_windows_invalid_input(self):
        """Test the create_time_windows function with invalid inputs"""
        
        # Test with non-DataFrame input
        result = create_time_windows("not a dataframe", window_size=10)
        self.assertEqual(result, [])
        
        # Test with non-datetime index
        df_without_datetime = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })
        result = create_time_windows(df_without_datetime, window_size=10)
        self.assertEqual(result, [])
        
        # Test with DataFrame too small for window
        small_df = pd.DataFrame({
            'A': range(5),
            'B': range(5, 10)
        }, index=pd.date_range('2023-01-01', periods=5))
        result = create_time_windows(small_df, window_size=10)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main() 