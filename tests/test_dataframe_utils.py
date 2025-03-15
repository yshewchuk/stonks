#!/usr/bin/env python3
"""
Unit tests for the dataframe utility functions in utils/dataframe.py.
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
import shutil
import concurrent.futures
from datetime import datetime, timedelta

from config import OUTPUT_DIR, MAX_WORKERS
from utils.dataframe import (
    extract_data_range,
    read_parquet_files_from_directory,
    write_dataframes_to_parquet
)

class TestExtractDataRange(unittest.TestCase):
    """Tests for the extract_data_range function."""

    def setUp(self):
        # Create sample DataFrame with date index
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = {
            'Open': np.random.rand(100) * 100,
            'Close': np.random.rand(100) * 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }
        self.sample_df = pd.DataFrame(data, index=dates)
        self.sample_df.index.name = 'Date'

    def test_extract_recent_data(self):
        """Test extracting the most recent data."""
        # Extract the most recent 20 rows
        extracted_df = extract_data_range(self.sample_df, num_rows=20, extract_recent=True)
        
        # Check that we got 20 rows
        self.assertEqual(len(extracted_df), 20)
        
        # Check that we got the last 20 rows
        pd.testing.assert_index_equal(extracted_df.index, self.sample_df.index[-20:])
        
        # Check that the data values match
        pd.testing.assert_frame_equal(extracted_df, self.sample_df.iloc[-20:])

    def test_truncate_recent_data(self):
        """Test truncating the most recent data."""
        # Truncate the most recent 20 rows
        truncated_df = extract_data_range(self.sample_df, num_rows=20, extract_recent=False)
        
        # Check that we got 80 rows (100 - 20)
        self.assertEqual(len(truncated_df), 80)
        
        # Check that we got the first 80 rows
        pd.testing.assert_index_equal(truncated_df.index, self.sample_df.index[:-20])
        
        # Check that the data values match
        pd.testing.assert_frame_equal(truncated_df, self.sample_df.iloc[:-20])

    def test_insufficient_rows(self):
        """Test when DataFrame has insufficient rows."""
        # Create a small DataFrame
        small_df = self.sample_df.iloc[:5]
        
        # Try to extract 10 rows (which is more than available)
        result = extract_data_range(small_df, num_rows=10, extract_recent=True)
        
        # Should return None
        self.assertIsNone(result)

    def test_custom_min_rows(self):
        """Test with custom minimum rows requirement."""
        # Try to extract 10 rows with a minimum requirement of 50
        result = extract_data_range(self.sample_df, num_rows=10, extract_recent=True, min_rows_required=50)
        
        # Should succeed since the DataFrame has 100 rows
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)
        
        # Try with higher minimum requirement
        result = extract_data_range(self.sample_df, num_rows=10, extract_recent=True, min_rows_required=150)
        
        # Should fail since the DataFrame only has 100 rows
        self.assertIsNone(result)

    def test_non_dataframe_input(self):
        """Test with non-DataFrame input."""
        # Pass a list instead of a DataFrame
        result = extract_data_range([1, 2, 3], num_rows=1, extract_recent=True)
        
        # Should return None
        self.assertIsNone(result)


class TestParquetReadWrite(unittest.TestCase):
    """Tests for parquet file read/write functions."""

    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample DataFrames for testing
        dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
        
        # Dictionary to hold test DataFrames
        self.test_dfs = {}
        
        for ticker in ['AAPL', 'GOOG', 'MSFT']:
            data = {
                'Open': np.random.rand(30) * 100,
                'Close': np.random.rand(30) * 100,
                'Volume': np.random.randint(1000, 10000, 30)
            }
            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Date'
            self.test_dfs[ticker] = df

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_write_and_read_parquet(self):
        """Test writing and reading parquet files using our utility functions."""
        # Create a config dictionary with the required keys
        config = {OUTPUT_DIR: self.temp_dir}
        
        # First, write the DataFrames to parquet files
        success = write_dataframes_to_parquet(self.test_dfs, config, max_workers=1)
        
        # Check that write was successful
        self.assertTrue(success)
        
        # Check that files were created
        files = os.listdir(self.temp_dir)
        self.assertEqual(len(files), len(self.test_dfs))
        
        # Now read the parquet files back
        read_dfs = read_parquet_files_from_directory(self.temp_dir, max_workers=1)
        
        # Check that we got the same number of DataFrames
        self.assertEqual(len(read_dfs), len(self.test_dfs))
        
        # Check that the DataFrames have the same content
        for ticker in self.test_dfs:
            self.assertIn(ticker, read_dfs)
            
            # Create copies of DataFrames to avoid modifying originals
            original_df = self.test_dfs[ticker].copy()
            read_df = read_dfs[ticker].copy()
            
            # Parquet doesn't preserve the frequency of DatetimeIndex
            # Reset the frequency to None for comparison
            if hasattr(original_df.index, 'freq') and original_df.index.freq is not None:
                original_df.index = pd.DatetimeIndex(original_df.index, freq=None)
            
            # Now compare the DataFrames
            pd.testing.assert_frame_equal(read_df, original_df)
    
    def test_max_workers_from_config(self):
        """Test that max_workers is correctly read from config."""
        # Create a config with MAX_WORKERS
        config = {OUTPUT_DIR: self.temp_dir, MAX_WORKERS: 3}
        
        # Mock the ProcessPoolExecutor to verify max_workers
        original_executor = concurrent.futures.ProcessPoolExecutor
        
        try:
            # Use a list to track the max_workers value
            captured_max_workers = []
            
            # Define a mock executor to capture max_workers
            def mock_executor(max_workers=None):
                captured_max_workers.append(max_workers)
                return original_executor(max_workers=max_workers)
            
            # Replace the executor with our mock
            concurrent.futures.ProcessPoolExecutor = mock_executor
            
            # Call write_dataframes_to_parquet without specifying max_workers
            write_dataframes_to_parquet(self.test_dfs, config)
            
            # Check that max_workers was correctly read from config
            self.assertEqual(captured_max_workers[0], 3)
            
        finally:
            # Restore the original executor
            concurrent.futures.ProcessPoolExecutor = original_executor


if __name__ == '__main__':
    unittest.main() 