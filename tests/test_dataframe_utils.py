import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from utils.dataframe import write_dataframes_to_parquet, read_parquet_files_from_directory, truncate_recent_data
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

if __name__ == '__main__':
    unittest.main() 