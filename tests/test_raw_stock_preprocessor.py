"""
Test script for the RawStockPreProcessor class
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transforms.raw_stock_preprocessor import RawStockPreProcessor

class TestRawStockPreProcessor(unittest.TestCase):
    """Test cases for the RawStockPreProcessor class"""
    
    def setUp(self):
        """Set up test data"""
        # Create a sample DataFrame with daily stock data
        self.dates = pd.date_range(start='2020-01-01', end='2020-12-31')
        self.sample_data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(self.dates)),
            'High': np.random.normal(105, 10, len(self.dates)),
            'Low': np.random.normal(95, 10, len(self.dates)),
            'Close': np.random.normal(102, 10, len(self.dates)),
            'Volume': np.random.randint(1000, 100000, len(self.dates))
        }, index=self.dates)
        
        # Ensure High >= Open, Low <= Open, High >= Close, Low <= Close
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            self.sample_data.at[self.dates[i], 'High'] = max(row['Open'], row['Close'], row['High'])
            self.sample_data.at[self.dates[i], 'Low'] = min(row['Open'], row['Close'], row['Low'])
        
        # Set the index name
        self.sample_data.index.name = 'Date'
        
        # Create a preprocessor with minimal configuration for faster tests
        self.preprocessor = RawStockPreProcessor(
            required_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
            windows=[5],
            lag_periods=[1, 2],
            ppc_configs=[{
                'start_days_future': 1,
                'end_days_future': 3,
                'percent_change_bounds': [1, 3]
            }]
        )
    
    def test_initialization(self):
        """Test the transformer initialization process"""
        # Check that transformers were created
        self.assertTrue(hasattr(self.preprocessor, 'transformers'))
        self.assertIsInstance(self.preprocessor.transformers, list)
        
        # We should have:
        # - 1 Moving Average transformer for window 5
        # - 1 Rolling HiLo transformer for window 5
        # - 1 RSI transformer for window 5
        # - 1 MACD transformer
        # - 3 PPC transformers (for: < 1%, 1-3%, > 3%)
        expected_transformers_count = 1 + 1 + 1 + 1 + 3
        self.assertEqual(len(self.preprocessor.transformers), expected_transformers_count)
    
    def test_process_dataframe(self):
        """Test processing a single DataFrame"""
        processed_df = self.preprocessor.process(self.sample_data)
        
        # Check basic properties
        self.assertIsNotNone(processed_df)
        self.assertIsInstance(processed_df, pd.DataFrame)
        
        # Print columns for debugging
        print("Available columns:", processed_df.columns.tolist())
        
        # Check that we have all the original columns
        for col in self.preprocessor.required_columns:
            self.assertIn(col, processed_df.columns)
        
        # Check that we have the calculated indicators
        for col_name in ['MA5', 'Hi5', 'Lo5', 'RSI5']:
            self.assertIn(col_name, processed_df.columns)
        
        # Check that we have MACD columns
        macd_columns = [col for col in processed_df.columns if 'MoACD' in col]
        self.assertGreater(len(macd_columns), 0)
        
        # Check that we have PPC columns
        ppc_columns = [col for col in processed_df.columns if 'PPCProb' in col]
        self.assertGreater(len(ppc_columns), 0)
        
        # Check that we have lagged features
        lagged_columns = [col for col in processed_df.columns if '_Lag' in col]
        self.assertGreater(len(lagged_columns), 0)
    
    def test_invalid_input(self):
        """Test handling of invalid input"""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = self.preprocessor.process(empty_df)
        self.assertIsNone(result)
        
        # Test with non-DataFrame input
        not_df = "not a dataframe"
        result = self.preprocessor.process(not_df)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main() 