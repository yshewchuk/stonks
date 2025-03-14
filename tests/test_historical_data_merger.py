"""
Test script for the HistoricalDataMerger class
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transforms.historical_data_merger import HistoricalDataMerger

class TestHistoricalDataMerger(unittest.TestCase):
    """Test cases for the HistoricalDataMerger class"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample DataFrames for different tickers
        self.dates = pd.date_range(start='2020-01-01', end='2020-12-31')
        
        # Create AAPL data
        self.aapl_data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(self.dates)),
            'High': np.random.normal(105, 10, len(self.dates)),
            'Low': np.random.normal(95, 10, len(self.dates)),
            'Close': np.random.normal(102, 10, len(self.dates)),
            'Volume': np.random.randint(1000, 100000, len(self.dates)),
            'MA5': np.random.normal(100, 10, len(self.dates)),
            'RSI5': np.random.uniform(0, 100, len(self.dates)),
            'PPCProb_F1-3D_Min-3_Max-1': np.random.uniform(0, 1, len(self.dates))
        }, index=self.dates)
        
        # Set the index name
        self.aapl_data.index.name = 'Date'
        
        # Create MSFT data
        self.msft_data = pd.DataFrame({
            'Open': np.random.normal(200, 20, len(self.dates)),
            'High': np.random.normal(210, 20, len(self.dates)),
            'Low': np.random.normal(190, 20, len(self.dates)),
            'Close': np.random.normal(205, 20, len(self.dates)),
            'Volume': np.random.randint(2000, 200000, len(self.dates)),
            'MA5': np.random.normal(200, 20, len(self.dates)),
            'RSI5': np.random.uniform(0, 100, len(self.dates)),
            'PPCProb_F1-3D_Min-3_Max-1': np.random.uniform(0, 1, len(self.dates))
        }, index=self.dates)
        
        # Set the index name
        self.msft_data.index.name = 'Date'
        
        # Create a merger with test prefixes
        self.merger = HistoricalDataMerger(
            historical_column_prefixes=['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'RSI']
        )
    
    def test_initialization(self):
        """Test the merger initialization"""
        self.assertTrue(hasattr(self.merger, 'historical_column_prefixes'))
        self.assertIsInstance(self.merger.historical_column_prefixes, list)
        self.assertTrue(all(isinstance(p, str) for p in self.merger.historical_column_prefixes))
    
    def test_is_historical_column(self):
        """Test the historical column identification"""
        # Test historical columns
        self.assertTrue(self.merger._is_historical_column('Open'))
        self.assertTrue(self.merger._is_historical_column('MA5'))
        self.assertTrue(self.merger._is_historical_column('RSI5'))
        
        # Test non-historical columns
        self.assertFalse(self.merger._is_historical_column('PPCProb_F1-3D_Min-3_Max-1'))
        self.assertFalse(self.merger._is_historical_column('OtherColumn'))
    
    def test_filter_historical_columns(self):
        """Test filtering of historical columns"""
        # Test with AAPL data
        filtered_df = self.merger._filter_historical_columns(self.aapl_data)
        
        # Check that historical columns are included
        self.assertIn('Open', filtered_df.columns)
        self.assertIn('MA5', filtered_df.columns)
        self.assertIn('RSI5', filtered_df.columns)
        
        # Check that non-historical columns are excluded
        self.assertNotIn('PPCProb_F1-3D_Min-3_Max-1', filtered_df.columns)
    
    def test_merge(self):
        """Test merging of multiple DataFrames by date"""
        # Create dictionary of DataFrames
        ticker_dataframes = {
            'AAPL': self.aapl_data,
            'MSFT': self.msft_data
        }
        
        # Merge the DataFrames
        merged_df = self.merger.merge(ticker_dataframes)
        
        # Check basic properties
        self.assertIsInstance(merged_df, pd.DataFrame)
        self.assertEqual(len(merged_df), len(self.dates))  # Should have one row per date
        
        # Check multi-level column structure
        self.assertIsInstance(merged_df.columns, pd.MultiIndex)
        self.assertEqual(merged_df.columns.names, ['Ticker', 'Feature'])
        
        # Check tickers in the first level
        tickers = merged_df.columns.levels[0]
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)
        
        # Check features in the second level
        features = merged_df.columns.levels[1]
        for feature in ['Open', 'Close', 'MA5', 'RSI5']:
            self.assertIn(feature, features)
        
        # Check that non-historical columns are excluded
        self.assertNotIn('PPCProb_F1-3D_Min-3_Max-1', features)
        
        # Check specific values for a date
        sample_date = self.dates[10]
        self.assertEqual(merged_df.loc[sample_date, ('AAPL', 'Open')], self.aapl_data.loc[sample_date, 'Open'])
        self.assertEqual(merged_df.loc[sample_date, ('MSFT', 'Close')], self.msft_data.loc[sample_date, 'Close'])
    
    def test_merge_with_missing_dates(self):
        """Test merging with missing dates in some DataFrames"""
        # Create AAPL data with some dates missing
        aapl_subset = self.aapl_data.iloc[10:300]  # Subset of dates
        msft_full = self.msft_data.copy()  # All dates
        
        ticker_dataframes = {
            'AAPL': aapl_subset,
            'MSFT': msft_full
        }
        
        # Merge with outer join (default) - should include all dates
        merged_df = self.merger.merge(ticker_dataframes)
        
        # Check that the merged DataFrame has all dates
        self.assertEqual(len(merged_df), len(self.dates))
        
        # Check that AAPL data is NaN for dates not in aapl_subset
        for date in self.dates[:10]:
            self.assertTrue(np.isnan(merged_df.loc[date, ('AAPL', 'Open')]))
            self.assertFalse(np.isnan(merged_df.loc[date, ('MSFT', 'Open')]))
    
    def test_invalid_input(self):
        """Test handling of invalid input"""
        # Test with empty dictionary
        with self.assertRaises(ValueError):
            self.merger.merge({})
        
        # Test with non-dictionary input
        with self.assertRaises(ValueError):
            self.merger.merge("not a dictionary")
        
        # Test with invalid DataFrame
        invalid_dataframes = {
            'AAPL': "not a DataFrame",
            'MSFT': self.msft_data
        }
        merged_df = self.merger.merge(invalid_dataframes)
        
        # Should still have all dates, but only MSFT data
        self.assertEqual(len(merged_df), len(self.dates))
        self.assertIn(('MSFT', 'Open'), merged_df.columns)
        self.assertNotIn(('AAPL', 'Open'), merged_df.columns)

if __name__ == '__main__':
    unittest.main() 