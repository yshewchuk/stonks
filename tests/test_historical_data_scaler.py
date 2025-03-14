import unittest
import pandas as pd
import numpy as np
from transforms.historical_data_scaler import HistoricalDataScaler

class TestHistoricalDataScaler(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for each test case."""
        self.scaler = HistoricalDataScaler(
            price_column_tags=['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo'],
            volume_prefix='Volume',
            rsi_prefix='RSI',
            macd_prefix='MoACD'
        )
        
        # Create test data for single-level columns
        data = {
            'Open': [100, 102, 105, 103, 101],
            'High': [105, 108, 110, 106, 103],
            'Low': [98, 100, 102, 100, 99],
            'Close': [102, 105, 103, 101, 100],
            'Volume': [1000, 1200, 800, 1100, 900],
            'RSI5': [30, 45, 60, 75, 90],  # RSI values between 0-100
            'MoACD_Fast12_Slow26': [-0.5, -0.2, 0.1, 0.3, 0.2]
        }
        self.single_df = pd.DataFrame(data, index=pd.date_range('2023-01-01', periods=5))
        
        # Create test data for multi-index columns (ticker, feature)
        # Two tickers: AAPL and MSFT
        columns = pd.MultiIndex.from_product(
            [['AAPL', 'MSFT'], 
             ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI5', 'MoACD_Fast12_Slow26']],
            names=['Ticker', 'Feature']
        )
        
        # AAPL data
        aapl_data = np.array([
            [150, 155, 148, 152, 2000, 40, -0.3],
            [152, 158, 150, 156, 2200, 55, -0.1],
            [156, 160, 154, 158, 1800, 70, 0.2],
            [158, 162, 155, 160, 2100, 85, 0.4],
            [155, 159, 152, 154, 1900, 65, 0.1]
        ])
        
        # MSFT data
        msft_data = np.array([
            [250, 255, 248, 252, 1500, 45, -0.4],
            [252, 258, 250, 256, 1700, 60, -0.2],
            [256, 260, 254, 258, 1300, 75, 0.3],
            [258, 262, 255, 260, 1600, 90, 0.5],
            [255, 259, 252, 254, 1400, 50, 0.0]
        ])
        
        # Combine data
        multi_data = np.hstack([aapl_data, msft_data])
        self.multi_df = pd.DataFrame(multi_data, columns=columns, 
                                     index=pd.date_range('2023-01-01', periods=5))
    
    def test_single_df_scaling(self):
        """Test scaling of DataFrame with single-level columns."""
        scaled_df = self.scaler.scale_dataframe(self.single_df)
        
        # Check that the first Open price is zeroed out
        self.assertAlmostEqual(scaled_df['Open'].iloc[0], 0.0)
        
        # Check that price columns are scaled relative to first Open price
        # For example, High should be (105-100)/100 = 0.05
        self.assertAlmostEqual(scaled_df['High'].iloc[0], 0.05)
        
        # Check that Volume is scaled to [0,1] range
        self.assertTrue(0 <= scaled_df['Volume'].min() <= scaled_df['Volume'].max() <= 1)
        
        # Check that RSI is scaled by dividing by 100
        # RSI values were [30, 45, 60, 75, 90], so should be [0.3, 0.45, 0.6, 0.75, 0.9]
        expected_rsi = [0.3, 0.45, 0.6, 0.75, 0.9]
        for i, expected in enumerate(expected_rsi):
            self.assertAlmostEqual(scaled_df['RSI5'].iloc[i], expected)
        
        # Check that MACD is min-max scaled to [0,1]
        self.assertTrue(0 <= scaled_df['MoACD_Fast12_Slow26'].min() <= 
                        scaled_df['MoACD_Fast12_Slow26'].max() <= 1)
    
    def test_multi_df_scaling(self):
        """Test scaling of DataFrame with multi-index columns."""
        scaled_df = self.scaler.scale_dataframe(self.multi_df)
        
        # Check scaling for each ticker
        for ticker in ['AAPL', 'MSFT']:
            # First Open price should be zeroed out
            self.assertAlmostEqual(scaled_df[(ticker, 'Open')].iloc[0], 0.0)
            
            # Check relative price scaling
            first_open = self.multi_df[(ticker, 'Open')].iloc[0]
            expected_high = (self.multi_df[(ticker, 'High')].iloc[0] - first_open) / first_open
            self.assertAlmostEqual(scaled_df[(ticker, 'High')].iloc[0], expected_high)
            
            # Check Volume scaling to [0,1]
            self.assertTrue(0 <= scaled_df[(ticker, 'Volume')].min() <= 
                            scaled_df[(ticker, 'Volume')].max() <= 1)
            
            # Check RSI scaling (divide by 100)
            for i in range(5):
                original_rsi = self.multi_df[(ticker, 'RSI5')].iloc[i]
                expected_rsi = original_rsi / 100.0
                self.assertAlmostEqual(scaled_df[(ticker, 'RSI5')].iloc[i], expected_rsi)
            
            # Check MACD min-max scaling
            self.assertTrue(0 <= scaled_df[(ticker, 'MoACD_Fast12_Slow26')].min() <= 
                            scaled_df[(ticker, 'MoACD_Fast12_Slow26')].max() <= 1)
    
    def test_empty_df(self):
        """Test that the scaler handles empty DataFrames appropriately."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.scaler.scale_dataframe(empty_df)
    
    def test_missing_columns(self):
        """Test handling of DataFrames missing expected columns."""
        # DataFrame without Open price
        no_open_df = self.single_df.drop(columns=['Open'])
        scaled_df = self.scaler.scale_dataframe(no_open_df)
        
        # Should still scale other column types appropriately
        # RSI scaling should still work
        expected_rsi = [0.3, 0.45, 0.6, 0.75, 0.9]
        for i, expected in enumerate(expected_rsi):
            self.assertAlmostEqual(scaled_df['RSI5'].iloc[i], expected)
        
        # Volume should still be scaled
        self.assertTrue(0 <= scaled_df['Volume'].min() <= scaled_df['Volume'].max() <= 1)

if __name__ == '__main__':
    unittest.main() 