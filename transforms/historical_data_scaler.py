import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class HistoricalDataScaler:
    """
    Scales historical market data for model training.
    
    This class handles scaling of different types of financial data:
    1. Price data (Open, High, Low, Close, etc.) - scaled relative to first open price
    2. Volume data - scaled using MinMaxScaler on the data window
    3. RSI data - scaled based on known range (0-100)
    4. MACD data - scaled using MinMaxScaler on the data window
    
    The scaler can be configured to handle multi-index DataFrames with ticker symbols
    as the first level of the index.
    """
    
    def __init__(self, price_column_tags, volume_prefix, rsi_prefix, macd_prefix):
        """
        Initialize the HistoricalDataScaler.
        
        Args:
            price_column_tags (list): List of strings that identify price-related columns
                (e.g., ['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo'])
            volume_prefix (str): Prefix that identifies volume columns (e.g., 'Volume')
            rsi_prefix (str): Prefix that identifies RSI columns (e.g., 'RSI')
            macd_prefix (str): Prefix that identifies MACD columns (e.g., 'MACD')
        """
        # Lists of column tags/prefixes for different data types
        self.price_column_tags = price_column_tags
        self.volume_prefix = volume_prefix
        self.rsi_prefix = rsi_prefix
        self.macd_prefix = macd_prefix
        
        # Pre-create scalers to avoid creating new ones for each column
        self.volume_scaler = MinMaxScaler()
        self.macd_scaler = MinMaxScaler()
        
    def _is_price_column(self, column_name):
        """
        Check if a column is a price-related column.
        
        Args:
            column_name (str): Name of the column to check
            
        Returns:
            bool: True if the column is price-related, False otherwise
        """
        return any(tag in column_name for tag in self.price_column_tags)
    
    def _is_volume_column(self, column_name):
        """Check if column is a volume column."""
        return self.volume_prefix in column_name
    
    def _is_rsi_column(self, column_name):
        """Check if column is an RSI column."""
        return self.rsi_prefix in column_name
    
    def _is_macd_column(self, column_name):
        """Check if column is a MACD column."""
        return self.macd_prefix in column_name
    
    def scale_dataframe(self, df):
        """
        Scale the entire DataFrame using appropriate scaling for each column type.
        
        Args:
            df (pd.DataFrame): DataFrame to scale, with datetime index and potentially
                               multi-index columns with tickers as the first level
                               
        Returns:
            pd.DataFrame: Scaled DataFrame
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or invalid")
      
        # Create a copy to avoid modifying the original
        # Using copy is actually more efficient for multiprocessing
        # as it avoids potential race conditions and makes sure each process
        # has its own memory space
        scaled_df = df.copy()
        
        # Scale the content
        self._scale_content(scaled_df)
            
        return scaled_df
    
    def _scale_content(self, df):
        """
        Scale the DataFrame content based on column types.
        Works with both multi-index and single-index DataFrames.
        
        Args:
            df (pd.DataFrame): DataFrame to scale
        """
        # Check if we have multi-index columns (ticker, feature)
        has_multi_index = isinstance(df.columns, pd.MultiIndex)
        
        # Group columns by type for more efficient processing
        price_columns = []
        volume_columns = []
        rsi_columns = []
        macd_columns = []
        
        # Identify column types and organize them for batch processing
        for col in df.columns:
            if has_multi_index:
                _, feature_name = col
            else:
                feature_name = col
                
            if self._is_price_column(feature_name):
                price_columns.append(col)
            elif self._is_volume_column(feature_name):
                volume_columns.append(col)
            elif self._is_rsi_column(feature_name):
                rsi_columns.append(col)
            elif self._is_macd_column(feature_name):
                macd_columns.append(col)
        
        price_scalers = {}

        # Process price columns
        if price_columns:
            # Scale all price columns
            for col in price_columns:
                
                if has_multi_index:
                    open_col = (col[0], 'Open')
                else:
                    open_col = 'Open'

                if not open_col in price_scalers:
                    if open_col in df.columns:
                        first_price = df[open_col].iloc[0]
                    else:
                        first_price = 0
                        
                    price_scalers[open_col] = 1.0 if first_price == 0 else first_price

                scaler = price_scalers[open_col]
                df[col] = (df[col] - scaler) / scaler  
                
        # Process volume columns - use vectorized operations where possible
        if volume_columns:
            for col in volume_columns:
                data = df[col].values.reshape(-1, 1)
                df[col] = self.volume_scaler.fit_transform(data).flatten()
        
        # Process RSI columns - optimized with vectorized operations
        if rsi_columns:
            for col in rsi_columns:
                # Ensure values are within expected range before scaling
                df[col] = np.clip(df[col], 0, 100) / 100.0
        
        # Process MACD columns
        if macd_columns:
            # Process each MACD column
            for col in macd_columns:
                data = df[col].values.reshape(-1, 1)
                df[col] = self.macd_scaler.fit_transform(data).flatten()
    