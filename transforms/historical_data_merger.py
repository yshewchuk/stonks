import pandas as pd
import numpy as np

class HistoricalDataMerger:
    """
    Merges historical data from multiple tickers into a single dataset.
    
    This class:
    1. Filters columns based on allowlisted prefixes
    2. Creates a multi-level column structure (ticker, feature)
    3. Merges data by date, so each row contains data for all tickers on a particular date
    """
    
    def __init__(self, historical_column_prefixes):
        """
        Initialize the merger with prefixes that identify historical columns.
        
        Args:
            historical_column_prefixes (list): List of prefixes that identify historical columns.
                                             Only columns starting with these prefixes will be included.
        """
        if not isinstance(historical_column_prefixes, list) or not all(isinstance(p, str) for p in historical_column_prefixes):
            raise ValueError("historical_column_prefixes must be a list of strings.")
            
        self.historical_column_prefixes = historical_column_prefixes
    
    def _is_historical_column(self, column_name):
        """Check if a column name starts with any of the historical prefixes."""
        return any(column_name.startswith(prefix) for prefix in self.historical_column_prefixes)
    
    def _filter_historical_columns(self, df):
        """Filter DataFrame to only include historical columns."""
        historical_columns = [col for col in df.columns if self._is_historical_column(col)]
        return df[historical_columns]  # Return view instead of copy
    
    def merge(self, ticker_dataframes):
        """
        Merge historical data from multiple tickers into a single DataFrame.
        Creates a wide-format DataFrame where:
        - Each row represents a single date
        - Columns have a multi-level structure: (ticker, feature)
        
        Args:
            ticker_dataframes (dict): Dictionary mapping ticker symbols to DataFrames
            
        Returns:
            pd.DataFrame: Merged DataFrame containing historical data for all tickers by date
        """
        if not isinstance(ticker_dataframes, dict):
            raise ValueError("ticker_dataframes must be a dictionary.")
            
        if not ticker_dataframes:
            raise ValueError("ticker_dataframes is empty.")
            
        # Process each ticker's DataFrame
        processed_dfs = {}
        for ticker, df in ticker_dataframes.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"⚠️ Warning: Skipping invalid DataFrame for ticker {ticker}")
                continue
                
            # Filter to historical columns
            historical_df = self._filter_historical_columns(df)
            
            # Ensure the DataFrame has a datetime index named 'Date'
            if historical_df.index.name != 'Date':
                # If the index is not named 'Date', check if there's a 'Date' column
                if 'Date' in historical_df.columns:
                    historical_df = historical_df.set_index('Date')
                else:
                    print(f"⚠️ Warning: DataFrame for {ticker} has no Date index or column. Using current index.")
                    historical_df.index.name = 'Date'
            
            # Create multi-level columns with ticker as the first level
            historical_df.columns = pd.MultiIndex.from_product([[ticker], historical_df.columns], 
                                                              names=['Ticker', 'Feature'])
            
            processed_dfs[ticker] = historical_df
        
        if not processed_dfs:
            raise ValueError("No valid DataFrames to merge.")
            
        # Join all DataFrames on the date index
        # The 'how' parameter determines which dates to include:
        # - 'inner': Only dates present in all DataFrames
        # - 'outer': All dates from any DataFrame (will have NaNs where data is missing)
        merged_df = pd.concat(processed_dfs.values(), axis=1, join='outer')
        
        # Sort the index to ensure chronological order
        merged_df = merged_df.sort_index()
        
        # Print summary
        print(f"✅ Merged {len(processed_dfs)} tickers into a single dataset by date")
        print(f"📊 Final shape: {merged_df.shape}")
        print(f"📅 Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"📈 Number of features per ticker: {len(merged_df.columns.levels[1])}")
        
        return merged_df 