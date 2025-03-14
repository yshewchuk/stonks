import pandas as pd
import numpy as np
import os

from transforms.moving_average import MovingAverage
from transforms.rolling_hi_lo import RollingHiLo
from transforms.relative_strength import RSI
from transforms.macd import MACD
from transforms.percent_price_change_probability import PercentPriceChangeProbability
from transforms.lagged_features import LaggedFeatures

class RawStockPreProcessor:
    """
    Pre-processes raw stock data by applying various transformations.
    
    This class consolidates multiple transformations:
    - Moving Averages
    - Rolling Highs and Lows
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Percent Price Change Probabilities
    - Lagged Features
    
    It handles filtering to required columns and ensures data is properly formatted.
    """
    
    def __init__(self, 
                 required_columns,
                 windows,
                 lag_periods,
                 ppc_configs=None):
        """
        Initialize the pre-processor with configuration for each transformation.
        
        Args:
            required_columns (list): List of columns required from raw data
            windows (list): List of windows for Moving Average, Rolling High/Low, and RSI
            lag_periods (list): List of periods for lagged features
            ppc_configs (list): List of configs for Percent Price Change Probability 
                                Each config is a dict with keys:
                                - start_days_future
                                - end_days_future
                                - percent_change_bounds (list of bounds)
        """
        # Store configuration
        self.required_columns = required_columns
        self.windows = windows
        self.lag_periods = lag_periods
        self.ppc_configs = ppc_configs
        
        # Initialize transformers
        self.transformers = []
        self._init_transformers()
        
    def _init_transformers(self):
        """Initialize all transformer objects and store them in a single list."""
        # Clear existing transformers list
        self.transformers = []
        
        # Add Moving Average transformers
        for window in self.windows:
            self.transformers.append(MovingAverage(window))
        
        # Add Rolling Hi-Lo transformers
        for window in self.windows:
            self.transformers.append(RollingHiLo(window))
        
        # Add RSI transformers
        for window in self.windows:
            self.transformers.append(RSI(window))
        
        # Add MACD transformer (using default parameters)
        self.transformers.append(MACD())
        
        # Add Percent Price Change Probability transformers
        if self.ppc_configs:
            for config in self.ppc_configs:
                start_days = config.get('start_days_future')
                end_days = config.get('end_days_future')
                bounds = config.get('percent_change_bounds', [])
                
                # Convert bounds to min/max pairs
                if bounds and len(bounds) > 1:
                    # Create a transformer for each adjacent pair of bounds
                    for i in range(len(bounds) - 1):
                        lower_bound = bounds[i] / 100.0  # Convert to decimal
                        upper_bound = bounds[i + 1] / 100.0  # Convert to decimal
                        
                        self.transformers.append(
                            PercentPriceChangeProbability(
                                start_days_future=start_days,
                                end_days_future=end_days,
                                min_percent_change=lower_bound,
                                max_percent_change=upper_bound
                            )
                        )
                    
                    # Add transformer for values below the first bound
                    self.transformers.append(
                        PercentPriceChangeProbability(
                            start_days_future=start_days,
                            end_days_future=end_days,
                            max_percent_change=bounds[0] / 100.0
                        )
                    )
                    
                    # Add transformer for values above the last bound
                    self.transformers.append(
                        PercentPriceChangeProbability(
                            start_days_future=start_days,
                            end_days_future=end_days,
                            min_percent_change=bounds[-1] / 100.0
                        )
                    )
    
    def process(self, df):
        """
        Process a DataFrame with raw stock data, applying all transformations.
        
        Args:
            df (pd.DataFrame): DataFrame with raw stock data
            
        Returns:
            pd.DataFrame: Processed DataFrame with all calculated indicators and features
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            print("❌ Error: Input DataFrame is empty or invalid")
            return None
            
        print(f"Processing DataFrame of shape: {df.shape}")
        
        # Make a copy to avoid modifying the original
        processed_df = df[self.required_columns].copy()
        
        # Set the index name to 'Date' if it doesn't have a name
        if processed_df.index.name is None:
            processed_df.index.name = 'Date'
        
        try:
            # Apply all transformers
            for transformer in self.transformers:
                try:
                    processed_df = transformer.extend(processed_df)
                except Exception as e:
                    print(f"⚠️ Warning: Error applying transformer {transformer.__class__.__name__}: {e}")
            
            # Determine which columns to lag (all except PPC columns)
            historical_columns = [col for col in processed_df.columns 
                                 if not col.startswith('PPCProb')]
            
            # Apply lagged features
            lag_transformer = LaggedFeatures(self.lag_periods, historical_columns)
            processed_df = lag_transformer.extend(processed_df)
            
            # Drop rows with NaN values (expected at the beginning due to rolling and lagging)
            original_rows = len(processed_df)
            processed_df.dropna(inplace=True)
            print(f"✅ Dropped {original_rows - len(processed_df)} rows with NaN values")
            
            return processed_df
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            return None 