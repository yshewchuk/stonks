from model.portfolio import Portfolio
from utils.dataframe import print_dataframe_debugging_info
import pandas as pd
import numpy as np

class Simulation:
    """
    Manages data and portfolio for a single trading simulation run.

    This class encapsulates the data for a specific simulation period, manages a Portfolio object,
    and provides methods to initialize the simulation, advance through time steps, and retrieve
    data for model input at each step.
    """

    def __init__(self, data, portfolio):
        """
        Initializes a Simulation object.

        Args:
            data (pd.DataFrame): DataFrame containing stock data for the simulation period.
                                  It is expected to have a DatetimeIndex and MultiIndex columns with 'Ticker' level.
            portfolio (Portfolio): Portfolio object to be used for this simulation.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a Pandas DataFrame.")
        if not isinstance(portfolio, Portfolio):
            raise ValueError("Input 'portfolio' must be a Portfolio object.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame 'data' must have a DatetimeIndex.")
        if data.index.name != 'Date':
            raise ValueError("DataFrame 'data' index must be named 'Date'.")

        self.data = data.copy() # Use a copy to avoid modifying original data source
        self.portfolio = portfolio
        self.current_step_index = -1 # Initialize to -1, 'start()' will move to index 0
        self.simulation_dates = sorted(self.data.index.unique()) # Get sorted unique dates for iteration

        if not self.simulation_dates:
            raise ValueError("Input data DataFrame is empty or has no valid dates.")

        self.valuations = self.data[list(map(lambda x: (x, 'Close'), self.portfolio.holdings.keys()))].copy()

        self.start_date = self.simulation_dates[0] # First date in the data
        self.end_date = self.simulation_dates[-1] # Last date in the data


        self._initialize_portfolio_metrics_columns() # Setup columns for portfolio value tracking


    def _initialize_portfolio_metrics_columns(self):
        """
        Initializes columns in the DataFrame to store portfolio-related metrics.

        Adds columns for 'Relative Value', 'Cash Percent of Value', and 'Holdings Percent of Value'
        for each ticker at each date step in the simulation data. These are initialized to 0 or a default value.
        """
        self.data['Relative Value'] = 0.0
        self.data['Cash Percent of Value'] = 0.0
        for ticker in self.portfolio.holdings:
            self.data[(ticker, 'Holdings Percent of Value')] = 0.0 # MultiIndex column

    def start(self, window):
        """
        Starts the simulation and returns the initial data window for the model.
        ... (rest of docstring is the same) ...
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        if window > len(self.simulation_dates):
            raise ValueError("Window size is larger than the available simulation data.")

        self.current_step_index = 0 # Start at the first date
        end_index = min(window, len(self.simulation_dates)) # Ensure end_index is within bounds

        # Calculate portfolio metrics for the initial window
        for i in range(end_index):
            current_date = self.simulation_dates[i]
            # Correct way to get stock prices - create simple dict {ticker: price}
            stock_prices_at_date = {}
            for ticker in self.portfolio.holdings:
                stock_prices_at_date[ticker] = self.data.loc[current_date, (ticker, 'Close')]

            portfolio_value = self.portfolio.value(stock_prices_at_date)
            relative_value = portfolio_value / self.portfolio.cash # Example: Relative to starting cash (can be adjusted)

            self.data.loc[current_date, 'Relative Value'] = relative_value
            self.data.loc[current_date, 'Cash Percent of Value'] = self.portfolio.cash / portfolio_value if portfolio_value != 0 else 0 # Avoid division by zero

            for ticker in self.portfolio.holdings:
                ticker_holding_value = self.portfolio.holdings[ticker] * stock_prices_at_date[ticker]
                self.data.loc[current_date, (ticker, 'Holdings Percent of Value')] = ticker_holding_value / portfolio_value if portfolio_value != 0 else 0 # Avoid division by zero

        return self.get_current_data_window(window) # Return data for the first 'window' steps


    def step(self, window):
        """
        Advances the simulation by one time step and returns the next data window.
        ... (rest of docstring is the same) ...
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        if self.current_step_index == -1:
            raise Exception("Simulation must be started first by calling 'start()'.")

        self.current_step_index += 1 # Move to the next date

        if self.current_step_index >= len(self.simulation_dates):
            print("Simulation reached end date.")
            return None # Simulation has ended

        current_date = self.simulation_dates[self.current_step_index]
        # Correct way to get stock prices - create simple dict {ticker: price}
        stock_prices_at_date = {}
        for ticker in self.portfolio.holdings:
            stock_prices_at_date[ticker] = self.data.loc[current_date, (ticker, 'Close')]

        portfolio_value = self.portfolio.value(stock_prices_at_date)
        relative_value = portfolio_value / self.portfolio.cash # Example relative value

        self.data.loc[current_date, 'Relative Value'] = relative_value
        self.data.loc[current_date, 'Cash Percent of Value'] = self.portfolio.cash / portfolio_value if portfolio_value != 0 else 0 # Avoid division by zero

        for ticker in self.portfolio.holdings:
            ticker_holding_value = self.portfolio.holdings[ticker] * stock_prices_at_date[ticker]
            self.data.loc[current_date, (ticker, 'Holdings Percent of Value')] = ticker_holding_value / portfolio_value if portfolio_value != 0 else 0 # Avoid division by zero


        return self.get_current_data_window(window) # Return data for the current 'window' steps


    def get_current_data_window(self, window):
        """
        Returns a DataFrame slice representing the current data window.

        Extracts a DataFrame slice from 'self.data' starting from the 'current_step_index'
        backwards for 'window' steps. This provides the data window for model input.

        Args:
            window (int): The size of the data window (number of days).

        Returns:
            pd.DataFrame: DataFrame slice representing the current data window,
                          or None if current step index is invalid.
        """
        if self.current_step_index < 0 or self.current_step_index >= len(self.simulation_dates):
            return None # Invalid step index

        start_index = max(0, self.current_step_index - window + 1) # Ensure start index is not negative
        start_date = self.simulation_dates[start_index]
        current_date = self.simulation_dates[self.current_step_index]

        data_window = self.data.loc[start_date : current_date].copy() # Get slice of data for window
        return data_window


    @property
    def current_date(self):
        """
        Returns the current simulation date.

        Returns:
            pd.Timestamp or None: The current simulation date, or None if simulation hasn't started or has ended.
        """
        if 0 <= self.current_step_index < len(self.simulation_dates):
            return self.simulation_dates[self.current_step_index]
        return None

    @property
    def is_simulation_finished(self):
        """
        Checks if the simulation has reached the end date.

        Returns:
            bool: True if simulation has finished, False otherwise.
        """
        return self.current_step_index >= len(self.simulation_dates)



# Example Usage (for testing and demonstration)
if __name__ == '__main__':
    # Create sample data and portfolio (replace with your actual data loading and portfolio setup)
    dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-10'])
    tickers = ['AAPL', 'GOOG']
    index = pd.MultiIndex.from_product([dates, tickers], names=['Date', 'Ticker'])
    data_values = np.random.rand(len(dates) * len(tickers), 5) # 5 columns of random data
    columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']], names=['Ticker', 'OHLCV'])
    sample_data = pd.DataFrame(data_values, index=index, columns=columns)

    initial_cash = 100000
    sample_portfolio = Portfolio(initial_cash, tickers)


    # Create Simulation instance
    simulation = Simulation(sample_data, sample_portfolio)

    print("--- Simulation Start ---")
    window_size = 5
    initial_data_window = simulation.start(window_size)
    if initial_data_window is not None:
        print("\nInitial Data Window (First 5 days):")
        print(initial_data_window)
        print_dataframe_debugging_info(initial_data_window, name="Initial Data Window")


    print("\n--- Simulation Steps ---")
    for _ in range(3): # Take a few steps
        current_data_window = simulation.step(window_size)
        if current_data_window is not None:
            print(f"\nData Window at Date: {simulation.current_date.date()}")
            print(current_data_window)
            print_dataframe_debugging_info(current_data_window, name=f"Data Window at Date: {simulation.current_date.date()}")
        else:
            print("Simulation ended.")
            break

    print("\n--- Simulation End ---")
    print(f"Is simulation finished? {simulation.is_simulation_finished}")
    print(f"Current portfolio cash: ${simulation.portfolio.cash:.2f}")
    print(f"Current portfolio holdings: {simulation.portfolio.holdings}")