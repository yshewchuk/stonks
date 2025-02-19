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

        self.start_date = self.simulation_dates[0] # First date in the data
        self.end_date = self.simulation_dates[-1] # Last date in the data

        # Store original closing prices for valuation BEFORE scaling
        self.valuations = {}
        for date in self.simulation_dates:
            self.valuations[date] = {}
            for ticker in self.portfolio.holdings:
                self.valuations[date][ticker] = self.data.loc[date, (ticker, 'Close')]

        self.transaction_prices = self.data[list(map(lambda x: (x, 'Open'), self.portfolio.holdings.keys()))].copy()


        self._initialize_portfolio_metrics_columns() # Setup columns for portfolio value tracking
        self._scale_price_data() # Apply price scaling
        self._calculate_initial_portfolio_value() # Calculate initial portfolio value after scaling, but using original prices for initial valuation


    def _calculate_initial_portfolio_value(self):
        """Calculates and stores the initial portfolio value at the start of the simulation."""
        initial_stock_prices = {}
        first_date = self.simulation_dates[0]
        for ticker in self.portfolio.holdings:
            initial_stock_prices[ticker] = self.valuations[first_date][ticker] # Use original prices for initial valuation
        self.initial_portfolio_value = self.portfolio.value(initial_stock_prices)
        if self.initial_portfolio_value == 0:
            self.initial_portfolio_value = self.portfolio.cash


    def _scale_price_data(self):
        """
        Scales price-related data in self.data to percentage difference from the first day's Open price.

        Identifies price-based columns using tags and scales them, excluding 'Volume' columns.
        Scaling is done relative to the Open price of the first date for each ticker.
        """
        first_date = self.simulation_dates[0]
        price_column_tags = ['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo'] # Tags to identify price-based columns

        for ticker in self.portfolio.holdings:
            initial_open_price = self.data.loc[first_date, (ticker, 'Open')]
            if initial_open_price == 0:
                initial_open_price = 1.0

            for col in self.data.columns:
                if col[0] == ticker and any(tag in col[1] for tag in price_column_tags): # Check ticker,if it's a price column
                    self.data[col] = (self.data[col] - initial_open_price) / initial_open_price



    def _initialize_portfolio_metrics_columns(self):
        """ ... """
        self.data['Relative Value'] = 0.0
        self.data['Cash Percent of Value'] = 0.0
        for ticker in self.portfolio.holdings:
            self.data[(ticker, 'Holdings Percent of Value')] = 0.0 # MultiIndex column


    def _calculate_portfolio_metrics(self, current_date):
        """
        Calculates and updates portfolio metrics for a given date in self.data.
        ... (rest of docstring) ...
        """
        stock_prices_at_date = {}
        for ticker in self.portfolio.holdings:
            stock_prices_at_date[ticker] = self.valuations[current_date][ticker] # Use original closing prices from valuations

        portfolio_value = self.portfolio.value(stock_prices_at_date)
        relative_value = portfolio_value / self.initial_portfolio_value
        self.data.loc[current_date, 'Relative Value'] = relative_value
        self.data.loc[current_date, 'Cash Percent of Value'] = self.portfolio.cash / portfolio_value if portfolio_value != 0 else 0
        for ticker in self.portfolio.holdings:
            ticker_holding_value = self.portfolio.holdings[ticker] * stock_prices_at_date[ticker]
            self.data.loc[current_date, (ticker, 'Holdings Percent of Value')] = ticker_holding_value / portfolio_value if portfolio_value != 0 else 0


    def start(self, window):
        """
        Starts the simulation and returns the initial data window for the model.

        Initializes portfolio metrics for the first 'window' days and returns the data
        slice for the first 'window' steps, ready to be used as input for a model.

        Args:
            window (int): The number of initial time steps (days) for the data window. Must be positive.

        Returns:
            pd.DataFrame: DataFrame slice containing the data for the initial 'window' steps,
                          with portfolio metric columns initialized.

        Raises:
            ValueError: If 'window' is not a positive integer or is larger than the available data.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        if window > len(self.simulation_dates):
            raise ValueError("Window size is larger than the available simulation data.")

        self.current_step_index = window - 1

        end_index = min(window, len(self.simulation_dates))

        for i in range(end_index):
            current_date = self.simulation_dates[i]
            self._calculate_portfolio_metrics(current_date)

        return self.get_current_data_window(window)


    def step(self, window):
        """
        Advances the simulation by one time step and returns the next data window.

        Increments the simulation step index to the next date in the data.
        Calculates updated portfolio metrics for the new current date.
        Returns a DataFrame slice containing the data for the current 'window' steps
        up to the new current date.

        Args:
            window (int): The number of time steps (days) for the data window. Must be positive and consistent with 'start()' window.

        Returns:
            pd.DataFrame or None: DataFrame slice for the current 'window' steps, or None if simulation end is reached.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        if self.current_step_index == -1:
            raise Exception("Simulation must be started first by calling 'start()'.")

        self.current_step_index += 1

        if self.current_step_index >= len(self.simulation_dates):
            print("Simulation reached end date.")
            return None

        current_date = self.simulation_dates[self.current_step_index]
        self._calculate_portfolio_metrics(current_date)

        return self.get_current_data_window(window)



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
            return None

        start_index = max(0, self.current_step_index - window + 1)
        start_date = self.simulation_dates[start_index]
        current_date = self.simulation_dates[self.current_step_index]

        data_window = self.data.loc[start_date : current_date].copy()
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
    # Create MultiIndex for columns - Ticker and OHLCV
    columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']], names=['Ticker', 'OHLCV']) # Added MA5 and MA20 for testing price scaling
    data_values = np.random.rand(len(dates), len(tickers) * 7) # Adjusted for extra MA columns
    sample_data = pd.DataFrame(data_values, index=dates, columns=columns) # dates as DatetimeIndex
    sample_data.index.name = 'Date'

    initial_cash = 100000
    sample_portfolio = Portfolio(initial_cash, tickers)

    # Create Simulation instance
    simulation = Simulation(sample_data, sample_portfolio)

    print("--- Simulation Start ---")
    window_size = 5
    initial_data_window = simulation.start(window_size)
    if initial_data_window is not None:
        print("\nInitial Data Window (First 5 days) - Scaled Prices:")
        print(initial_data_window)
        print_dataframe_debugging_info(initial_data_window, name="Initial Data Window - Scaled Prices")


    print("\n--- Simulation Steps (Running to End) ---")
    step_count = 0
    while True:
        current_data_window = simulation.step(window_size)
        if current_data_window is not None:
            step_count += 1
            print(f"\nData Window at Date: {simulation.current_date.date()} (Step {step_count}) - Scaled Prices:")
            print(current_data_window)
            print_dataframe_debugging_info(current_data_window, name=f"Data Window at Date: {simulation.current_date.date()} (Step {step_count}) - Scaled Prices")
        else:
            print("Simulation ended within loop.")
            break

    print("\n--- Simulation End ---")
    print(f"Is simulation finished? {simulation.is_simulation_finished}")
    print(f"Total steps taken: {step_count}")
    print(f"Current portfolio cash: ${simulation.portfolio.cash:.2f}")
    print(f"Current portfolio holdings: {simulation.portfolio.holdings}")