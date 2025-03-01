from sklearn.preprocessing import MinMaxScaler
from model.portfolio import Portfolio
from model.simulation_state import SimulationState # Import SimulationState
from utils.dataframe import print_dataframe_debugging_info
import pandas as pd
import numpy as np

class Simulation:
    """
    Manages data and portfolio for a single trading simulation run.

    This class encapsulates the data for a specific simulation period, manages a Portfolio object,
    and provides methods to initialize the simulation, advance through time steps, and retrieve
    simulation state at each step.
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
            for ticker in self.portfolio.tickers():
                self.valuations[date][ticker] = self.data.loc[date, (ticker, 'Close')]

        self.transaction_prices = self.data[list(map(lambda x: (x, 'Open'), self.portfolio.tickers()))].copy()


        self._initialize_portfolio_metrics_columns() # Setup columns for portfolio value tracking
        self._scale_price_data() # Apply price scaling
        self._calculate_initial_portfolio_value() # Calculate initial portfolio value after scaling, but using original prices for initial valuation


    def _calculate_initial_portfolio_value(self):
        """Calculates and stores the initial portfolio value at the start of the simulation."""
        first_date = self.simulation_dates[0]

        self.initial_portfolio_value = self.portfolio.update_valuations(pd.to_datetime(first_date), self.valuations[first_date]) # Pass date as Timestamp
        if self.initial_portfolio_value == 0:
            self.initial_portfolio_value = self.portfolio.cash


    def _scale_price_data(self):
        """
        Scales price-related and volume data in self.data.

        Price-related columns are scaled to percentage difference from the first day's Open price.
        Volume columns are scaled using MinMaxScaler fitted on the *first window* of data.
        """
        first_date = self.simulation_dates[0]
        price_column_tags = ['Open', 'High', 'Low', 'Close', 'MA', 'Hi', 'Lo']
        self.volume_scalers = {} # Store scalers for volume columns

        initial_window_size = 60 # Define your initial window size (e.g., 60 days)
        initial_dates_for_scaling = self.simulation_dates[:min(initial_window_size, len(self.simulation_dates))] # Get dates for initial window

        for ticker in self.portfolio.tickers():
            # --- Volume Scaling (Fit scaler on initial window data) ---
            initial_volume_data = self.data.loc[initial_dates_for_scaling, (ticker, 'Volume')].values.reshape(-1, 1) # Get volume data for the initial window
            scaler = MinMaxScaler()
            scaler.fit(initial_volume_data) # FIT SCALER ONLY ON INITIAL WINDOW DATA
            self.volume_scalers[ticker] = scaler # Store fitted scaler

            # Apply the SAME scaler to the ENTIRE Volume column for all dates
            volume_data_full = self.data[(ticker, 'Volume')].values.reshape(-1, 1)
            self.data[(ticker, 'Volume')] = scaler.transform(volume_data_full)

            # --- Price Scaling (remains as before - relative to first date's Open) ---
            initial_open_price = self.data.loc[first_date, (ticker, 'Open')]
            if initial_open_price == 0:
                initial_open_price = 1.0
            for col in self.data.columns:
                if col[0] == ticker and any(tag in col[1] for tag in price_column_tags):
                    self.data[col] = (self.data[col] - initial_open_price) / initial_open_price


    def _initialize_portfolio_metrics_columns(self):
        """ ... """
        self.data['Relative Value'] = 0.0
        self.data['Cash Percent of Value'] = 0.0
        for ticker in self.portfolio.tickers():
            self.data[(ticker, 'Holdings Percent of Value')] = 0.0 # MultiIndex column


    def _calculate_portfolio_metrics(self, current_date):
        """
        Calculates and updates portfolio metrics for a given date in self.data.
        ... (rest of docstring) ...
        """
        portfolio_value = self.portfolio.update_valuations(pd.to_datetime(current_date), self.valuations[current_date]) # Pass date as Timestamp
        relative_value = portfolio_value / self.initial_portfolio_value
        self.data.loc[current_date, 'Relative Value'] = relative_value
        self.data.loc[current_date, 'Cash Percent of Value'] = self.portfolio.cash / portfolio_value if portfolio_value != 0 else 0
        for ticker in self.portfolio.tickers():
            ticker_holding_value = self.portfolio.get_holding_quantity(ticker) * self.valuations[current_date][ticker] # Use getter
            self.data.loc[current_date, (ticker, 'Holdings Percent of Value')] = ticker_holding_value / portfolio_value if portfolio_value != 0 else 0


    def start(self, window):
        """
        Starts the simulation and returns the initial SimulationState for the model.

        Initializes portfolio metrics for the first 'window' days and returns a SimulationState
        object containing the data window, portfolio, and current date for the first 'window' steps,
        ready to be used as input for a model.

        Args:
            window (int): The number of initial time steps (days) for the data window. Must be positive.

        Returns:
            SimulationState: Object encapsulating the initial simulation state,
                             or None if initialization fails.

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

        data_window = self.get_current_data_window(window)
        if data_window is None: # Handle case where no data window could be created (e.g., no data)
            return None

        return SimulationState(data_window=data_window, portfolio=self.portfolio, stock_values=self.valuations[current_date], current_date=current_date) # Return SimulationState


    def step(self, window, orders=None):
        """
        Advances the simulation by one time step and returns the next SimulationState.

        Increments the simulation step index to the next date in the data.
        Executes any provided orders, calculates updated portfolio metrics for the new current date.
        Returns a SimulationState object containing the data window, portfolio, and current date
        for the current 'window' steps up to the new current date.

        Args:
            window (int): The number of time steps (days) for the data window. Must be positive and consistent with 'start()' window.
            orders: The set of orders to execute on the current step date (optional).

        Returns:
            SimulationState or None: SimulationState object for the current step,
                                     or None if simulation end is reached.
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

        if orders: # Check if orders were provided
            self._execute_orders(orders, current_date) # Execute orders before the end of the day

        self._calculate_portfolio_metrics(current_date)

        data_window = self.get_current_data_window(window)
        if data_window is None: # Handle case where no data window could be created
            return None

        return SimulationState(data_window=data_window, portfolio=self.portfolio, stock_values=self.valuations[current_date], current_date=current_date) # Return SimulationState


    def _execute_orders(self, orders, current_date):
        """
        Executes a list of orders at the current date's Open price.
        Assumes all orders are filled at the open price for simplicity.

        Args:
            orders (list): List of order dictionaries, each like:
                           {'ticker': 'AAPL', 'order_type': 'buy', 'quantity': 10}
            current_date (pd.Timestamp): The current simulation date.
        """
        for order in orders:
            ticker = order['ticker']
            order_type = order['order_type']
            quantity = order['quantity']

            fill_price = self.transaction_prices.loc[current_date, (ticker, 'Open')] # Get Open price for fill

            if order_type == 'buy':
                self.portfolio.buy(pd.to_datetime(current_date), ticker, fill_price, quantity) # Pass date as Timestamp
            elif order_type == 'sell':
                self.portfolio.sell(pd.to_datetime(current_date), ticker, fill_price, quantity) # Pass date as Timestamp
            else:
                raise ValueError(f"Invalid order type: {order_type}. Must be 'buy' or 'sell'.")


    def get_current_data_window(self, window):
        """
        Returns a DataFrame slice representing the current data window.
        ... (rest of get_current_data_window docstring) ...
        """
        if self.current_step_index < 0 or self.current_step_index >= len(self.simulation_dates):
            return None

        start_index = max(0, self.current_step_index - window + 1)
        start_date = self.simulation_dates[start_index]
        current_date = self.simulation_dates[self.current_step_index]

        data_window = self.data.loc[start_date : current_date]
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
    dates = pd.to_datetime(['2012-08-01', '2012-08-02', '2012-08-03', '2012-08-06', '2012-08-07', '2012-08-08', '2012-08-09', '2012-08-10']) # Using same dates as sample data
    tickers = ['AAPL', 'MSFT']
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
    initial_state = simulation.start(window_size) # Get SimulationState
    if initial_state is not None:
        print("\nInitial Data Window:")
        print(initial_state.data_window.tail())
        print("\nInitial Simulation State:")
        print(initial_state) # Print SimulationState object

        print("\n--- Simulation Step with Orders ---")
        # Example orders to place at the next step (after the initial window)
        orders_day1 = [
            {'ticker': 'AAPL', 'order_type': 'buy', 'quantity': 10},
            {'ticker': 'MSFT', 'order_type': 'buy', 'quantity': 5}
        ]

        next_state_step1 = simulation.step(window_size, orders=orders_day1) # Get next SimulationState
        if next_state_step1 is not None:
            print("\nData Window after Step 1 (with orders):")
            print(next_state_step1.data_window.tail())
            print("\nSimulation State after Step 1 (with orders):")
            print(next_state_step1) # Print SimulationState object
            print(f"Portfolio cash after step 1: ${next_state_step1.portfolio.cash:.2f}") # Access portfolio from SimulationState
            # Corrected lines using get_holding_quantity()
            print(f"AAPL holdings after step 1: {simulation.portfolio.get_holding_quantity('AAPL')}") # Use getter!
            print(f"MSFT holdings after step 1: {simulation.portfolio.get_holding_quantity('MSFT')}") # Use getter!
        else:
            print("Simulation ended at step 1.")


    print("\n--- Simulation Steps (Running to End, No Orders) ---")
    step_count = 1 # Start from step 1 as we already did step 1 with orders
    while True:
        orders_day2 = [
            {'ticker': 'AAPL', 'order_type': 'sell', 'quantity': 1},
            {'ticker': 'MSFT', 'order_type': 'sell', 'quantity': 1}
        ]
        next_state = simulation.step(window_size, orders_day2) # Get next SimulationState
        if next_state is not None:
            step_count += 1
            print(f"\nData Window at Date: {next_state.current_date.date()} (Step {step_count}) - No Orders:")
            print(next_state.data_window.tail())
            print(f"Simulation State at Date: {next_state.current_date.date()} (Step {step_count}) - No Orders:")
            print(next_state) # Print SimulationState object
        else:
            print("Simulation ended within loop.")
            break

    print("\n--- Simulation End ---")
    print(f"Is simulation finished? {simulation.is_simulation_finished}")
    print(f"Total steps taken: {step_count}")
    print(f"Current portfolio cash: ${simulation.portfolio.cash:.2f}") # Access final portfolio from simulation object
    # Corrected lines using get_holding_quantity() for final holdings printout
    print(f"Current portfolio holdings - AAPL: {simulation.portfolio.get_holding_quantity('AAPL')}") # Use getter!
    print(f"Current portfolio holdings - MSFT: {simulation.portfolio.get_holding_quantity('MSFT')}") # Use getter!