import pandas as pd

class Portfolio:
    """
    Represents a financial portfolio, managing cash, stock holdings, and portfolio value.

    Simplified holdings tracking, clarified valuation methods, and refined profit calculation.
    """

    TRADING_FEE = 20.0  # Define a constant for the trading fee

    def __init__(self, starting_cash, tickers):
        """
        Initializes a Portfolio object.

        Args:
            starting_cash (float or int): The initial cash balance for the portfolio.
            tickers (list of str): A list of stock ticker symbols that the portfolio can hold.
        """
        if not isinstance(starting_cash, (int, float)) or starting_cash < 0:
            raise ValueError("Starting cash must be a non-negative number.")
        if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
            raise ValueError("Tickers must be a list of strings.")

        self.__cash = float(starting_cash)  # Use float for cash to handle decimal values
        # __holdings now stores quantity and total_cost directly
        self.__holdings = {ticker: {'quantity': 0, 'total_cost': 0.0} for ticker in tickers}
        self.__tickers = list(tickers)  # Store tickers as a list for consistent iteration
        self.latest_valuation_date = None # Store only the latest date for which valuation is updated
        self.value_history = {}  # Dictionary to store portfolio value history (date: value)
        self.transaction_log = []  # List to store transaction logs


    @property
    def cash(self):
        """Returns the current cash balance of the portfolio."""
        return self.__cash

    def tickers(self):
        """Returns the list of tickers that this portfolio is initialized on"""
        return self.__tickers

    def get_holding_quantity(self, ticker):
        """Returns current stock holdings for a ticker (number of shares)."""
        return self.__holdings[ticker]['quantity']

    def get_average_buy_price(self, ticker):
        """
        Calculates the average buy price for a given ticker.

        Returns:
            float: The average buy price per share, or 0 if no holdings.
        """
        quantity = self.__holdings[ticker]['quantity']
        if quantity == 0:
            return 0.0
        return self.__holdings[ticker]['total_cost'] / quantity


    def update_valuations(self, date, stock_values):
        """
        Updates the last known valuations of each held ticker and records portfolio value for the date.

        Args:
            date (pd.Timestamp): The date for which valuations are updated.
            stock_values (dict): Dictionary of tickers to their current closing prices.

        Returns:
            float: The total portfolio value on the given date.
        """
        if not isinstance(stock_values, dict):
            raise ValueError("stock_values must be a dictionary.")
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date must be a Pandas Timestamp.")
        for ticker in self.__tickers:
            if ticker not in stock_values:
                raise ValueError(f"stock_values dictionary is missing price for ticker: '{ticker}': {stock_values}.")
            if not isinstance(stock_values[ticker], (int, float)) or stock_values[ticker] < 0:
                raise ValueError(f"Price for ticker '{ticker}' in stock_values must be a non-negative number.")

        portfolio_value = self.value(stock_values) # Calculate portfolio value with provided stock_values
        self.value_history[date] = portfolio_value # Record portfolio value in history
        self.latest_valuation_date = date # Update latest valuation date
        return portfolio_value


    def latest_value(self):
        """
        Returns the portfolio value calculated for the latest valuation date.

        Returns:
            float: The total portfolio value at the latest date evaluated, or just cash if no valuation yet.
        """
        if self.latest_valuation_date is None:
            return self.__cash # Return cash if no valuation has been set yet
        return self.value_history[self.latest_valuation_date]


    def value(self, stock_values):
        """
        Calculates the total value of the portfolio, including cash and stock holdings,
        using provided or latest stock valuations.

        Args:
            stock_values (dict, optional): A dictionary where keys are ticker symbols (str) and values are the current stock prices (float or int).
                                        If None, uses the latest valuations.

        Returns:
            float: The total portfolio value (cash + value of all stock holdings).
        """
        if stock_values is None or not isinstance(stock_values, dict):
            raise ValueError("stock_values must be a dictionary.")
        for ticker in self.__tickers:
            if ticker not in stock_values:
                raise ValueError(f"stock_values dictionary is missing price for ticker: '{ticker}': {stock_values}.")
            if not isinstance(stock_values[ticker], (int, float)) or stock_values[ticker] < 0:
                raise ValueError(f"Price for ticker '{ticker}' in stock_values must be a non-negative number.")

        stock_value_sum = 0.0  # Initialize sum for stock values
        for ticker in self.__tickers:
            stock_value_sum += self.get_holding_quantity(ticker) * stock_values[ticker] # Use getter for holdings

        return self.__cash + stock_value_sum


    def buy(self, date, ticker, price, quantity):
        """
        Simulates buying a quantity of stock shares, updates holdings with transaction details,
        and logs the transaction.

        Args:
            date (pd.Timestamp): The date of the transaction.
            ticker (str): The stock ticker symbol to buy.
            price (float or int): The price per share of the stock.
            quantity (int): The number of shares to buy.

        Returns:
            bool: True if the buy order was successful (sufficient cash), False otherwise.
        """
        if ticker not in self.__tickers:
            raise ValueError(f"Ticker '{ticker}' is not valid for this portfolio. Valid tickers are: {self.__tickers}")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("Price must be a positive number.")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date must be a Pandas Timestamp.")

        total_cost_before_fees = price * quantity
        transaction_cost = total_cost_before_fees + self.TRADING_FEE

        if self.__cash < transaction_cost:
            return False  # Insufficient cash for the purchase

        self.__cash -= transaction_cost
        # Update holdings - simpler structure
        self.__holdings[ticker]['quantity'] += quantity
        self.__holdings[ticker]['total_cost'] += transaction_cost


        # Log transaction
        self.transaction_log.append({
            'date': date,
            'ticker': ticker,
            'transaction_type': 'buy',
            'quantity': quantity,
            'price': price,
            'transaction_cost': self.TRADING_FEE,
            'total_cost': transaction_cost
        })
        return True  # Buy order successful


    def sell(self, date, ticker, price, quantity):
        """
        Simulates selling a quantity of stock shares, updates holdings, calculates profit,
        and logs the transaction.

        Args:
            date (pd.Timestamp): The date of the transaction.
            ticker (str): The stock ticker symbol to sell.
            price (float or int): The price per share of the stock.
            quantity (int): The number of shares to sell.

        Returns:
            float: Profit from the sell transaction.
        """
        if ticker not in self.__tickers:
            raise ValueError(f"Ticker '{ticker}' is not valid for this portfolio. Valid tickers are: {self.__tickers}")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("Price must be a positive number.")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date must be a Pandas Timestamp.")


        current_holdings_qty = self.get_holding_quantity(ticker)
        if current_holdings_qty < quantity:
            raise ValueError(f"Insufficient holdings of {ticker} to sell {quantity} shares. Holdings: {current_holdings_qty}, Sell Quantity: {quantity}")

        transaction_revenue_before_fees = price * quantity
        transaction_proceeds = transaction_revenue_before_fees - self.TRADING_FEE
        self.__cash += transaction_proceeds

        avg_buy_price = self.get_average_buy_price(ticker)
        profit = transaction_proceeds - (avg_buy_price * quantity) # Profit calculation adjusted

        # Update holdings - simpler structure
        self.__holdings[ticker]['quantity'] -= quantity
        cost_reduction = avg_buy_price * quantity # Cost basis reduction based on average buy price
        self.__holdings[ticker]['total_cost'] -= cost_reduction


        # Log transaction
        self.transaction_log.append({
            'date': date,
            'ticker': ticker,
            'transaction_type': 'sell',
            'quantity': quantity,
            'price': price,
            'transaction_cost': self.TRADING_FEE,
            'transaction_proceeds': transaction_proceeds,
            'profit': profit
        })

        return profit  # Sell order successful, return profit

    def copy(self):
        """
        Creates and returns a deep copy of the Portfolio object.

        Returns:
            Portfolio: A new Portfolio object with the same state as the original.
        """
        portfolio_copy = Portfolio(self.cash, self.__tickers)
        portfolio_copy.__holdings = {ticker: holding_data.copy() for ticker, holding_data in self.__holdings.items()} # Deep copy holdings dicts
        portfolio_copy.latest_valuation_date = self.latest_valuation_date # Copy latest_valuation_date
        portfolio_copy.value_history = self.value_history.copy()
        portfolio_copy.transaction_log = [log_entry.copy() for log_entry in self.transaction_log] # Deep copy transaction logs
        return portfolio_copy



# Example Usage (for testing and demonstration)
if __name__ == '__main__':
    # Define tickers and starting cash
    tickers = ['AAPL', 'GOOG', 'MSFT']
    starting_cash = 100000

    # Create a Portfolio instance
    portfolio = Portfolio(starting_cash, tickers)

    print("--- Initial Portfolio ---")
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Holdings: {portfolio.holdings}")
    print(f"Total Value (assuming initial prices are $0): ${portfolio.value({ticker: 0 for ticker in tickers}):.2f}")

    # Simulate buying stocks
    date1 = pd.to_datetime('2024-01-02')
    buy_successful_aapl = portfolio.buy(date1, 'AAPL', price=170, quantity=10)
    date2 = pd.to_datetime('2024-01-03')
    buy_successful_goog = portfolio.buy(date2, 'GOOG', price=2700, quantity=2)


    print("\n--- After Buying ---")
    print(f"Buy AAPL successful: {buy_successful_aapl}")
    print(f"Buy GOOG successful: {buy_successful_goog}")
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Holdings: {portfolio.holdings}")
    print(f"AAPL Average Buy Price: ${portfolio.get_average_buy_price('AAPL'):.2f}")
    print(f"GOOG Average Buy Price: ${portfolio.get_average_buy_price('GOOG'):.2f}")


    # Simulate selling stocks
    date4 = pd.to_datetime('2024-01-05')
    sell_successful_aapl_profit = portfolio.sell(date4, 'AAPL', price=180, quantity=5)
    date5 = pd.to_datetime('2024-01-06')
    try:
        sell_failed_goog = portfolio.sell(date5, 'GOOG', price=2800, quantity=3) # Should raise error
    except ValueError as e:
        sell_failed_goog_error = e
    else:
        sell_failed_goog_error = None


    print("\n--- After Selling ---")
    print(f"Sell AAPL successful: Profit from AAPL sell: ${sell_successful_aapl_profit:.2f}")
    print(f"Sell GOOG error (expected): {sell_failed_goog_error}")
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Holdings: {portfolio.holdings}")
    print(f"AAPL Average Buy Price: ${portfolio.get_average_buy_price('AAPL'):.2f}") # Should remain the same or adjust if more buys happened
    print(f"GOOG Average Buy Price: ${portfolio.get_average_buy_price('GOOG'):.2f}") # Should remain the same


    # Calculate portfolio value with example stock prices
    current_stock_prices = {'AAPL': 185, 'GOOG': 2850, 'MSFT': 310}
    date6 = pd.to_datetime('2024-01-07')
    portfolio_value = portfolio.update_valuations(date6, current_stock_prices) # Update and get value
    print(f"\n--- Portfolio Value ---")
    print(f"Portfolio Value on {date6.date()} (updated and returned): ${portfolio_value:.2f}")
    print(f"Latest Portfolio Value (using latest_value()): ${portfolio.latest_value():.2f}") # Get value using latest known prices
    print(f"Value History: {portfolio.value_history}")
    print(f"\n--- Transaction Log ---")
    for log_entry in portfolio.transaction_log:
        print(log_entry)