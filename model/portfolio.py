import pandas as pd
from model.transaction import Transaction # Import Transaction class from transaction.py

class Portfolio:
    """
    Represents a financial portfolio, managing cash, stock holdings, and portfolio value.

    Simplified holdings tracking, clarified valuation methods, refined profit calculation,
    transaction logging using Transaction class, and optimized transaction retrieval.
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
        self.__latest_valuation_date = None
        self.__value_history = {}  # Dictionary to store portfolio value history (date: value)
        self.__transaction_log = []  # **Private attribute**: Transaction log now stores Transaction objects


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
        self.__value_history[date] = portfolio_value # Record portfolio value in history
        self.__latest_valuation_date = date # Update latest valuation date
        return portfolio_value


    def latest_value(self):
        """
        Returns the portfolio value calculated for the latest valuation date.

        Returns:
            float: The total portfolio value at the latest date evaluated, or just cash if no valuation yet.
        """
        if self.__latest_valuation_date is None:
            return self.__cash # Return cash if no valuation has been set yet
        return self.__value_history[self.__latest_valuation_date]


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


    def _validate_transaction_date(self, date):
        """
        Ensures that the transaction date is chronologically valid using transaction log.
        """
        if self.__transaction_log: # Check if transaction log is not empty
            last_transaction_date = self.__transaction_log[-1].date # Get date of the last transaction in log
            if date < last_transaction_date:
                raise ValueError(f"Transaction date {date.strftime('%Y-%m-%d')} is before the last transaction date {last_transaction_date.strftime('%Y-%m-%d')}. Transactions must be ordered chronologically.")


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

        self._validate_transaction_date(date) # Validate transaction date

        total_cost_before_fees = price * quantity
        transaction_cost = total_cost_before_fees + self.TRADING_FEE

        if self.__cash < transaction_cost:
            return False  # Insufficient cash for the purchase

        self.__cash -= transaction_cost
        # Update holdings - simpler structure
        self.__holdings[ticker]['quantity'] += quantity
        self.__holdings[ticker]['total_cost'] += transaction_cost


        # Log transaction - Now using Transaction class
        transaction = Transaction(
            date=date,
            ticker=ticker,
            transaction_type='buy',
            quantity=quantity,
            price=price,
            transaction_cost=self.TRADING_FEE,
            total_cost=transaction_cost # For buy orders, we have total_cost
        )
        self.__transaction_log.append(transaction) # Store Transaction object

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

        self._validate_transaction_date(date) # Validate transaction date

        current_holdings_qty = self.get_holding_quantity(ticker) # Use getter
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


        # Log transaction - Now using Transaction class
        transaction = Transaction(
            date=date,
            ticker=ticker,
            transaction_type='sell',
            quantity=quantity,
            price=price,
            transaction_cost=self.TRADING_FEE,
            transaction_proceeds=transaction_proceeds, # For sell orders, we have proceeds and profit
            profit=profit
        )
        self.__transaction_log.append(transaction) # Store Transaction object

        return profit  # Sell order successful, return profit

    def get_transactions_on_date(self, date):
        """
        Retrieves all transactions that occurred on a specific date.
        Optimized for chronologically ordered transaction log - searching from the end.

        Args:
            date (pd.Timestamp): The date for which to retrieve transactions.

        Returns:
            list[Transaction]: A list of Transaction objects that occurred on the given date.
        """
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date must be a Pandas Timestamp.")
        transactions_on_date = []
        # Iterate in reverse order for efficiency when querying latest dates
        for txn in reversed(self.__transaction_log):
            if txn.date == date:
                transactions_on_date.append(txn)
            elif txn.date < date: # Stop searching if transaction date is before the target date
                break
        return transactions_on_date


    @property
    def transaction_log(self):
        """
        Returns a copy of the transaction log (list of Transaction objects).
        """
        return list(self.__transaction_log) # Return a copy to prevent external modification


    def copy(self):
        """
        Creates and returns a deep copy of the Portfolio object.

        Returns:
            Portfolio: A new Portfolio object with the same state as the original.
        """
        portfolio_copy = Portfolio(self.cash, self.__tickers)
        portfolio_copy.__holdings = {ticker: holding_data.copy() for ticker, holding_data in self.__holdings.items()}
        portfolio_copy.__latest_valuation_date = self.__latest_valuation_date
        portfolio_copy.__value_history = self.__value_history.copy()
        portfolio_copy.__transaction_log = list(self.transaction_log) # Create a new list of Transaction objects
        return portfolio_copy



# Example Usage (for testing and demonstration)
if __name__ == '__main__':
    # Define tickers and starting cash
    tickers = ['AAPL', 'GOOG', 'MSFT']
    starting_cash = 100000

    # Create a Portfolio instance
    portfolio = Portfolio(starting_cash, tickers)

    # Simulate buying stocks
    date1 = pd.to_datetime('2024-01-02')
    portfolio.buy(date1, 'AAPL', price=170, quantity=10)
    date2 = pd.to_datetime('2024-01-03')
    portfolio.buy(date2, 'GOOG', price=2700, quantity=2)

    # Simulate selling stocks
    date3 = pd.to_datetime('2024-01-05')
    portfolio.sell(date3, 'AAPL', price=180, quantity=5)


    # Get transactions for date3
    transactions_date3 = portfolio.get_transactions_on_date(date3)
    print(f"\n--- Transactions on {date3.date()} ---")
    for txn in transactions_date3:
        print(txn)

    print(f"\n--- Full Transaction Log ---")
    for txn in portfolio.transaction_log:
        print(txn) # Transaction objects will be printed using their __repr__ method

    # Calculate portfolio value with example stock prices
    current_stock_prices = {'AAPL': 185, 'GOOG': 3000,  'MSFT': 310}
    date4 = pd.to_datetime('2024-01-07')
    portfolio.update_valuations(date4, current_stock_prices)
    print(f"\n--- Portfolio Value on {date4.date()} ---")
    print(f"Portfolio Value: ${portfolio.latest_value():.2f}")

    # Test chronological order enforcement - should raise ValueError
    try:
        portfolio.buy(pd.to_datetime('2024-01-04'), 'MSFT', price=300, quantity=10) # Date before last transaction (2024-01-07 from update_valuations, but conceptually 2024-01-05 from sell)
    except ValueError as e:
        print(f"\n--- Chronological Order Test (Expected Error) ---")
        print(f"Error caught: {e}")
    else:
        print("\n--- Chronological Order Test (Error NOT Caught - PROBLEM!) ---")
        print("Expected ValueError was not raised when adding out-of-order transaction.")