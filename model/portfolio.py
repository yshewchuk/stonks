class Portfolio:
    """
    Represents a financial portfolio, managing cash, stock holdings, and portfolio value.

    This class allows for simulating buying and selling stocks, tracking cash balance,
    stock holdings, and calculating the total portfolio value based on current stock prices.
    """

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
        self.__holdings = {ticker: 0 for ticker in tickers} # Initialize holdings for each ticker to 0
        self.__tickers = list(tickers) # Store tickers as a list for consistent iteration
        self.latest_valuation = {ticker: None for ticker in tickers} # Initialize last valuations to None

        self.TRADING_FEE = 20.0  # Define a constant for the trading fee

    @property
    def cash(self):
        """
        Returns the current cash balance of the portfolio.

        Returns:
            float: The current cash balance.
        """
        return self.__cash

    @property
    def holdings(self):
        """
        Returns a dictionary of current stock holdings.

        Returns:
            dict: A dictionary where keys are ticker symbols (str) and values are the number of shares held (int).
        """
        return self.__holdings

    def update_valuations(self, stock_values):
        """
        Updates the last known valuations of each held ticker.

        Args:
            stock_values (dict): Dictionary of tickers to their current closing prices.
        """
        if not isinstance(stock_values, dict):
            raise ValueError("stock_values must be a dictionary.")
        for ticker in self.__tickers:
            if ticker not in stock_values:
                raise ValueError(f"stock_values dictionary is missing price for ticker: '{ticker}': {stock_values}.")
            if not isinstance(stock_values[ticker], (int, float)) or stock_values[ticker] < 0:
                raise ValueError(f"Price for ticker '{ticker}' in stock_values must be a non-negative number.")
            
        for ticker, valuation in stock_values.items():
            if ticker in self.tickers: # Ensure we only update for tickers in our portfolio
                self.latest_valuation[ticker] = valuation

    def buy(self, ticker, price, quantity):
        """
        Simulates buying a quantity of stock shares.

        This method attempts to purchase the specified quantity of shares for the given ticker
        at the provided price. It checks if there is sufficient cash in the portfolio to make the purchase
        (including trading fees). If the purchase is successful, the cash balance is reduced, and the
        holding for the ticker is increased.

        Args:
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

        total_cost = price * quantity
        transaction_cost = total_cost + self.TRADING_FEE

        if self.__cash < transaction_cost:
            return False  # Insufficient cash for the purchase

        self.__cash -= transaction_cost
        self.__holdings[ticker] += quantity
        return True  # Buy order successful

    def sell(self, ticker, price, quantity):
        """
        Simulates selling a quantity of stock shares.

        This method attempts to sell the specified quantity of shares for the given ticker
        at the provided price. It checks if the portfolio holds sufficient shares of the ticker
        to fulfill the sell order. If the sell order is successful, the cash balance is increased
        (after deducting trading fees), and the holding for the ticker is decreased.

        Args:
            ticker (str): The stock ticker symbol to sell.
            price (float or int): The price per share of the stock.
            quantity (int): The number of shares to sell.

        Returns:
            bool: True if the sell order was successful (sufficient holdings), False otherwise.
        """
        if ticker not in self.__tickers:
            raise ValueError(f"Ticker '{ticker}' is not valid for this portfolio. Valid tickers are: {self.__tickers}")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("Price must be a positive number.")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")

        if self.__holdings[ticker] < quantity:
            return False  # Insufficient holdings to sell

        transaction_revenue = price * quantity
        transaction_proceeds = transaction_revenue - self.TRADING_FEE

        self.__cash += transaction_proceeds
        self.__holdings[ticker] -= quantity
        return True  # Sell order successful

    def value(self, stock_values=None):
        """
        Calculates the total value of the portfolio, including cash and stock holdings.

        Args:
            stock_values (dict): A dictionary where keys are ticker symbols (str) and values are the current stock prices (float or int).

        Returns:
            float: The total portfolio value (cash + value of all stock holdings).
        """
        if stock_values is None:
            stock_values = self.latest_valuation

        if not isinstance(stock_values, dict):
            raise ValueError("stock_values must be a dictionary.")
        for ticker in self.__tickers:
            if ticker not in stock_values:
                raise ValueError(f"stock_values dictionary is missing price for ticker: '{ticker}': {stock_values}.")
            if not isinstance(stock_values[ticker], (int, float)) or stock_values[ticker] < 0:
                raise ValueError(f"Price for ticker '{ticker}' in stock_values must be a non-negative number.")

        stock_value_sum = 0.0  # Initialize sum for stock values
        for ticker in self.__tickers:
            stock_value_sum += self.__holdings[ticker] * stock_values[ticker]

        return self.__cash + stock_value_sum


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
    buy_successful_aapl = portfolio.buy('AAPL', price=170, quantity=10)
    buy_successful_goog = portfolio.buy('GOOG', price=2700, quantity=2)
    buy_failed_msft = portfolio.buy('MSFT', price=300, quantity=500) # Should fail due to insufficient cash

    print("\n--- After Buying ---")
    print(f"Buy AAPL successful: {buy_successful_aapl}")
    print(f"Buy GOOG successful: {buy_successful_goog}")
    print(f"Buy MSFT successful: {buy_failed_msft}") # Expected: False
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Holdings: {portfolio.holdings}")

    # Simulate selling stocks
    sell_successful_aapl = portfolio.sell('AAPL', price=175, quantity=5)
    sell_failed_goog = portfolio.sell('GOOG', price=2800, quantity=3) # Should fail due to insufficient holdings

    print("\n--- After Selling ---")
    print(f"Sell AAPL successful: {sell_successful_aapl}")
    print(f"Sell GOOG successful: {sell_failed_goog}") # Expected: False
    print(f"Cash: ${portfolio.cash:.2f}")
    print(f"Holdings: {portfolio.holdings}")

    # Calculate portfolio value with example stock prices
    current_stock_prices = {'AAPL': 175, 'GOOG': 2750, 'MSFT': 305}
    portfolio_value = portfolio.value(current_stock_prices)
    print(f"\n--- Portfolio Value ---")
    print(f"Portfolio Value: ${portfolio_value:.2f}")