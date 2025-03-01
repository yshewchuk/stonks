import pandas as pd

class Transaction:
    """
    Represents a financial transaction (buy or sell).
    """
    def __init__(self, date, ticker, transaction_type, quantity, price, transaction_cost, transaction_proceeds=None, total_cost=None, profit=None):
        """
        Initializes a Transaction object.

        Args:
            date (pd.Timestamp): The date of the transaction.
            ticker (str): The stock ticker symbol.
            transaction_type (str): 'buy' or 'sell'.
            quantity (int): The number of shares involved in the transaction.
            price (float): The price per share at which the transaction occurred.
            transaction_cost (float): The fee associated with the transaction.
            transaction_proceeds (float, optional): Proceeds from a sell transaction (revenue after fees). Only for 'sell' transactions.
            total_cost (float, optional): Total cost of a buy transaction (cost + fees). Only for 'buy' transactions.
            profit (float, optional): Profit from a sell transaction. Only for 'sell' transactions.
        """
        if not isinstance(date, pd.Timestamp):
            raise ValueError("date must be a Pandas Timestamp.")
        if not isinstance(ticker, str):
            raise ValueError("ticker must be a string.")
        if transaction_type not in ['buy', 'sell']:
            raise ValueError("transaction_type must be 'buy' or 'sell'.")
        if not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive integer.")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("price must be a positive number.")
        if not isinstance(transaction_cost, (int, float)) or transaction_cost < 0:
            raise ValueError("transaction_cost must be a non-negative number.")
        if transaction_proceeds is not None and not isinstance(transaction_proceeds, (int, float)):
            raise ValueError("transaction_proceeds must be a number or None.")
        if total_cost is not None and not isinstance(total_cost, (int, float)):
            raise ValueError("total_cost must be a number or None.")
        if profit is not None and not isinstance(profit, (int, float)):
            raise ValueError("profit must be a number or None.")

        # Store attributes with a leading underscore to indicate they are "protected"
        self.__date = date
        self.__ticker = ticker
        self.__transaction_type = transaction_type
        self.__quantity = quantity
        self.__price = price
        self.__transaction_cost = transaction_cost
        self.__transaction_proceeds = transaction_proceeds
        self.__total_cost = total_cost
        self.__profit = profit

    @property
    def date(self):
        """Date of the transaction."""
        return self.__date

    @property
    def ticker(self):
        """Stock ticker symbol."""
        return self.__ticker

    @property
    def transaction_type(self):
        """Transaction type ('buy' or 'sell')."""
        return self.__transaction_type

    @property
    def quantity(self):
        """Number of shares involved."""
        return self.__quantity

    @property
    def price(self):
        """Price per share."""
        return self.__price

    @property
    def transaction_cost(self):
        """Transaction fee."""
        return self.__transaction_cost

    @property
    def transaction_proceeds(self):
        """Proceeds from a sell transaction (revenue after fees)."""
        return self.__transaction_proceeds

    @property
    def total_cost(self):
        """Total cost of a buy transaction (cost + fees)."""
        return self.__total_cost

    @property
    def profit(self):
        """Profit from a sell transaction."""
        return self.__profit

    def __repr__(self):
        """
        Provides a readable string representation of the Transaction object.
        """
        if self.__transaction_type == 'buy':
            return (f"Transaction(date={self.__date.strftime('%Y-%m-%d')}, ticker='{self.__ticker}', "
                    f"type='buy', quantity={self.__quantity}, price={self.__price:.2f}, total_cost={self.__total_cost:.2f})")
        elif self.__transaction_type == 'sell':
            return (f"Transaction(date={self.__date.strftime('%Y-%m-%d')}, ticker='{self.__ticker}', "
                    f"type='sell', quantity={self.__quantity}, price={self.__price:.2f}, proceeds={self.__transaction_proceeds:.2f}, profit={self.__profit:.2f})")
        return super().__repr__() # Fallback for any other cases