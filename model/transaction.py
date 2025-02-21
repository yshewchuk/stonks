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


        self.date = date
        self.ticker = ticker
        self.transaction_type = transaction_type
        self.quantity = quantity
        self.price = price
        self.transaction_cost = transaction_cost
        self.transaction_proceeds = transaction_proceeds
        self.total_cost = total_cost
        self.profit = profit

    def __repr__(self):
        """
        Provides a readable string representation of the Transaction object.
        """
        if self.transaction_type == 'buy':
            return (f"Transaction(date={self.date.strftime('%Y-%m-%d')}, ticker='{self.ticker}', "
                    f"type='buy', quantity={self.quantity}, price={self.price:.2f}, total_cost={self.total_cost:.2f})")
        elif self.transaction_type == 'sell':
            return (f"Transaction(date={self.date.strftime('%Y-%m-%d')}, ticker='{self.ticker}', "
                    f"type='sell', quantity={self.quantity}, price={self.price:.2f}, proceeds={self.transaction_proceeds:.2f}, profit={self.profit:.2f})")
        return super().__repr__() # Fallback for any other cases
