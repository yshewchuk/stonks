import pandas as pd
from model.portfolio import Portfolio

class SimulationState:
    """
    Encapsulates the state of a simulation at a given time step.

    Includes the current data window, the portfolio, and the current date.
    """
    def __init__(self, historical_data_window, complete_simulation_data, portfolio, stock_values, current_date):
        """
        Initializes a SimulationState object.

        Args:
            historical_data_window (pd.DataFrame): The historical data that can be used to train or execute the model.
            complete_simulation_data (pd.DataFrame): The complete set of model data for the simulation, unscaled and containing additional columns which can be used for evaluation purposes.
            portfolio (Portfolio): The current Portfolio object.
            stock_values (dict): Dictionary of tickers to their current closing prices.
            current_date (pd.Timestamp): The current simulation date.
        """

        if not isinstance(historical_data_window, pd.DataFrame):
            raise ValueError("historical_data_window must be a dataframe")
        if not isinstance(complete_simulation_data, pd.DataFrame):
            raise ValueError("complete_simulation_data must be a dataframe")
        if not isinstance(portfolio, Portfolio):
            raise ValueError("portfolio must be a Portfolio object.")
        if not isinstance(stock_values, dict):
            raise ValueError("stock values must be a dictionary of tickers to closing price.")
        if not isinstance(current_date, pd.Timestamp):
            raise ValueError("current_date must be a pandas timestamp.")
        
        self.__historical_data_window = historical_data_window
        self.__complete_simulation_data = complete_simulation_data
        self.__portfolio = portfolio
        self.__stock_values = stock_values
        self.__current_date = current_date

    def __repr__(self):
        """
        Returns a string representation of the SimulationState object.
        """
        return f"SimulationState(date={self.current_date.strftime('%Y-%m-%d') if self.current_date else None}, portfolio_value=${self.portfolio.latest_value():.2f})"
    
    @property
    def historical_data_window(self):
        return self.__historical_data_window
    
    @property
    def complete_simulation_data(self):
        return self.__complete_simulation_data
    
    @property
    def portfolio(self):
        return self.__portfolio
    
    @property
    def stock_values(self):
        return self.__stock_values
    
    @property
    def current_date(self):
        return self.__current_date