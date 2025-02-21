class SimulationState:
    """
    Encapsulates the state of a simulation at a given time step.

    Includes the current data window, the portfolio, and the current date.
    """
    def __init__(self, data_window, portfolio, current_date):
        """
        Initializes a SimulationState object.

        Args:
            data_window (pd.DataFrame): The data window for the current step.
            portfolio (Portfolio): The current Portfolio object.
            current_date (pd.Timestamp): The current simulation date.
        """
        self.data_window = data_window
        self.portfolio = portfolio
        self.current_date = current_date

    def __repr__(self):
        """
        Returns a string representation of the SimulationState object.
        """
        return f"SimulationState(date={self.current_date.strftime('%Y-%m-%d') if self.current_date else None}, portfolio_value=${self.portfolio.latest_value():.2f})"