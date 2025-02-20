import pandas as pd
import numpy as np
from model.simulation import Simulation
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler

class TrainingAgent:
    """
    Manages the training of a trading model using Simulation objects.
    """

    def __init__(self, simulations, model=None, model_params=None, reward_function=None, reward_weights=None, optimizer=None):
        """
        Initializes the TrainingAgent.

        Args:
            simulations (list): List of Simulation objects to train on.
            model (optional): The trading model instance. If None, a default model will be created.
            model_params (dict, optional): Parameters for the model. If None, default parameters will be used.
            reward_function (callable, optional): Function to calculate reward. If None, a default reward function will be used.
            reward_weights (dict, optional): Weights for different components of the reward function.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer for training. If None, Adam optimizer is used.
        """
        if not isinstance(simulations, list) or not all(isinstance(sim, Simulation) for sim in simulations):
            raise ValueError("simulations must be a list of Simulation objects.")
        self.simulations = simulations

        self.model_params = model_params if model_params is not None else self._get_default_model_params()
        self.model = model if model is not None else self._create_default_model(self.model_params) # Use provided model or create default
        self.reward_function = reward_function if reward_function is not None else self._default_reward_function
        self.reward_weights = reward_weights if reward_weights is not None else self._get_default_reward_weights()
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam() # Default optimizer


    def _get_default_model_params(self):
        """Returns default LSTM model parameters for buy/sell outputs."""
        sample_simulation = self.simulations[0] # Assume simulations have same tickers and features
        n_tickers = len(sample_simulation.portfolio.holdings)
        n_features_per_ticker = 0
        for col in sample_simulation.data.columns:
            if col[0] == list(sample_simulation.portfolio.holdings.keys())[0]: # Just check the first ticker
                n_features_per_ticker += 1
        n_features_total = n_features_per_ticker * n_tickers
        n_output_total =  2 * n_tickers # Output for each ticker: [buy_percentage, sell_percentage]

        return {
            'n_steps': 60, # Example window size, adjust as needed
            'n_features_total': n_features_total,
            'n_units': 64,      # LSTM units
            'n_output_total': n_output_total
        }


    def _create_default_model(self, model_params):
        """Creates the default LSTM model using build_model_multi_output function."""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.model_params.n_steps, self.model_params.n_features_total)),
            tf.keras.layers.LSTM(units=self.model_params.n_units, return_sequences=True),
            tf.keras.layers.LSTM(units=self.model_params.n_units),
            tf.keras.layers.Dense(units=self.model_params.n_units//2, activation='relu'),
            tf.keras.layers.Dense(units=self.model_params.n_output_total, activation='sigmoid') # Sigmoid for percentage outputs
        ])


    def _default_reward_function(self, simulation):
        """
        Default reward function based on portfolio value change.
        You can customize this to incorporate more sophisticated metrics ( Sharpe ratio, etc.)
        """
        current_portfolio_value = simulation.portfolio.value() # Use valuations for reward calculation
        previous_date_index = simulation.simulation_dates.index(simulation.current_date) - 1
        if previous_date_index >= 0:
            previous_portfolio_value = simulation.portfolio.value() # Use valuations for reward calculation
            reward = current_portfolio_value - previous_portfolio_value
        else:
            reward = 0 # No reward for the first step

        return reward


    def _get_default_reward_weights(self):
        """
        Returns default reward weights.
        This is a placeholder. You might want to weight different components of your reward.
        """
        return {'portfolio_value_change': 1.0} # Example: weight for portfolio value change


    def _model_output_to_orders(self, model_output, tickers):
        """
        Converts the model's output (buy/sell percentages) to a list of orders.

        Args:
            model_output (tf.Tensor): Model output tensor of shape (1, n_output_total).
            tickers (list): List of tickers corresponding to the model outputs.
            current_cash (float): Portfolio's current cash balance.
            current_holdings (dict): Portfolio's current stock holdings.
            last_valuations (dict): Portfolio's last known valuations (closing prices).

        Returns:
            list: List of order dictionaries.
        """
        probabilities = model_output.numpy().flatten() # Flatten to 1D array
        orders = []
        num_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            buy_percentage = probabilities[i * 2]      # Buy percentage is at even indices
            sell_percentage = probabilities[i * 2 + 1] # Sell percentage is at odd indices

            # --- Buy Orders ---
            cash_to_use_for_buy = self.portfolio.cash() * buy_percentage
            if cash_to_use_for_buy > 0:
                buy_quantity = np.round(cash_to_use_for_buy / self.portfolio.latest_valuations[ticker]) # Calculate buy quantity based on available cash and open price
                if buy_quantity > 0:
                    orders.append({'ticker': ticker, 'order_type': 'buy', 'quantity': buy_quantity})
                    print(f"  - Ticker: {ticker}, Buy %: {buy_percentage:.2f}, Cash to use: ${cash_to_use_for_buy:.2f}, Quantity: {buy_quantity}")


            # --- Sell Orders ---
            quant_to_sell = np.round(self.portfolio.holdings[ticker] * sell_percentage) # Quantity to sell
            if quant_to_sell > 0:
                orders.append({'ticker': ticker, 'order_type': 'sell', 'quantity': quant_to_sell})
                print(f"  - Ticker: {ticker}, Sell %: {sell_percentage:.2f}, Quantity to sell: ${quant_to_sell:.2f}")

        return orders


    def train_model(self, window_size=60, epochs=1):
        """
        Trains the model using the provided simulations.

        Args:
            window_size (int): Size of the data window passed to the model.
            epochs (int): Number of times to iterate through the entire set of simulations.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        print(f"Starting model training for {epochs} epochs using window size {window_size}...")

        X_train_all_simulations = [] # Collect training data across simulations
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            for sim_index, simulation in enumerate(self.simulations):
                print(f"\n-- Simulation {sim_index+1}/{len(self.simulations)} --")
                initial_data_window = simulation.start(window_size=window_size) # Start each simulation

                if initial_data_window is None:
                    print(f"Warning: No initial data window for simulation {sim_index+1}. Skipping.")
                    continue # Skip to the next simulation if no data window

                step_count = 0
                global_index = 0 # Index to track position in the simulation data
                simulation_history_data = [] # Store data windows and rewards for the entire simulation

                while True:
                    # 1. Prepare Input Data for Model
                    current_data_window = simulation.get_current_data_window(window_size)
                    if current_data_window is None:
                        break # End of simulation

                    X_current = np.array([current_data_window.values]) # Shape: (1, window_size, n_features) - adjust as needed for your model
                    tickers = [col[0] for col in current_data_window.columns.levels[0]] # Get tickers in the current window

                    # 2. Model Prediction
                    with tf.GradientTape() as tape_outer: # Outer tape for gradient calculation across day
                        orders_output = self.model(X_current) # Get model output (probabilities)
                        orders = self._model_output_to_orders(orders_output, tickers) # Convert model output to order format

                        # 3. Simulation Step (execute orders and get next data window)
                        next_data_window = simulation.step(window_size=window_size, orders=orders) # Pass orders to step()

                        if next_data_window is not None:
                            step_count += 1
                            # 4. Reward Calculation
                            reward = self.reward_function(simulation) # Calculate reward based on simulation state

                            # Store data and reward for this step in simulation history
                            simulation_history_data.append({'data_window': X_current, 'reward': reward})

                            if step_count % 10 == 0: # Print progress every 10 steps
                                print(f"  - Simulation {sim_index+1}, Step {step_count}, Date: {simulation.current_date.date()}, Reward: {reward:.2f}, Portfolio Value: ${simulation.portfolio.value(simulation.valuations[simulation.current_date]):.2f}")
                            global_index += 1 # Increment global index for next step

                        else:
                            print(f"  Simulation {sim_index+1} ended after {step_count} steps. Final Portfolio Value: ${simulation.portfolio.value(simulation.valuations[simulation.current_date]):.2f}")
                            break # End of simulation


                # --- Epoch-based Model Update (Simplified Reward-Based Learning) ---
                # Process collected history data for the entire simulation to update model once per simulation (epoch).
                if simulation_history_data: # Only train if there was simulation data
                    print(f"  - Training model on simulation {sim_index+1} history...")
                    simulation_X_train = np.concatenate([item['data_window'] for item in simulation_history_data], axis=0) # Stack data windows
                    simulation_rewards = np.array([item['reward'] for item in simulation_history_data]) # Rewards

                    with tf.GradientTape() as tape_inner: # Inner tape for loss calculation - Calculate for entire simulation
                        output_probs_tape = self.model(simulation_X_train) # Get model output for the entire simulation
                        target_probs = np.zeros_like(output_probs_tape) # Initialize target probabilities for the simulation

                        # Assign target probabilities based on rewards for each step in the simulation
                        for step_idx, reward_val in enumerate(simulation_rewards):
                            if reward_val > 0:
                                target_probs[step_idx] = np.ones_like(output_probs_tape[step_idx]) * 0.9 # Encourage action for positive reward
                            else:
                                target_probs[step_idx] = np.zeros_like(output_probs_tape[step_idx]) # Discourage for negative reward

                        loss = tf.keras.losses.MeanSquaredError()(target_probs, output_probs_tape) # MSE loss for entire simulation

                    gradients = tape_inner.gradient(loss, self.model.trainable_variables) # Gradients for simulation loss
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # Apply gradients

                    print(f"    - Simulation {sim_index+1} training completed. Loss: {loss.numpy():.4f}")
                else:
                    print(f"  - No training data for simulation {sim_index+1} (simulation history empty).")


        print("\n--- Training Complete ---")



# Example Usage (for testing)
if __name__ == '__main__':
    # --- Sample Data and Portfolio Setup (as in previous example) ---
    dates = pd.to_datetime(['2012-08-01', '2012-08-02', '2012-08-03', '2012-08-06', '2012-08-07', '2012-08-08', '2012-08-09', '2012-08-10'] * 10) # Extended dates for more steps
    dates.sort_values(inplace=True) # Ensure dates are sorted
    dates = dates.unique() # Get unique dates if duplicates were introduced
    tickers = ['AAPL', 'MSFT', 'GOOG'] # Added GOOG to increase output size
    columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']], names=['Ticker', 'OHLCV'])
    data_values = np.random.rand(len(dates), len(tickers) * 7)
    sample_data = pd.DataFrame(data_values, index=dates, columns=columns)
    sample_data.index.name = 'Date'

    initial_cash = 100000
    sample_portfolio = Portfolio(initial_cash, tickers)

    # --- Create Multiple Simulations ---
    num_simulations = 2
    simulations_list = []
    for i in range(num_simulations):
        sim_data = sample_data.copy() # Independent copy for each simulation
        sim_portfolio = sample_portfolio.copy() # Independent portfolio for each simulation
        sim = Simulation(sim_data, sim_portfolio)
        simulations_list.append(sim)


    # --- Initialize and Train Training Agent ---
    agent_params = {'n_steps': 5, 'n_units': 32, 'n_features_total': 3 * 7, 'n_output_total': 3} # Example params, matching tickers and features
    agent = TrainingAgent(simulations=simulations_list, model_params=agent_params) # Using LSTM model
    agent.train_model(window_size=5, epochs=2) # Train for 2 epochs, window size 5

    print("\n--- LSTM Training Agent Run Completed ---")