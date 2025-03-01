import pandas as pd
import numpy as np
from model.simulation import Simulation
from model.portfolio import Portfolio
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler

class TrainingAgent:
    """
    Manages the training of a trading model using Simulation objects.
    """

    def __init__(self, simulations, model=None, model_params=None, optimizer=None):
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
        print(f"Model parameters being used: {self.model_params}") # Debugging print: Print model parameters!
        self.model = model if model is not None else self._create_default_model(self.model_params) # Use provided model or create default
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam() # Default optimizer
        self.previous_portfolio_cash_is_all = True # Track if portfolio was all cash previously for static buy reward

    def _get_default_model_params(self):
        """Returns default LSTM model parameters using simplified feature count."""
        sample_simulation = self.simulations[0] # Assume simulations have same tickers and features
        n_tickers = len(sample_simulation.portfolio.tickers())

        # Directly get total features from data column shape
        n_features_total = sample_simulation.data.columns.shape[0]

        n_output_total =  2 * n_tickers # Output for each ticker: [buy_percentage, sell_percentage]

        print(f"Debugging _get_default_model_params (Simplified):") # Debugging print - Updated message
        print(f"  Number of tickers: {n_tickers}") # Debugging print
        print(f"  Total features (directly from data columns): {n_features_total}") # Debugging print - Updated message
        print(f"  Total outputs: {n_output_total}") # Debugging print

        return {
            'n_steps': 60, # Example window size, adjust as needed
            'n_features_total': n_features_total, # Use direct feature count
            'n_units': 64,     # LSTM units
            'n_output_total': n_output_total
        }

    def _create_default_model(self, model_params):
        """Creates the default LSTM model using build_model_multi_output function."""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(model_params['n_steps'], model_params['n_features_total'])), # Use dictionary key access
            tf.keras.layers.LSTM(units=model_params['n_units'], return_sequences=True), # Use dictionary key access
            tf.keras.layers.LSTM(units=model_params['n_units']), # Use dictionary key access
            tf.keras.layers.Dense(units=model_params['n_units']//2, activation='relu'), # Use dictionary key access
            tf.keras.layers.Dense(units=model_params['n_output_total'], activation='sigmoid') # Sigmoid for percentage outputs # Use dictionary key access
        ])


    def reward_function(self, simulation_state):
        """
        Reward function returning a DataFrame with output-specific rewards based on transaction rules.
        """
        tickers = simulation_state.portfolio.tickers()
        output_names = []
        for ticker in tickers:
            output_names.extend([f'{ticker}_buy', f'{ticker}_sell'])
        reward_dict = {name: 0.5 for name in output_names}

        profitable_sell_tickers = []
        transactions = simulation_state.portfolio.get_transactions_on_date(simulation_state.current_date)
        for trade in transactions:
            if trade.transaction_type == 'sell' and trade.profit > 0:
                profitable_sell_tickers.append(trade.ticker)

        # --- Penalize Buy on Profitable Sell Day, Reward Buy on Non-Profitable Sell Day ---
        buy_penalty_on_profit_sell_day = 0.01  # Penalty for buying on a profitable sell day
        buy_reward_static = 0.75  # Static reward for buy orders on other days

        for trade in transactions:
            if trade.transaction_type == 'buy':
                if trade.ticker in profitable_sell_tickers:
                    reward_dict[f"{trade.ticker}_buy"] = 0
                    print(f"   - Reward: Buy Penalty of 0 for buying {trade.quantity} shares of {trade.ticker} on profitable sell day")
                else:
                    reward_dict[f"{trade.ticker}_buy"] += buy_reward_static
                    print(f"   - Reward: Static Buy Reward of {buy_reward_static:.2f} for buying {trade.quantity} shares of {trade.ticker}")

        # --- Profit/Loss Based Sell Reward/Punishment ---
        sell_reward_profit_factor = 0.1
        sell_punishment_loss_factor = 0.05
        for trade in transactions:
            if trade.transaction_type == 'sell':
                if trade.profit > 0:
                    reward_dict[f"{trade.ticker}_sell"] = min(1, trade.profit / 1000)
                    print(f"   - Reward: Sell Profit Reward of {(trade.profit * sell_reward_profit_factor):.2f} for selling {trade.quantity} shares of {trade.ticker} (Profit: ${trade.profit:.2f})")
                elif trade.profit < 0:
                    reward_dict[f"{trade.ticker}_sell"] = 0
                    print(f"   - Reward: Sell Loss Punishment of 0 for selling {trade.quantity} shares of {trade.ticker} (Loss: ${trade.profit:.2f})")

        return reward_dict


    def _model_output_to_orders(self, model_output, tickers, simulation_state):
        """
        Converts the model's output (buy/sell percentages) to a list of orders.
        """
        probabilities = model_output.numpy().flatten() # Flatten to 1D array
        orders = []
        num_tickers = len(tickers)

        for i, ticker in enumerate(tickers):
            buy_percentage = probabilities[i * 2]     # Buy percentage is at even indices
            sell_percentage = probabilities[i * 2 + 1] # Sell percentage is at odd indices

            # --- Buy Orders ---
            cash_to_use_for_buy = simulation_state.portfolio.cash * buy_percentage
            if cash_to_use_for_buy > 0:
                buy_quantity = int(np.round(cash_to_use_for_buy / simulation_state.stock_values[ticker])) # Calculate buy quantity based on available cash and open price
                if buy_quantity > 0:
                    orders.append({'ticker': ticker, 'order_type': 'buy', 'quantity': buy_quantity})
                    print(f"  - Ticker: {ticker}, Buy %: {buy_percentage:.2f}, Cash to use: ${cash_to_use_for_buy:.2f}, Quantity: {buy_quantity}")


            # --- Sell Orders ---
            quant_to_sell = int(np.round(simulation_state.portfolio.get_holding_quantity(ticker) * sell_percentage)) # Quantity to sell
            if quant_to_sell > 0:
                orders.append({'ticker': ticker, 'order_type': 'sell', 'quantity': quant_to_sell})
                print(f"  - Ticker: {ticker}, Sell %: {sell_percentage:.2f}, Quantity to sell: {quant_to_sell}")

        return orders

    def train_model(self, window_size=60):
        """
        Trains the model using the provided simulations with output-specific targets.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        print(f"Starting model training using window size {window_size}...")

        X_train_all_simulations = [] # Collect training data across simulations
        for sim_index, simulation in enumerate(self.simulations):
            print(f"\n-- Simulation {sim_index+1}/{len(self.simulations)} --")
            current_state = simulation.start(window_size) # Start each simulation, get SimulationState

            if current_state is None:
                print(f"Warning: No initial data window for simulation {sim_index+1}. Skipping.")
                continue # Skip to the next simulation if no data window

            step_count = 0
            global_index = 0 # Index to track position in the simulation data
            simulation_history_data = [] # Store data windows, rewards, and orders for the entire simulation

            while True:
                # 1. Prepare Input Data for Model
                if current_state is None:
                    break # End of simulation

                tickers = current_state.portfolio.tickers() # Get tickers in the current window

                # 2. Model Prediction
                X_current = np.array([current_state.data_window.values]) # Shape: (1, window_size, n_features)

                print(f'Simulating {current_state.current_date} closing prices: {current_state.stock_values}')

                orders_output = self.model(X_current) # Get model output (probabilities)
                orders = self._model_output_to_orders(orders_output, tickers, current_state) # Pass simulation state to order conversion

                # 3. Simulation Step (execute orders and get next data window)
                current_state = simulation.step(window_size, orders=orders) # Pass orders to step()

                if current_state is not None:
                    step_count += 1
                    # 4. Reward Calculation
                    reward_df = self.reward_function(current_state) # Calculate reward based on simulation state - now returns DataFrame

                    # Store data, reward, and orders for this step in simulation history
                    simulation_history_data.append({'data_window': X_current, 'reward_df': reward_df, 'orders': orders}) # Store reward_df - CHANGED

                    print(f" - Simulation {sim_index+1}, Step {step_count}, Date: {simulation.current_date.date()}, Reward DataFrame:\n{str(reward_df)}, Portfolio Value: ${simulation.portfolio.value(simulation.valuations[simulation.current_date]):.2f}") # Print DataFrame reward

                    global_index += 1 # Increment global index for next step

                else:
                    print(f"  Simulation {sim_index+1} ended after {step_count} steps.")
                    break # End of simulation


            # --- Epoch-based Model Update with Output-Specific Targets ---
            if simulation_history_data: # Only train if there was simulation data
                print(f"  - Training model on simulation {sim_index+1} history (output-specific targets)...")
                simulation_X_train = np.concatenate([item['data_window'] for item in simulation_history_data], axis=0) # Stack data windows
                simulation_reward_dicts = [item['reward_df'] for item in simulation_history_data] # List of reward DataFrames - CHANGED

                with tf.GradientTape() as tape_inner: # Inner tape for loss calculation
                    output_probs_tape = self.model(simulation_X_train) # Get model output for the entire simulation
                    target_probs = np.zeros_like(output_probs_tape) # Initialize target probabilities

                    # Generate output-specific target probabilities based on DataFrame rewards and orders
                    for step_idx in range(len(simulation_reward_dicts)): # Iterate through steps
                        reward_df_step = simulation_reward_dicts[step_idx] # Get reward DataFrame for this step

                        for i, ticker in enumerate(tickers):
                            target_probs[step_idx, i * 2] = reward_df_step[f'{ticker}_buy']
                            target_probs[step_idx, i * 2 + 1] = reward_df_step[f'{ticker}_sell']

                    loss = tf.keras.losses.MeanSquaredError()(target_probs, output_probs_tape) # MSE loss

                gradients = tape_inner.gradient(loss, self.model.trainable_variables) # Gradients for simulation loss
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # Apply gradients

                print(f"  - Simulation {sim_index+1} training completed. Loss: {loss.numpy():.4f}")
            else:
                print(f" - No training data for simulation {sim_index+1} (simulation history empty).")


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

    # --- Prepare Agent Parameters Dynamically ---
    sample_simulation = simulations_list[0] # Use the first simulation to dynamically determine parameters
    n_tickers = len(sample_simulation.portfolio.tickers)
    n_features_per_ticker = 0
    for col in sample_simulation.data.columns:
        if col[0] == sample_simulation.portfolio.tickers[0]:
            n_features_per_ticker += 1
    n_ticker_features_total = n_features_per_ticker * n_tickers
    n_portfolio_state_features = 2 + n_tickers
    n_features_total = n_ticker_features_total + n_portfolio_state_features
    n_output_total = 2 * n_tickers

    agent_params = {
        'n_steps': 5, # Example window size
        'n_units': 32,
        'n_features_total': n_features_total, # Dynamically calculated total features
        'n_output_total': n_output_total  # Dynamically calculated total outputs
    }

    # --- Initialize and Train Training Agent ---
    agent = TrainingAgent(simulations=simulations_list, model_params=agent_params) # Using LSTM model
    agent.train_model(window_size=5) # Train for 1 epoch, window size 5

    print("\n--- LSTM Training Agent Run Completed ---")