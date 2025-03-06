import pandas as pd
import numpy as np
import tensorflow as tf

from model.model_data import ModelData # Import ModelData DTO

class PricePredictionTrainingAgent:
    """
    Manages the training of a price prediction model using ModelData objects.
    This agent trains a model to directly predict the pre-calculated price change probabilities
    for a single ticker.
    """

    def __init__(self, train_data_list, eval_data_list, ticker, model=None, model_params=None, optimizer=None):
        """
        Initializes the PricePredictionTrainingAgent.

        Args:
            train_data_list (list): List of ModelData objects for training.
            eval_data_list (list): List of ModelData objects for evaluation.
            ticker (str): The stock ticker symbol to focus on for prediction.
            model (optional): The price prediction model instance. If None, a default model will be created.
            model_params (dict, optional): Parameters for the model. If None, default parameters will be used.
            optimizer (tf.keras.optimizers.Optimizer, optional): Optimizer for training. If None, Adam optimizer is used.
        """
        if not isinstance(train_data_list, list) or not all(isinstance(md, ModelData) for md in train_data_list):
            raise ValueError("train_data_list must be a list of ModelData objects.")
        if not isinstance(eval_data_list, list) or not all(isinstance(md, ModelData) for md in eval_data_list):
            raise ValueError("eval_data_list must be a list of ModelData objects.")
        if not isinstance(ticker, str):
            raise ValueError("ticker must be a string.")

        self.train_data_list = train_data_list # Separate training data
        self.eval_data_list = eval_data_list   # Separate evaluation data
        self.ticker = ticker
        self.model_params = model_params if model_params is not None else self._get_default_model_params()
        
        print(f"PricePredictionTrainingAgent Model parameters being used: {self.model_params}")
        self.model = model if model is not None else self._create_default_model(self.model_params)
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam()

        # Validate that ModelData objects contain data for the specified ticker
        for model_data_train in self.train_data_list:
            if self.ticker not in model_data_train.tickers:
                raise ValueError(f"Training ModelData object does not contain data for ticker: {self.ticker}. Available tickers: {model_data_train.tickers}")
        for model_data_eval in self.eval_data_list:
            if self.ticker not in model_data_eval.tickers:
                raise ValueError(f"Evaluation ModelData object does not contain data for ticker: {self.ticker}. Available tickers: {model_data_eval.tickers}")


    def _get_default_model_params(self):
        """Returns default LSTM model parameters optimized for price prediction task."""
        sample_model_data = self.train_data_list[0] # Use the first TRAINING ModelData to determine parameters

        # Assuming ModelData.historical_data contains the scaled data
        n_features_total = sample_model_data.historical_data.xs(self.ticker, level='Ticker', axis=1).shape[1]
        n_output_probabilities = 21 # Number of price change probabilities

        print(f"Debugging _get_default_model_params for Price Prediction:")
        print(f"  Ticker: {self.ticker}")
        print(f"  Total features for ticker: {n_features_total}")
        print(f"  Number of output probabilities: {n_output_probabilities}")


        return {
            'n_steps': 60, # Example window size, adjust as needed
            'n_features_total': n_features_total,
            'n_units': 64,
            'n_output_probabilities': n_output_probabilities # Output size is now number of probabilities
        }

    def _create_default_model(self, model_params):
        """Creates a default LSTM model for price prediction."""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(model_params['n_steps'], model_params['n_features_total'])),
            tf.keras.layers.LSTM(units=model_params['n_units'], return_sequences=True),
            tf.keras.layers.LSTM(units=model_params['n_units']),
            tf.keras.layers.Dense(units=model_params['n_units']//2, activation='relu'),
            tf.keras.layers.Dense(units=model_params['n_output_probabilities'], activation='sigmoid') # Sigmoid for probability outputs
        ])

    def _calculate_historical_average_baseline(self, model_data_list):
        """
        Calculates the historical average price change probabilities from the training data.

        Args:
            model_data_list (list): List of ModelData objects (training set).

        Returns:
            np.array: Average price change probability vector (shape: (1, 21)).
                       Returns None if no valid PPC data is found.
        """
        all_ppc_data = []
        for model_data in model_data_list:
            ppc_data = model_data.complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').values
            if ppc_data.size > 0: # Only include if there's PPC data in this window
                all_ppc_data.append(ppc_data)

        if not all_ppc_data:
            print("Warning: No Percent Price Change Probability (PPC) data found to calculate baseline.")
            return None

        all_ppc_data_concatenated = np.concatenate(all_ppc_data, axis=0) # Shape: (total_days_training_data, 21)
        average_ppc = np.mean(all_ppc_data_concatenated, axis=0, keepdims=True) # Average across days, shape: (1, 21)
        return average_ppc


    def train_model(self, epochs=10):
        """
        Trains the price prediction model using the provided ModelData objects.

        Args:
            epochs (int): Number of training epochs.
        """
        print(f"Starting price prediction model training for ticker {self.ticker}...")

        # --- Calculate Historical Average Baseline Prediction ---
        baseline_prediction = self._calculate_historical_average_baseline(self.train_data_list) # Calculate baseline using TRAINING data
        if baseline_prediction is None:
            print("Warning: Could not calculate historical average baseline. Baseline comparison will be skipped.")
            baseline_prediction = np.zeros((1, self.model_params['n_output_probabilities'])) # Fallback to zero prediction if no baseline


        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            epoch_loss = 0.0
            epoch_eval_loss = 0.0 # NEW: Track evaluation loss
            epoch_baseline_train_loss = 0.0 # NEW: Track baseline training loss
            epoch_baseline_eval_loss = 0.0 # NEW: Track baseline evaluation loss
            num_batches = 0
            num_eval_batches = 0 # NEW: Count evaluation batches

            # --- Training Loop ---
            for model_data in self.train_data_list: # Iterate through TRAINING data
                if model_data.historical_data is None or model_data.historical_data.empty:
                    print("Warning: No historical data in ModelData object. Skipping.")
                    continue

                # Extract input data and target probabilities for the specified ticker
                X_train = model_data.historical_data.xs(self.ticker, level='Ticker', axis=1).values # Shape: (window_size, n_features)
                # Assuming ModelData.complete_data contains the target price change probabilities
                y_target = model_data.complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').values # Shape: (window_size, 21) - filter columns by 'PPC' tag

                if X_train.shape[0] < self.model_params['n_steps']:
                    print(f"Warning: Data window size too small ({X_train.shape[0]} days) for model input ({self.model_params['n_steps']} steps). Skipping this window.")
                    continue # Skip windows smaller than n_steps

                # Reshape for LSTM input (batch_size, timesteps, features) - batch_size=1 for single window
                X_train_reshaped = np.array([X_train[-self.model_params['n_steps']:]]) # Take last n_steps days
                y_target_reshaped = np.array([y_target[-self.model_params['n_steps']:]]) # Take corresponding target probabilities

                if np.isnan(X_train_reshaped).any() or np.isnan(y_target_reshaped).any():
                    print("Warning: NaN values found in training data or targets. Skipping this batch.")
                    continue


                with tf.GradientTape() as tape:
                    y_pred = self.model(X_train_reshaped) # Get model predictions - shape (1, n_steps, 21)
                    # Take only the LAST prediction in the sequence for each probability, shape (1, 21)
                    y_pred_last_step = y_pred[:, :]  # Shape: (1, 21)

                    # Calculate loss using MSE - comparing last prediction with the LAST target probabilities
                    loss = tf.keras.losses.MeanSquaredError()(y_target_reshaped[:, -1, :], y_pred_last_step) # Compare last step predictions with last step target

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                epoch_loss += loss.numpy()

                # --- NEW: Calculate Baseline Training Loss ---
                if baseline_prediction is not None:
                    baseline_train_loss = tf.keras.losses.MeanSquaredError()(y_target_reshaped[:, -1, :], baseline_prediction) # Baseline loss on training batch
                    epoch_baseline_train_loss += baseline_train_loss.numpy()


                num_batches += 1
                print(f"  - Trained on window starting {model_data.start_date.date()}, Loss: {loss.numpy():.4f}")


            # --- Evaluation Loop (AFTER training epoch) ---
            for eval_model_data in self.eval_data_list: # Iterate through EVALUATION data
                eval_historical_data = eval_model_data.get_historical_data()
                eval_X_train = eval_historical_data.xs(self.ticker, level='Ticker', axis=1).values
                eval_y_target = eval_model_data.complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').values

                if eval_historical_data.empty:
                    print("Warning: No historical data in evaluation ModelData object. Skipping evaluation for this window.")
                    continue

                if eval_X_train.shape[0] < self.model_params['n_steps']:
                    continue # Skip small windows

                eval_X_train_reshaped = np.array([eval_X_train[-self.model_params['n_steps']:]])
                eval_y_target_reshaped = np.array([eval_y_target[-self.model_params['n_steps']:]])

                y_eval_pred = self.model(eval_X_train_reshaped) # Model prediction on evaluation data
                y_eval_pred_last_step = y_eval_pred[:, :]

                eval_loss = tf.keras.losses.MeanSquaredError()(eval_y_target_reshaped[:, -1, :], y_eval_pred_last_step) # Evaluation loss

                epoch_eval_loss += eval_loss.numpy()

                # --- NEW: Calculate Baseline Evaluation Loss ---
                if baseline_prediction is not None:
                    baseline_eval_loss = tf.keras.losses.MeanSquaredError()(eval_y_target_reshaped[:, -1, :], baseline_prediction) # Baseline loss on evaluation batch
                    epoch_baseline_eval_loss += baseline_eval_loss.numpy()

                num_eval_batches += 1
                print(f"  - Evaluated on window starting {eval_model_data.start_date.date()}, Loss: {eval_loss.numpy():.4f}")


            # --- Epoch Summary Output ---
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_baseline_train_loss = epoch_baseline_train_loss / num_batches if num_batches > 0 and baseline_prediction is not None else 0 # Average baseline training loss
                print(f"Epoch {epoch+1} Training completed, Avg. Training Loss: {avg_loss:.4f}, Baseline Train Loss: {avg_baseline_train_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} Training completed, No training batches in this epoch.")

            if num_eval_batches > 0: # NEW: Evaluation summary
                avg_eval_loss = epoch_eval_loss / num_eval_batches
                avg_baseline_eval_loss = epoch_baseline_eval_loss / num_eval_batches if num_eval_batches > 0 and baseline_prediction is not None else 0 # Average baseline eval loss

                print(f"Epoch {epoch+1} Evaluation completed, Avg. Evaluation Loss: {avg_eval_loss:.4f}, Baseline Eval Loss: {avg_baseline_eval_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} Evaluation completed, No evaluation batches in this epoch.")


        print(f"\n--- Price Prediction Training for ticker {self.ticker} Complete ---")


    def evaluate_model(self, model_data_entries):
        """
        Tests the model (without altering weights) with data that it hasn't seen before.
        """
        print(f"Starting price prediction model evaluation for ticker {self.ticker}...")

        all_predictions = []
        average_loss = 0
        y_target_columns = model_data_entries[0].complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').columns.tolist()

        for model_data in model_data_entries:
            if model_data.historical_data is None or model_data.historical_data.empty:
                print("Warning: No historical data in ModelData object. Skipping.")
                continue

            # Extract input data and target probabilities for the specified ticker
            X_train = model_data.historical_data.xs(self.ticker, level='Ticker', axis=1).values # Shape: (window_size, n_features)
            # Assuming ModelData.complete_data contains the target price change probabilities

            y_target = model_data.complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').values # Shape: (window_size, 21) - filter columns by 'PPC' tag

            if X_train.shape[0] < self.model_params['n_steps']:
                print(f"Warning: Data window size too small ({X_train.shape[0]} days) for model input ({self.model_params['n_steps']} steps). Skipping this window.")
                continue # Skip windows smaller than n_steps

            # Reshape for LSTM input (batch_size, timesteps, features) - batch_size=1 for single window
            X_train_reshaped = np.array([X_train[-self.model_params['n_steps']:]]) # Take last n_steps days
            y_target_reshaped = np.array([y_target[-self.model_params['n_steps']:]]) # Take corresponding target probabilities

            if np.isnan(X_train_reshaped).any() or np.isnan(y_target_reshaped).any():
                print("Warning: NaN values found in training data or targets. Skipping this batch.")
                continue

            y_pred = self.model(X_train_reshaped) # Get model predictions - shape (1, n_steps, 21)

            all_predictions.append([model_data.end_date] + y_target_reshaped[:, -1, :].flatten().tolist() + y_pred.numpy().flatten().tolist())

            # Take only the LAST prediction in the sequence for each probability, shape (1, 21)
            y_pred_last_step = y_pred[:, :]  # Shape: (1, 21)

            # Calculate loss using MSE - comparing last prediction with the LAST target probabilities
            loss = tf.keras.losses.MeanSquaredError()(y_target_reshaped[:, -1, :], y_pred_last_step) # Compare last step predictions with last step target
            average_loss += loss.numpy() / len(model_data_entries)

            print(f"  - Evaluated on window starting {model_data.start_date.date()}, Loss: {loss.numpy():.4f}")

        print(f"\n--- Price Prediction Training for ticker {self.ticker} Complete, average loss after training: {average_loss:.4f} ---")
        return pd.DataFrame(all_predictions, columns=['Date'] + y_target_columns + list(map(lambda c: f'Predicted_{c}', y_target_columns)))

# Example Usage (for testing PricePredictionTrainingAgent)
if __name__ == '__main__':
    # --- Sample Data (Adapt your DataManager and ModelData creation here) ---
    # For now, creating dummy ModelData list similar to previous simulation example
    dates = pd.to_datetime(['2012-08-01', '2012-08-02', '2012-08-03', '2012-08-06', '2012-08-07', '2012-08-08', '2012-08-09', '2012-08-10'] * 60) # 60 days per window * num_windows
    dates.sort_values(inplace=True)
    dates = dates.unique()
    ticker = 'AAPL' # Train for a single ticker
    tickers = [ticker] # Ticker list for DataManager compatibility
    columns_historical = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']], names=['Ticker', 'OHLCV'])
    columns_complete = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20'] + [f'PPC_{i}' for i in range(21)]], names=['Ticker', 'Data']) # Include PPC cols in complete data
    num_windows = 20 # Example number of ModelData windows - increased for train/eval split

    model_data_list = []
    for i in range(num_windows):
        start_date_index = i * 30 # Overlapping windows for example, adjust as needed
        end_date_index = start_date_index + 60 # Window size 60 days

        window_dates = dates[start_date_index:end_date_index]
        if len(window_dates) < 60: # Ensure enough dates in window
            continue # Skip if not enough dates

        historical_data_values = np.random.rand(len(window_dates), len(tickers) * 7) # Dummy historical data
        complete_data_values = np.random.rand(len(window_dates), len(tickers) * (7 + 21)) # Dummy complete data including PPC cols

        historical_data_df = pd.DataFrame(historical_data_values, index=window_dates, columns=columns_historical)
        complete_data_df = pd.DataFrame(complete_data_values, index=window_dates, columns=columns_complete)
        historical_data_df.index.name = 'Date'
        complete_data_df.index.name = 'Date'

        model_data = ModelData(historical_data_df, complete_data_df, tickers, start_date=window_dates[0], end_date=window_dates[-1]) # Added start and end dates
        model_data_list.append(model_data)

    # --- Data Splitting for Train/Eval ---
    split_index = int(0.8 * len(model_data_list)) # 80% for training
    train_data_list = model_data_list[:split_index]
    eval_data_list = model_data_list[split_index:]

    print(f"Training windows: {len(train_data_list)}, Evaluation windows: {len(eval_data_list)}")


    # --- Initialize and Train PricePredictionTrainingAgent ---
    agent_params = {
        'n_steps': 60, # Example window size matching data window
        'n_units': 32,
    }

    agent = PricePredictionTrainingAgent(
        train_data_list=train_data_list, # Pass train data list
        eval_data_list=eval_data_list,   # Pass eval data list
        ticker=ticker,
        model_params=agent_params
    )
    agent.train_model(epochs=5) # Train for 5 epochs

    print("\n--- PricePredictionTrainingAgent Run Completed ---")