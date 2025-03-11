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

    def __init__(self, ticker, feature_count, model=None, model_params=None, optimizer=None):
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
        if not isinstance(ticker, str):
            raise ValueError("ticker must be a string.")

        self.ticker = ticker
        self.model_params = model_params if model_params is not None else self._get_default_model_params()
        self.model_params['n_features_total'] = feature_count

        print(f"PricePredictionTrainingAgent Model parameters being used: {self.model_params}")
        self.model = model if model is not None else self._create_default_model(self.model_params)
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=self.model_params['learning_rate'])

        # --- COMPILE THE MODEL HERE ---
        self.model.compile(
            optimizer=self.optimizer, # Use the configured optimizer
            loss='mse',         # Mean Squared Error loss function
            metrics=['mse']         # Track Mean Squared Error during training and evaluation
        )
        print("Price Prediction Model Compiled.")

    def _get_default_model_params(self):
        """Returns default LSTM model parameters optimized for price prediction task."""
        # Assuming ModelData.historical_data contains the scaled data
        n_output_probabilities = 21 # Number of price change probabilities

        print(f"Debugging _get_default_model_params for Price Prediction:")
        print(f"  Ticker: {self.ticker}")
        print(f"  Number of output probabilities: {n_output_probabilities}")


        return {
            'n_steps': 60, # Example window size, adjust as needed
            'n_units': 256,
            'n_output_probabilities': n_output_probabilities, # Output size is now number of probabilities
            'learning_rate': 0.0025,
            'dropout_rate': 0.2
        }

    def _create_default_model(self, model_params):
        """Creates a default LSTM model for price prediction."""
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(model_params['n_steps'], model_params['n_features_total'])),
            tf.keras.layers.LSTM(units=model_params['n_units']),
            tf.keras.layers.Dropout(model_params['dropout_rate']),
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

    def _create_dataset(self, model_data_list, batch_size, shuffle=False):
        """
        Creates a tf.data.Dataset from a list of ModelData objects.

        Args:
            model_data_list (list): List of ModelData objects.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the dataset (for training).

        Returns:
            tf.data.Dataset: TensorFlow Dataset object.
        """
        all_X = []
        all_y = []

        for model_data in model_data_list:
            if model_data.historical_data is None or model_data.historical_data.empty:
                continue

            X_train = model_data.historical_data.values
            y_target = model_data.complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').values

            if X_train.shape[0] < self.model_params['n_steps']:
                continue

            X_train_windowed = []
            y_target_windowed = []
            for i in range(self.model_params['n_steps'], X_train.shape[0] + 1): # Create sliding windows
                X_train_windowed.append(X_train[i - self.model_params['n_steps']:i])
                # --- MODIFICATION: Take ONLY the LAST value of y_target window ---
                y_target_windowed.append(y_target[i-1, :]) # Use i-1 to get the target corresponding to the window ending at i

            all_X.extend(X_train_windowed)
            all_y.extend(y_target_windowed)

        if not all_X: # Return empty dataset if no valid data
            return tf.data.Dataset.from_tensor_slices(([], []))

        X_dataset = np.array(all_X) # Convert lists to NumPy arrays for dataset creation
        y_dataset = np.array(all_y) # y_dataset now shape (num_samples, n_output_probabilities)

        dataset = tf.data.Dataset.from_tensor_slices((X_dataset, y_dataset)) # Create dataset from slices
        dataset = dataset.batch(batch_size) # Batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(all_X)) # Shuffle if training
        dataset = dataset.prefetch(tf.data.AUTOTUNE) # Optimize data loading

        return dataset


    def train_model(self, train_data_list, eval_data_list, epochs=10, batch_size=32):
        """
        Trains the price prediction model using model.fit().

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        if not isinstance(train_data_list, list) or not all(isinstance(md, ModelData) for md in train_data_list):
            raise ValueError("train_data_list must be a list of ModelData objects.")
        if not isinstance(eval_data_list, list) or not all(isinstance(md, ModelData) for md in eval_data_list):
            raise ValueError("eval_data_list must be a list of ModelData objects.")

        print(f"Starting price prediction model training for ticker {self.ticker} using model.fit() with batch size {batch_size}...")

        # --- Calculate Historical Average Baseline Prediction (remains - for comparison outside of fit) ---
        baseline_prediction = self._calculate_historical_average_baseline(train_data_list)
        if baseline_prediction is None:
            baseline_prediction = np.zeros((1, self.model_params['n_output_probabilities']))

        # --- Create Training and Evaluation Datasets (remains the same) ---
        train_dataset = self._create_dataset(train_data_list, batch_size=batch_size, shuffle=True)
        eval_dataset = self._create_dataset(eval_data_list, batch_size=batch_size, shuffle=False)

        print("\n--- Starting model.fit() training ---")
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001)
        stop_training = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=30, verbose=1, restore_best_weights=True, start_from_epoch=5)
        history = self.model.fit( # --- Use model.fit() ---
            train_dataset,           # Training data dataset
            epochs=epochs,             # Number of epochs
            batch_size=None,             # Batch size is handled by the dataset
            validation_data=eval_dataset, # Evaluation dataset for validation
            verbose=1,                 # Set verbosity (0=silent, 1=progress bar, 2=one line per epoch)
            callbacks=[reduce_lr, stop_training]
        )

        print("\n--- model.fit() training Complete ---")

        # --- Evaluation Loop (Baseline Comparison - remains, but could be simplified further if desired) ---
        epoch_baseline_eval_loss = 0.0
        num_eval_batches = 0
        for eval_batch_X, eval_batch_y in eval_dataset: # Keep baseline evaluation for comparison
            if tf.reduce_sum(tf.cast(tf.math.is_nan(eval_batch_X), tf.float32)) > 0 or tf.reduce_sum(tf.cast(tf.math.is_nan(eval_batch_y), tf.float32)) > 0:
                continue
            if baseline_prediction is not None:
                baseline_eval_loss = tf.keras.losses.MeanSquaredError()(eval_batch_y, baseline_prediction) # Compare directly with batched y_eval_batch
                epoch_baseline_eval_loss += baseline_eval_loss.numpy()
            num_eval_batches += 1

        if num_eval_batches > 0:
            avg_baseline_eval_loss = epoch_baseline_eval_loss / num_eval_batches if num_eval_batches > 0 and baseline_prediction is not None else 0
            print(f"\n--- Baseline Evaluation on Evaluation Set ---")
            print(f"Baseline Evaluation completed, Avg. Baseline Eval Loss: {avg_baseline_eval_loss:.4f}")


        print(f"\n--- Price Prediction Training for ticker {self.ticker} Complete ---")
        return history # Return the training history object from model.fit()


    def evaluate_model(self, eval_data_list, batch_size=32):
        """
        Evaluates the model using model.predict() on unseen data and returns a DataFrame
        with predictions and expected values.

        Args:
            eval_data_list (list): List of ModelData objects for evaluation.
            batch_size (int): Batch size for evaluation.

        Returns:
            pd.DataFrame: DataFrame of predictions with expected values and loss.
        """
        if not isinstance(eval_data_list, list) or not all(isinstance(md, ModelData) for md in eval_data_list):
            raise ValueError("eval_data_list must be a list of ModelData objects.")

        print(f"Starting price prediction model evaluation for ticker {self.ticker} using model.predict()...")

        # --- Create Evaluation Dataset ---
        eval_dataset = self._create_dataset(eval_data_list, batch_size=batch_size, shuffle=False)

        print("\n--- Starting model.predict() evaluation ---")
        predictions = self.model.predict(eval_dataset, verbose=1) # --- Use model.predict() ---
        print("\n--- model.predict() evaluation Complete ---")

        all_expected_ppc = []
        all_predicted_ppc = []
        all_dates = []

        # Get PPC column names from the first ModelData entry (assuming they are consistent)
        y_target_columns = eval_data_list[0].complete_data.xs(self.ticker, level='Ticker', axis=1).filter(like='PPC').columns.tolist()

        # Iterate through eval dataset to collect expected values and dates in order
        for features, expected_prices in eval_dataset:
            all_expected_ppc.extend(expected_prices.numpy()) # Collect expected PPC values
            # We need to reconstruct dates corresponding to these expected_prices.
            # Since _create_dataset windowed the data, we need to track original dates.
            # For simplicity in this correction, we are not directly associating dates with each prediction in DataFrame output.
            # A more advanced approach would involve tracking original dates during dataset creation.


        # Flatten predictions if necessary (model.predict might return extra dimensions)
        if predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)

        if not all_expected_ppc:
            print("Warning: No expected PPC values collected during evaluation.")
            return pd.DataFrame() # Return empty DataFrame if no data

        all_expected_ppc_np = np.array(all_expected_ppc).reshape(predictions.shape) # Ensure expected values have the same shape as predictions
        loss_value = tf.keras.losses.MeanSquaredError()(all_expected_ppc_np, predictions).numpy() # Calculate loss over all predictions

        evaluation_df = pd.DataFrame()
        for i, col_name in enumerate(y_target_columns):
            evaluation_df[f'Expected_{col_name}'] = all_expected_ppc_np[:, i] # Expected PPC values
            evaluation_df[f'Predicted_{col_name}'] = predictions[:, i] # Predicted PPC values

        print(f"\n--- Price Prediction Evaluation for ticker {self.ticker} Complete, average loss after evaluation: {loss_value:.4f} ---")
        return evaluation_df


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
        'learning_rate': 0.0001
    }
    batch_size = 64
    feature_count = train_data_list[0].historical_data.shape[1] if train_data_list else 7 * len(tickers) # Get feature count dynamically

    agent = PricePredictionTrainingAgent(
        ticker=ticker,
        feature_count=feature_count,
        model_params=agent_params
    )
    agent.train_model(train_data_list=train_data_list, eval_data_list=eval_data_list, epochs=5, batch_size=batch_size) # Train for 5 epochs

    print("\n--- Evaluating Model ---")
    evaluation_df = agent.evaluate_model(eval_data_list=eval_data_list, batch_size=batch_size)

    if not evaluation_df.empty:
        print("\n--- Evaluation Results DataFrame Sample ---")
        print(evaluation_df.sample(10).to_string())
    else:
        print("\nWarning: Evaluation DataFrame is empty. No predictions generated.")


    print("\n--- PricePredictionTrainingAgent Run Completed ---")