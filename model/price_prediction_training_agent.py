import pandas as pd
import numpy as np
import tensorflow as tf

from model.model_data import ModelData # Import ModelData DTO

class PricePredictionTrainingAgent:
    """
    Manages the training of a price prediction model using ModelData objects.
    This agent trains a model to directly predict the pre-calculated price change probabilities
    for a single ticker.
    
    The agent is responsible for:
    - Training the provided model using training data
    - Evaluating the model on evaluation data
    - Calculating baseline predictions for comparison
    
    Note: This class does not handle model creation or parameter management.
    Models should be created externally using ModelBuilder.
    """

    def __init__(self, ticker, model):
        """
        Initializes the PricePredictionTrainingAgent.

        Args:
            ticker (str): The stock ticker symbol to focus on for prediction.
            model (tf.keras.Model): The pre-built and compiled price prediction model.
        """
        if not isinstance(ticker, str):
            raise ValueError("ticker must be a string.")
            
        if not isinstance(model, tf.keras.Model):
            raise ValueError("model must be a valid TensorFlow Keras model.")

        self.ticker = ticker
        self.model = model
        
        # Check if the model is compiled in a more resilient way
        is_compiled = (
            (hasattr(self.model, '_is_compiled') and self.model._is_compiled) or 
            (hasattr(self.model, 'optimizer') and self.model.optimizer is not None)
        )
        
        if not is_compiled:
            raise ValueError("The provided model must be compiled before being passed to the training agent. "
                           "Use model.compile() with appropriate loss function and optimizer.")

    def _calculate_historical_average_baseline(self, model_data_list):
        """
        Calculates the historical average price change probabilities from the training data.

        Args:
            model_data_list (list): List of ModelData objects (training set).

        Returns:
            np.array: Average price change probability vector.
                     Returns None if no valid PPC data is found.
        """
        all_ppc_data = []
        for model_data in model_data_list:
            # Get PPC columns for THIS TICKER ONLY from the complete data
            ppc_cols = [col for col in model_data.complete_data.columns if self.ticker in col[0] and 'PPC' in col[1]]
            if ppc_cols:
                ppc_data = model_data.complete_data[ppc_cols].values
                if ppc_data.size > 0:  # Only include if there's PPC data in this window
                    all_ppc_data.append(ppc_data)

        if not all_ppc_data:
            print("Warning: No Percent Price Change Probability (PPC) data found to calculate baseline.")
            return None

        all_ppc_data_concatenated = np.concatenate(all_ppc_data, axis=0) # Shape: (total_days_training_data, n_probabilities)
        average_ppc = np.mean(all_ppc_data_concatenated, axis=0, keepdims=True) # Average across days
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

            # Get the historical data and ensure index is datetime
            historical_df = model_data.historical_data
            if not pd.api.types.is_datetime64_any_dtype(historical_df.index):
                print(f"Warning: Historical data index is not datetime type. Skipping this ModelData.")
                continue
                
            # Get the complete data and ensure index is datetime
            complete_df = model_data.complete_data
            if not pd.api.types.is_datetime64_any_dtype(complete_df.index):
                print(f"Warning: Complete data index is not datetime type. Skipping this ModelData.")
                continue
                
            # Extract PPC columns directly (no longer using multi-level columns)
            ppc_cols = [col for col in complete_df.columns if self.ticker in col[0] and 'PPC' in col[1]]
            if not ppc_cols:
                print(f"Warning: No PPC columns found for ticker {self.ticker}. Skipping this ModelData.")
                continue
                
            ppc_df = complete_df[ppc_cols]
            
            # Check if there are any PPC columns
            if ppc_df.empty:
                print(f"Warning: No PPC data found for ticker {self.ticker}. Skipping this ModelData.")
                continue

            # Convert to arrays for windowing
            X_values = historical_df.values
            
            # Get the expected number of time steps from the model input shape
            expected_steps = self.model.input_shape[1]
            
            # Ensure the historical data has enough rows for a window
            if len(X_values) < expected_steps:
                print(f"Warning: Historical data has insufficient rows ({len(X_values)}) for window size {expected_steps}. Skipping this ModelData.")
                continue
                
            X_train_windowed = []
            y_target_windowed = []
            dates_windowed = []
            
            # Create sliding windows
            for i in range(expected_steps, len(X_values) + 1):
                # Get the window of X values
                window_X = X_values[i - expected_steps:i]
                
                # Get the date corresponding to the end of this window
                window_end_date = historical_df.index[i-1]
                
                # Try to find the matching date in the complete data
                if window_end_date in ppc_df.index:
                    # Extract target values for this specific date
                    target_values = ppc_df.loc[window_end_date].values
                    
                    # Save the feature window and corresponding target
                    X_train_windowed.append(window_X)
                    y_target_windowed.append(target_values)
                    dates_windowed.append(window_end_date)
                else:
                    print(f"Warning: End date {window_end_date} from historical data not found in complete data. Skipping this window.")
            
            if not X_train_windowed:
                print(f"Warning: No valid windows found for ModelData. Skipping this ModelData.")
                continue
                
            all_X.extend(X_train_windowed)
            all_y.extend(y_target_windowed)

        if not all_X: # Return empty dataset if no valid data
            print("Warning: No valid data found for any ModelData objects.")
            return tf.data.Dataset.from_tensor_slices(([], []))

        X_dataset = np.array(all_X) # Convert lists to NumPy arrays for dataset creation
        y_dataset = np.array(all_y) # y_dataset now shape (num_samples, n_output_probabilities)
        
        print(f"Created dataset with {len(X_dataset)} samples. X shape: {X_dataset.shape}, y shape: {y_dataset.shape}")

        dataset = tf.data.Dataset.from_tensor_slices((X_dataset, y_dataset)) # Create dataset from slices
        dataset = dataset.batch(batch_size) # Batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(all_X)) # Shuffle if training
        dataset = dataset.prefetch(tf.data.AUTOTUNE) # Optimize data loading

        return dataset


    def train_model(self, train_data_list, eval_data_list=None, epochs=10, batch_size=32, early_stopping_patience=20):
        """
        Trains the price prediction model using model.fit().

        Args:
            train_data_list (list): List of ModelData objects for training.
            eval_data_list (list, optional): List of ModelData objects for evaluation.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
            
        Returns:
            history: Training history object from model.fit().
        """
        if not isinstance(train_data_list, list) or not train_data_list or not all(isinstance(md, ModelData) for md in train_data_list):
            raise ValueError("train_data_list must be a non-empty list of ModelData objects.")
        
        if eval_data_list is not None and (not isinstance(eval_data_list, list) or not all(isinstance(md, ModelData) for md in eval_data_list)):
            raise ValueError("eval_data_list must be a list of ModelData objects.")

        print(f"Starting price prediction model training for ticker {self.ticker} using model.fit() with batch size {batch_size}...")

        # --- Calculate Historical Average Baseline Prediction (for comparison) ---
        baseline_prediction = self._calculate_historical_average_baseline(train_data_list)
        if baseline_prediction is None:
            # Determine output shape from model
            output_dim = self.model.output_shape[-1]
            baseline_prediction = np.zeros((1, output_dim))

        # --- Create Training and Evaluation Datasets ---
        train_dataset = self._create_dataset(train_data_list, batch_size=batch_size, shuffle=True)
        eval_dataset = self._create_dataset(eval_data_list, batch_size=batch_size, shuffle=False) if eval_data_list else None

        # --- Configure Callbacks ---
        callbacks = []
        
        # Add ReduceLROnPlateau callback
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if eval_dataset else 'loss',
            factor=0.5,
            patience=5,
            min_lr=0.0000001,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Add EarlyStopping callback if evaluation data is provided
        if eval_dataset:
            stop_training = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=early_stopping_patience,
                verbose=1,
                restore_best_weights=True,
                start_from_epoch=5
            )
            callbacks.append(stop_training)

        print("\n--- Starting model.fit() training ---")
        history = self.model.fit(
            train_dataset,                        # Training data dataset
            epochs=epochs,                        # Number of epochs
            batch_size=None,                      # Batch size is handled by the dataset
            validation_data=eval_dataset,         # Evaluation dataset for validation (can be None)
            verbose=1,                            # Set verbosity (0=silent, 1=progress bar, 2=one line per epoch)
            callbacks=callbacks                   # Callbacks for learning rate reduction and early stopping
        )

        print("\n--- model.fit() training Complete ---")

        # --- Evaluation Loop (Baseline Comparison) ---
        if eval_dataset:
            epoch_baseline_eval_loss = 0.0
            num_eval_batches = 0
            for eval_batch_X, eval_batch_y in eval_dataset:
                if tf.reduce_sum(tf.cast(tf.math.is_nan(eval_batch_X), tf.float32)) > 0 or tf.reduce_sum(tf.cast(tf.math.is_nan(eval_batch_y), tf.float32)) > 0:
                    continue
                if baseline_prediction is not None:
                    baseline_eval_loss = tf.keras.losses.MeanSquaredError()(eval_batch_y, baseline_prediction)
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

        # Get PPC column names from the first ModelData entry
        ppc_cols = [col for col in eval_data_list[0].complete_data.columns if self.ticker in col[0] and 'PPC' in col[1]]
        if not ppc_cols:
            print("Warning: No PPC columns found in evaluation data.")
            return pd.DataFrame()

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
        for i, col_name in enumerate(ppc_cols):
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
    
    # Create flat columns for historical data (multi-level still expected here)
    columns_historical = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']], names=['Ticker', 'OHLCV'])
    
    # Create flat columns for complete data (single-level now)
    columns_complete = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20'] + [f'PPC_{i}' for i in range(21)]
    
    num_windows = 20 # Example number of ModelData windows - increased for train/eval split

    model_data_list = []
    for i in range(num_windows):
        start_date_index = i * 30 # Overlapping windows for example, adjust as needed
        end_date_index = start_date_index + 60 # Window size 60 days

        window_dates = dates[start_date_index:end_date_index]
        if len(window_dates) < 60: # Ensure enough dates in window
            continue # Skip if not enough dates

        historical_data_values = np.random.rand(len(window_dates), len(tickers) * 7) # Dummy historical data
        complete_data_values = np.random.rand(len(window_dates), len(columns_complete)) # Dummy complete data including PPC cols

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

    # --- Create a model using ModelBuilder ---
    feature_count = train_data_list[0].historical_data.shape[1] if train_data_list else 7 * len(tickers)
    model_params = {
        'n_steps': 60,
        'n_features_total': feature_count,
        'n_units': 64,
        'n_output_probabilities': 21,
        'dropout_rate': 0.3,
        'cnn_filters': [32, 64],
        'cnn_kernel_sizes': [3, 3],
        'lstm_layers': 1,
        'dense_layers': [32],
        'activation': 'relu',
        'recurrent_dropout_rate': 0.0,
        'l2_reg': 0.001,
        'learning_rate': 0.0001
    }
    
    # Build the model using ModelBuilder
    model = ModelBuilder.build_price_prediction_model(model_params)
    print(f"Model created: {model.summary()}")
    
    # --- Initialize and Train PricePredictionTrainingAgent ---
    batch_size = 64
    agent = PricePredictionTrainingAgent(
        ticker=ticker,
        model=model
    )
    
    history = agent.train_model(
        train_data_list=train_data_list, 
        eval_data_list=eval_data_list, 
        epochs=10, 
        batch_size=batch_size,
        early_stopping_patience=5
    )

    print("\n--- Evaluating Model ---")
    evaluation_df = agent.evaluate_model(eval_data_list=eval_data_list, batch_size=batch_size)

    if not evaluation_df.empty:
        print("\n--- Evaluation Results DataFrame Sample ---")
        print(evaluation_df.sample(10).to_string())
    else:
        print("\nWarning: Evaluation DataFrame is empty. No predictions generated.")

    print("\n--- PricePredictionTrainingAgent Run Completed ---")