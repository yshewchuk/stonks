import pandas as pd
import numpy as np
import tensorflow as tf
import time
from typing import List, Dict, Optional, Tuple, Any, Union

# Set matplotlib backend to non-interactive to avoid tkinter issues in multithreading
import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib imports

from model.model_data import ModelData # Import ModelData DTO
from model.training_result import TrainingResultDTO # Import our new DTO

class PricePredictionTrainingAgent:
    """
    Manages the training of a price prediction model using ModelData objects.
    This agent trains a model to directly predict the pre-calculated price change probabilities
    for a single ticker.
    
    The agent is responsible for:
    - Training the provided model using training data
    - Evaluating the model on evaluation data
    - Calculating baseline predictions for comparison
    
    This agent does not handle model creation, parameter management, or storage operations.
    Models should be created externally using ModelBuilder, and storage operations 
    should be handled by ModelStorageManager.
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


    def train_model(self, train_data_list, eval_data_list=None, epochs=10, batch_size=32, reduce_lr_patience=5, early_stopping_patience=20, model_dir=None, run_dir=None, run_id=None):
        """
        Trains the price prediction model using model.fit().

        Args:
            train_data_list (list): List of ModelData objects for training.
            eval_data_list (list, optional): List of ModelData objects for evaluation.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
            model_dir (str, optional): Directory where the model is stored.
            run_dir (str, optional): Directory for this training run.
            run_id (str, optional): ID for this training run.
            
        Returns:
            TrainingResultDTO: Object containing all training results and metrics
        """
        if not isinstance(train_data_list, list) or not train_data_list or not all(isinstance(md, ModelData) for md in train_data_list):
            raise ValueError("train_data_list must be a non-empty list of ModelData objects.")
        
        if eval_data_list is not None and (not isinstance(eval_data_list, list) or not all(isinstance(md, ModelData) for md in eval_data_list)):
            raise ValueError("eval_data_list must be a list of ModelData objects.")

        # Initialize our result DTO
        result = TrainingResultDTO(
            ticker=self.ticker,
            run_id=run_id,
            model_dir=model_dir,
            run_dir=run_dir,
            model=self.model
        )

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
        
        # Add ModelCheckpoint callback if run_dir is provided
        best_model_path = None
        if run_dir:
            best_model_path = f"{run_dir}/best_model.keras"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor='val_loss' if eval_dataset else 'loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
            # Store the best model path in the result
            result.best_model_path = best_model_path
        
        # Add TensorBoard callback if run_dir is provided
        if run_dir:
            log_dir = f"{run_dir}/logs"
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
        
        # Add CSV logger if run_dir is provided
        if run_dir:
            csv_path = f"{run_dir}/training_log.csv"
            csv_logger = tf.keras.callbacks.CSVLogger(
                csv_path,
                separator=',',
                append=False
            )
            callbacks.append(csv_logger)
        
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
            
        # Add timeout callback to stop training after 30 minutes
        class TimeoutCallback(tf.keras.callbacks.Callback):
            def __init__(self, timeout_minutes=120):
                super().__init__()
                self.timeout_minutes = timeout_minutes
                self.start_time = None
                
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                
            def on_epoch_end(self, epoch, logs=None):
                elapsed_minutes = (time.time() - self.start_time) / 60
                if elapsed_minutes > self.timeout_minutes:
                    print(f"\nTimeout reached after {elapsed_minutes:.2f} minutes. Stopping training.")
                    self.model.stop_training = True
        
        timeout_callback = TimeoutCallback(timeout_minutes=120)
        callbacks.append(timeout_callback)

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

        # --- Record training metrics and results ---
        metrics = {}
        train_loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])
        
        # Record final metrics
        metrics['final_train_loss'] = train_loss[-1] if train_loss else None
        metrics['final_val_loss'] = val_loss[-1] if val_loss else None
        metrics['best_val_loss'] = min(val_loss) if val_loss else None
        metrics['best_epoch'] = val_loss.index(min(val_loss)) + 1 if val_loss else None
        
        # Add information about timeout if it occurred
        elapsed_minutes = (time.time() - timeout_callback.start_time) / 60 if timeout_callback.start_time else 0
        metrics['training_time_minutes'] = elapsed_minutes
        metrics['timeout_occurred'] = elapsed_minutes > timeout_callback.timeout_minutes

        # --- Evaluation Loop (Baseline Comparison) ---
        baseline_eval_metrics = {}
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
                baseline_eval_metrics['avg_baseline_loss'] = avg_baseline_eval_loss
                print(f"\n--- Baseline Evaluation on Evaluation Set ---")
                print(f"Baseline Evaluation completed, Avg. Baseline Eval Loss: {avg_baseline_eval_loss:.4f}")

            # --- Generate evaluation data for best model ---
            if eval_data_list:
                # Evaluate the current model (might be the best model if restore_best_weights is True)
                evaluation_df = self.evaluate_model(eval_data_list, batch_size)
                
                if not evaluation_df.empty:
                    # Calculate and save metrics
                    expected_cols = [col for col in evaluation_df.columns if col.startswith('Expected_')]
                    predicted_cols = [col for col in evaluation_df.columns if col.startswith('Predicted_')]
                    
                    # Calculate MSE for each PPC
                    feature_metrics = {}
                    for expected_col, predicted_col in zip(expected_cols, predicted_cols):
                        feature_name = expected_col.replace('Expected_', '')
                        feature_metrics[f"MSE_{feature_name}"] = float(((evaluation_df[expected_col] - evaluation_df[predicted_col])**2).mean())
                    
                    # Calculate overall MSE
                    expected_values = evaluation_df[expected_cols].values
                    predicted_values = evaluation_df[predicted_cols].values
                    feature_metrics["Overall_MSE"] = float(((expected_values - predicted_values)**2).mean())
                    
                    # Save metrics
                    metrics['evaluation'] = feature_metrics
                    
                    # Save evaluation DataFrame in the result
                    result.evaluation_df = evaluation_df

        print(f"\n--- Price Prediction Training for ticker {self.ticker} Complete ---")
        
        # Populate the TrainingResultDTO with all results
        result.history = history.history
        result.metrics = metrics
        result.baseline_metrics = baseline_eval_metrics
        
        return result

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