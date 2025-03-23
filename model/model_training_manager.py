import os
import time
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union

import numpy as np
import tensorflow as tf

from model.model_builder import ModelBuilder
from model.model_storage_manager import ModelStorageManager
from model.model_identifier import ModelIdentifier
from model.model_data import ModelData
from model.price_prediction_training_agent import PricePredictionTrainingAgent
from model.training_result import TrainingResultDTO
from utils.logger import log_info, log_warning, log_error, log_debug, log_success
from model.model_definition import ModelDefinition

class ModelTrainingManager:
    """
    Manages the end-to-end training process for price prediction models.
    
    This class is responsible for:
    - Initializing a model from a model identifier or parameters
    - Running training using the PricePredictionTrainingAgent
    - Saving results using the ModelStorageManager
    """
    
    def __init__(
        self,
        ticker: str,
        output_dir: str = "data/models",
        epochs: int = 1000,
        early_stopping_patience: int = 30,
        logger = None
    ):
        """
        Initialize the ModelTrainingManager.
        
        Args:
            ticker: The ticker symbol to train for
            output_dir: Directory to save model and results
            epochs: Maximum number of training epochs
            early_stopping_patience: Number of epochs with no improvement before stopping
            logger: Logger instance (optional, not used with new logging system)
        """
        self.ticker = ticker
        self.output_dir = output_dir
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Create output directory for this ticker
        self.ticker_output_dir = os.path.join(output_dir, ticker)
        os.makedirs(self.ticker_output_dir, exist_ok=True)
    
    def _filter_features(self, data_list: List[ModelData], feature_indexes: Set[int]) -> List[ModelData]:
        """
        Filter the features in ModelData objects to include only the selected feature indexes.
        
        Args:
            data_list: List of ModelData objects to filter
            feature_indexes: Set of feature indexes to include
            
        Returns:
            List of new ModelData objects with only the selected features
        """
        log_info(f"Filtering features to include only {len(feature_indexes)} selected features")
        
        # Convert set to sorted list for consistent ordering
        sorted_indexes = sorted(list(feature_indexes))
        
        filtered_data_list = []
        for data in data_list:
            # Extract only the selected columns from historical_data
            filtered_historical = data.historical_data.iloc[:, sorted_indexes].copy()
            
            # Create a new ModelData object with the filtered data
            filtered_data = ModelData(
                filtered_historical,
                data.complete_data,
                data.tickers,
                data.start_date,
                data.end_date
            )
            
            filtered_data_list.append(filtered_data)
        
        log_info(f"Original data shape: {data_list[0].historical_data.shape}, Filtered data shape: {filtered_data_list[0].historical_data.shape}")
        return filtered_data_list

    def train_from_identifier(
        self,
        model_id: str,
        train_data_list: List[ModelData],
        eval_data_list: Optional[List[ModelData]] = None
    ) -> Dict[str, Any]:
        """
        Train a model using a pre-existing model identifier.
        
        Args:
            model_id: The model identifier string
            train_data_list: List of ModelData objects for training
            eval_data_list: List of ModelData objects for evaluation (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        log_info(f"Training model for {self.ticker} using model ID: {model_id}")
        log_info(f"Training data: {len(train_data_list)} windows")
        if eval_data_list:
            log_info(f"Evaluation data: {len(eval_data_list)} windows")
        else:
            log_warning("No evaluation data provided. Will train without evaluation.")
        
        # Check if we have enough data to train
        if not train_data_list:
            log_error(f"No training data available for ticker {self.ticker}")
            return {
                'success': False,
                'ticker': self.ticker,
                'error': 'No training data available',
                'train_windows': 0,
                'eval_windows': len(eval_data_list) if eval_data_list else 0
            }
        
        try:
            # Create a model directory using the model identifier
            model_dir = ModelStorageManager.create_model_directory_from_identifier(
                model_id=model_id,
                base_output_dir=self.output_dir
            )
            log_info(f"Created model directory: {model_dir}")
            
            # Create a run directory for this training run
            run_dir, run_id = ModelStorageManager.create_run_directory(model_dir)
            log_info(f"Created run directory: {run_dir}")
            
            # Get feature count from the first window's historical data
            feature_count = train_data_list[0].historical_data.shape[1]
            log_info(f"Total available features: {feature_count}")
            
            # Decode the model identifier to get parameters
            identifier = ModelIdentifier()
            decoded = identifier.decode_model_identifier(model_id)
            model_params = decoded['model_parameters']
            training_params = decoded['training_parameters']
            feature_indexes = decoded['feature_indexes']
            
            log_info(f"Model uses {len(feature_indexes)}/{feature_count} features ({len(feature_indexes)/feature_count:.1%})")
            
            # Filter the data to include only the selected features
            filtered_train_data = self._filter_features(train_data_list, feature_indexes)
            filtered_eval_data = self._filter_features(eval_data_list, feature_indexes) if eval_data_list else None
            
            # Build the model using ModelBuilder, using the filtered feature count
            log_info(f"Building model with parameters: {model_params}")
            
            # Adjust model construction to use filtered feature count
            input_shape = (model_params['n_steps'], len(feature_indexes))
            model = ModelBuilder().build_model(
                input_shape=input_shape,
                **model_params
            )
            
            # Save the model architecture visualization
            ModelStorageManager.save_model_architecture(model, model_dir)
            
            # Initialize the model to ensure all flags are properly set
            log_info("Initializing model to ensure all internal flags are set")
            dummy_input = np.zeros((1, model_params['n_steps'], len(feature_indexes)))
            _ = model(dummy_input)
            
            # Create the training agent with the model
            agent = PricePredictionTrainingAgent(
                ticker=self.ticker,
                model=model
            )
            
            # Train the model
            training_start = time.time()
            training_result = agent.train_model(
                train_data_list=filtered_train_data,
                eval_data_list=filtered_eval_data,
                epochs=self.epochs,
                batch_size=training_params['batch_size'],
                reduce_lr_patience=training_params['reduce_lr_patience'],
                early_stopping_patience=self.early_stopping_patience,
                model_dir=model_dir,
                run_dir=run_dir,
                run_id=run_id
            )
            training_time = time.time() - training_start
            
            # Save all training results
            save_start = time.time()
            ModelStorageManager.save_training_run(training_result)
            save_time = time.time() - save_start
            
            # Create a link to this model in the ticker output directory
            ticker_model_info = {
                "model_dir": model_dir,
                "run_dir": run_dir,
                "best_model_path": training_result.best_model_path,
                "ticker": self.ticker,
                "created_at": datetime.now().isoformat(),
                "feature_count": len(feature_indexes),
                "feature_indexes": list(feature_indexes)
            }
            with open(os.path.join(self.ticker_output_dir, "model_info.json"), "w") as f:
                json.dump(ticker_model_info, f, indent=2)
            
            # If this is the first model for this ticker, create a symlink or copy the best model
            ticker_model_path = os.path.join(self.ticker_output_dir, "model.keras")
            if training_result.best_model_path and os.path.exists(training_result.best_model_path):
                try:
                    # Try creating a symbolic link first (Windows may require admin privileges)
                    if os.path.exists(ticker_model_path):
                        os.remove(ticker_model_path)
                    try:
                        os.symlink(training_result.best_model_path, ticker_model_path)
                        log_info(f"Created symlink to best model at {ticker_model_path}")
                    except:
                        # Fall back to copy if symlink fails
                        shutil.copy2(training_result.best_model_path, ticker_model_path)
                        log_info(f"Copied best model to {ticker_model_path}")
                except Exception as e:
                    log_warning(f"Could not create model link in ticker directory: {e}")
            
            log_success(f"Successfully trained and saved model for {self.ticker}")
            
            # Get metrics from the training result
            metrics = training_result.metrics.get('evaluation', {}) if training_result.metrics else {}
            if metrics and "Overall_MSE" in metrics:
                log_info(f"Overall MSE: {metrics['Overall_MSE']}")
            
            # Return results
            return {
                'success': True,
                'ticker': self.ticker,
                'training_time': training_time,
                'save_time': save_time,
                'metrics': metrics,
                'train_windows': len(train_data_list),
                'eval_windows': len(eval_data_list) if eval_data_list else 0,
                'feature_count': len(feature_indexes),
                'feature_indexes': list(feature_indexes),
                'total_features': feature_count,
                'model_dir': model_dir,
                'run_dir': run_dir,
                'best_model_path': training_result.best_model_path,
                'model_id': model_id
            }
            
        except Exception as e:
            log_error(f"Error training model for {self.ticker}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'ticker': self.ticker,
                'error': str(e),
                'train_windows': len(train_data_list),
                'eval_windows': len(eval_data_list) if eval_data_list else 0
            }
    
    def train_from_parameters(
        self,
        model_params: Dict[str, Any],
        train_data_list: List[ModelData],
        eval_data_list: Optional[List[ModelData]] = None,
        training_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a model using model parameters.
        
        Args:
            model_params: Dictionary of model parameters
            train_data_list: List of ModelData objects for training
            eval_data_list: List of ModelData objects for evaluation (optional)
            training_params: Dictionary of training parameters (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Check if we have enough data to train
        if not train_data_list:
            log_error(f"No training data available for ticker {self.ticker}")
            return {
                'success': False,
                'ticker': self.ticker,
                'error': 'No training data available',
                'train_windows': 0,
                'eval_windows': len(eval_data_list) if eval_data_list else 0
            }
        
        # Get feature count from the first window's historical data
        feature_count = train_data_list[0].historical_data.shape[1]
        log_info(f"Feature count: {feature_count}")
        
        # Create training parameters dictionary if not provided
        if not training_params:
            # Use ModelDefinition default training parameters
            training_params = ModelDefinition.get_default_training_parameters()
            log_info(f"Using default training parameters: {training_params}")
        
        # Generate feature indexes - use all available features
        feature_indexes = set(range(feature_count))
        log_info(f"Using all {len(feature_indexes)} features for training")
        
        # Initialize ModelIdentifier and create model identifier
        identifier = ModelIdentifier()
        model_id = identifier.create_model_identifier(
            model_parameters=model_params,
            training_parameters=training_params,
            selected_feature_indexes=feature_indexes
        )
        log_info(f"Generated model identifier: {model_id}")
        
        # Train using the generated model identifier
        return self.train_from_identifier(
            model_id=model_id,
            train_data_list=train_data_list,
            eval_data_list=eval_data_list
        )
    
    def generate_summary(self, model_dir: str) -> Dict[str, Any]:
        """
        Generate a summary for a trained model.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            Dictionary containing summary information
        """
        try:
            log_info(f"Generating summary for {self.ticker}...")
            summary = ModelStorageManager.generate_model_summary(model_dir)
            log_success(f"Summary generated for {self.ticker}: Best MSE={summary.get('best_mse', 'N/A')}")
            return summary
        except Exception as e:
            log_error(f"Error generating summary for {self.ticker}: {e}")
            return {
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    # Example of using the ModelTrainingManager
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model using ModelTrainingManager")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol to train for")
    parser.add_argument("--input-dir", type=str, default="data/6_scaled_data", help="Directory with input data")
    parser.add_argument("--output-dir", type=str, default="data/models", help="Directory to save models")
    parser.add_argument("--model-type", choices=['simple', 'cnn', 'complex'], default='simple', 
                        help="Model type to use")
    args = parser.parse_args()
    
    # Mock model parameters based on the model type
    if args.model_type == 'simple':
        model_params = {
            'cnn_layers': 0,  # No CNN
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': False,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.3,
            'recurrent_dropout_rate': 0.0,
            'l2_reg': 0.0,
            'use_batch_norm': False,
            'learning_rate': 0.001,
            'activation': 'relu',
            'n_steps': 60
        }
    elif args.model_type == 'cnn':
        model_params = {
            'cnn_layers': 2,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.3,
            'recurrent_dropout_rate': 0.0,
            'l2_reg': 0.0,
            'use_batch_norm': False,
            'learning_rate': 0.001,
            'activation': 'relu',
            'n_steps': 60
        }
    else:  # complex
        model_params = {
            'cnn_layers': 2,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.3,
            'recurrent_dropout_rate': 0.2,
            'l2_reg': 0.001,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu',
            'n_steps': 60
        }
    
    # Example training parameters
    training_params = {
        'batch_size': 32,
        'reduce_lr_patience': 10
    }
    
    # Create a training manager for the specified ticker
    manager = ModelTrainingManager(
        ticker=args.ticker,
        output_dir=args.output_dir,
        epochs=100,  # Use fewer epochs for testing
        early_stopping_patience=10
    )
    
    log_info(f"Created ModelTrainingManager for {args.ticker}")
    log_info(f"Loading data from {args.input_dir}")
    
    # You would normally load data here
    # For this example, we'll just show the function signature
    log_info("\nTo train from parameters:")
    log_info("manager.train_from_parameters(")
    log_info("    model_params=model_params,")
    log_info("    train_data_list=train_data_list,")
    log_info("    eval_data_list=eval_data_list,")
    log_info("    training_params={'batch_size': 32, 'reduce_lr_patience': 10}")
    log_info(")")
    
    log_info("\nOr to train from an existing model identifier:")
    log_info("manager.train_from_identifier(")
    log_info("    model_id='your_model_id',")
    log_info("    train_data_list=train_data_list,")
    log_info("    eval_data_list=eval_data_list")
    log_info(")")
    
    log_info("\nAfter training, generate a summary:")
    log_info("summary = manager.generate_summary('path/to/model/directory')")
    
    log_info(f"\nModel type selected: {args.model_type}")
    log_debug(f"Model parameters: {model_params}")
    log_debug(f"Training parameters: {training_params}") 