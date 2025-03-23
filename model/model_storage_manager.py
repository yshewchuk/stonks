import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import glob
import math
from datetime import datetime

from model.training_result import TrainingResultDTO
from model.model_hash_manager import ModelHashManager
from model.model_identifier import ModelIdentifier
from model.model_definition import ModelDefinition
from utils.logger import log_info, log_warning, log_error, log_debug, log_success

class ModelStorageManager:
    """
    Manages the storage and retrieval of trained models and their associated data.
    
    This class is responsible for:
    - Creating model directories
    - Saving/loading models and their metadata
    - Managing training run directories and files
    - Generating model summaries and visualizations
    
    By separating this logic from the ModelBuilder and PricePredictionTrainingAgent classes,
    we keep those classes focused on their core responsibilities.
    """
    
    DEFAULT_BASE_OUTPUT_DIR = "data/models"
    
    @staticmethod
    def create_model_directory(model_params, feature_data=None, training_params=None, base_output_dir=None):
        """
        Create a directory for a new model based on its parameters, features, and training params.
        
        Args:
            model_params (dict): Model parameters
            feature_data: Data containing the features used by the model (optional)
            training_params (dict): Training parameters (optional)
            base_output_dir (str, optional): Base directory for models
            
        Returns:
            str: Path to the created model directory
        """
        # Use default output directory if none provided
        if base_output_dir is None:
            base_output_dir = ModelStorageManager.DEFAULT_BASE_OUTPUT_DIR
        
        # Make sure the base output directory exists
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Extract feature indexes from data if provided
        if feature_data is not None:
            # Calculate total features from data
            if hasattr(feature_data, 'shape'):
                # If feature_data is a DataFrame or ndarray with shape attribute
                feature_count = feature_data.shape[1] if len(feature_data.shape) > 1 else feature_data.shape[0]
            else:
                # Try to get data from dictionary
                if 'data' in feature_data and hasattr(feature_data['data'], 'shape'):
                    feature_count = feature_data['data'].shape[1] if len(feature_data['data'].shape) > 1 else feature_data['data'].shape[0]
                else:
                    # Default to ModelDefinition.TOTAL_FEATURE_COUNT
                    feature_count = ModelDefinition.TOTAL_FEATURE_COUNT
            
            # By default, include all feature indexes
            included_feature_indexes = set(range(feature_count))
        else:
            # If no feature data provided, use all features up to TOTAL_FEATURE_COUNT
            included_feature_indexes = set(range(ModelDefinition.TOTAL_FEATURE_COUNT))
        
        # Use empty training params if not provided
        if training_params is None:
            training_params = ModelDefinition.get_default_training_parameters()
        
        # Validate parameters
        model_params = ModelDefinition.validate_model_parameters(model_params)
        training_params = ModelDefinition.validate_training_parameters(training_params)
        
        # Initialize the ModelIdentifier (no need to pass feature names)
        identifier = ModelIdentifier()
        
        # Generate deterministic identifier for this model
        model_id = identifier.create_model_identifier(
            model_parameters=model_params,
            training_parameters=training_params,
            selected_feature_indexes=included_feature_indexes
        )
        
        # Create the model directory
        model_dir = os.path.join(base_output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model parameters to JSON file
        params_path = os.path.join(model_dir, "model_params.json")
        if not os.path.exists(params_path):
            try:
                with open(params_path, 'w') as f:
                    json.dump(model_params, f, indent=2)
            except Exception as e:
                log_warning(f"Could not save model parameters: {e}")
        
        # Save feature information if applicable
        features_path = os.path.join(model_dir, "feature_info.json")
        if not os.path.exists(features_path):
            try:
                feature_info = {
                    "total_features": len(included_feature_indexes),
                    "included_feature_indexes": list(included_feature_indexes)
                }
                with open(features_path, 'w') as f:
                    json.dump(feature_info, f, indent=2)
            except Exception as e:
                log_warning(f"Could not save feature information: {e}")
        
        # Save training parameters if provided
        if training_params:
            training_params_path = os.path.join(model_dir, "training_params.json")
            if not os.path.exists(training_params_path):
                try:
                    with open(training_params_path, 'w') as f:
                        json.dump(training_params, f, indent=2)
                except Exception as e:
                    log_warning(f"Could not save training parameters: {e}")
        
        # Create runs directory
        runs_dir = os.path.join(model_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(model_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        return model_dir
    
    @staticmethod
    def create_model_directory_from_identifier(model_id, base_output_dir=None):
        """
        Create a model directory from an existing model identifier.
        
        Args:
            model_id (str): The model identifier string
            base_output_dir (str, optional): Base directory for models
            
        Returns:
            str: Path to the created model directory
        """
        log_info(f"Creating model directory from identifier: {model_id}")

        # Use default output directory if none provided
        if base_output_dir is None:
            base_output_dir = ModelStorageManager.DEFAULT_BASE_OUTPUT_DIR
        
        # Make sure the base output directory exists
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create the model directory
        model_dir = os.path.join(base_output_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Decode the model identifier to get parameters and feature indexes
        identifier = ModelIdentifier()
        decoded = identifier.decode_model_identifier(model_id)
        
        model_params = decoded['model_parameters']
        training_params = decoded['training_parameters']
        included_feature_indexes = decoded['feature_indexes']
        
        # Save model parameters to JSON file
        params_path = os.path.join(model_dir, "model_params.json")
        if not os.path.exists(params_path):
            try:
                with open(params_path, 'w') as f:
                    json.dump(model_params, f, indent=2)
            except Exception as e:
                log_warning(f"Could not save model parameters: {e}")
        
        # Save feature information
        features_path = os.path.join(model_dir, "feature_info.json")
        if not os.path.exists(features_path):
            try:
                feature_info = {
                    "total_features": ModelDefinition.TOTAL_FEATURE_COUNT,
                    "included_feature_indexes": list(included_feature_indexes)
                }
                with open(features_path, 'w') as f:
                    json.dump(feature_info, f, indent=2)
            except Exception as e:
                log_warning(f"Could not save feature information: {e}")
        
        # Save training parameters
        training_params_path = os.path.join(model_dir, "training_params.json")
        if not os.path.exists(training_params_path):
            try:
                with open(training_params_path, 'w') as f:
                    json.dump(training_params, f, indent=2)
            except Exception as e:
                log_warning(f"Could not save training parameters: {e}")
        
        # Create runs directory
        runs_dir = os.path.join(model_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(model_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        return model_dir
    
    @staticmethod
    def create_run_directory(model_dir: str) -> Tuple[str, str]:
        """
        Creates a uniquely numbered run directory for a training run.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            Tuple of (run_directory_path, run_id)
        """
        # Create a runs directory if it doesn't exist
        runs_dir = os.path.join(model_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        
        # Create a unique run ID using timestamp and a random component
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(np.random.randint(1000, 9999))  # 4-digit random number
        run_id = f"{timestamp}_{random_suffix}"
        run_dir = os.path.join(runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Create necessary subdirectories
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Log the run creation
        log_info(f"Created training run directory: {run_dir}")
        
        return run_dir, run_id
    
    @staticmethod
    def save_model_architecture(model, model_dir):
        """
        Save model architecture as a diagram and summary text file.
        
        Args:
            model (tf.keras.Model): The model to save
            model_dir (str): Directory to save the architecture
        """
        # Save model architecture as a diagram
        try:
            diagram_path = os.path.join(model_dir, "model_architecture.png")
            tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True, show_dtype=True, show_layer_names=True)
            log_info(f"Saved model architecture diagram to {diagram_path}")
        except Exception as e:
            log_warning(f"Could not save model architecture diagram: {str(e)}")
        
        # Save model architecture as a summary text file
        try:
            with open(os.path.join(model_dir, "model_summary.txt"), 'w', encoding='utf-8') as f:
                # Redirect model.summary() to the file
                model.summary(print_fn=lambda line: f.write(line + '\n'))
            log_info(f"Saved model summary to {os.path.join(model_dir, 'model_summary.txt')}")
        except Exception as e:
            log_warning(f"Could not save model summary to file: {str(e)}")
            # Print to console instead
            model.summary()
    
    @staticmethod
    def save_training_run(result: TrainingResultDTO) -> None:
        """
        Save all data associated with a training run.
        
        Args:
            result: TrainingResultDTO containing training results
        """
        if not result.run_dir or not os.path.exists(result.run_dir):
            raise ValueError(f"Invalid run directory: {result.run_dir}")
            
        # Ensure metrics contains the ticker
        if result.metrics is None:
            result.metrics = {}
            
        # Always add ticker to metrics for easier retrieval later
        result.metrics['ticker'] = result.ticker
            
        # Save the model if it exists
        if result.model is not None and result.best_model_path:
            try:
                result.model.save(result.best_model_path)
                log_info(f"Saved model to {result.best_model_path}")
            except Exception as e:
                log_warning(f"Could not save model to {result.best_model_path}: {e}")
                
        # Save training history
        history_path = os.path.join(result.run_dir, "history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump(result.history, f, indent=2)
            log_info(f"Saved training history to {result.run_dir}/history.json")
        except Exception as e:
            log_warning(f"Could not save training history: {e}")
            
        # Save metrics
        metrics_path = os.path.join(result.run_dir, "metrics.json")
        try:
            with open(metrics_path, 'w') as f:
                json.dump(result.metrics, f, indent=2)
            log_info(f"Saved metrics to {result.run_dir}/metrics.json")
        except Exception as e:
            log_warning(f"Could not save metrics: {e}")
            
        # Save evaluation results if they exist
        if result.evaluation_df is not None and not result.evaluation_df.empty:
            eval_path = os.path.join(result.run_dir, "evaluation_results.csv")
            try:
                result.evaluation_df.to_csv(eval_path, index=False)
                log_info(f"Saved evaluation results to {result.run_dir}/evaluation_results.csv")
            except Exception as e:
                log_warning(f"Could not save evaluation results: {e}")
                
        # Create visualizations
        ModelStorageManager.create_training_visualizations(result, result.run_dir)
    
    @staticmethod
    def create_training_visualizations(result: TrainingResultDTO, run_dir: str) -> None:
        """
        Creates and saves visualizations for a training run.
        
        Args:
            result: The TrainingResultDTO containing run data
            run_dir: Path to the run directory
        """
        if not run_dir or not os.path.exists(run_dir):
            return
            
        # Create loss plot
        if result.history and 'loss' in result.history:
            plt.figure(figsize=(12, 8))
            
            # Training loss
            plt.subplot(2, 1, 1)
            plt.plot(result.history['loss'], label='Training Loss')
            if 'val_loss' in result.history:
                plt.plot(result.history['val_loss'], label='Validation Loss')
            plt.title(f'Loss History for {result.ticker}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # MSE plot
            plt.subplot(2, 1, 2)
            mse_key = None
            for key in ['mse', 'mean_squared_error', 'MSE']:
                if key in result.history:
                    mse_key = key
                    break
                    
            if mse_key:
                plt.plot(result.history[mse_key], label=f'Training {mse_key.upper()}')
                
            val_mse_key = None
            for key in ['val_mse', 'val_mean_squared_error', 'val_MSE']:
                if key in result.history:
                    val_mse_key = key
                    break
                    
            if val_mse_key:
                plt.plot(result.history[val_mse_key], label=f'Validation {val_mse_key.upper()}')
                
            plt.title(f'MSE History for {result.ticker}')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(run_dir, 'training_history.png')
            plt.savefig(plot_path)
            plt.close()
            log_info(f"Saved training history plot to {plot_path}")
    
    @staticmethod
    def load_model(model_path: str) -> Optional[tf.keras.Model]:
        """
        Loads a saved TensorFlow model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            The loaded model or None if loading fails
        """
        if not os.path.exists(model_path):
            log_warning(f"Model file not found: {model_path}")
            return None
            
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            log_error(f"Error loading model from {model_path}: {e}")
            return None
    
    @staticmethod
    def load_training_run(run_dir: str) -> Optional[TrainingResultDTO]:
        """
        Load data for a training run from disk.
        
        Args:
            run_dir (str): Path to the run directory
            
        Returns:
            TrainingResultDTO: The loaded training result
        """
        if not os.path.exists(run_dir):
            raise ValueError(f"Run directory '{run_dir}' does not exist")
            
        # Extract run_id from directory name
        run_id = os.path.basename(run_dir)
        model_dir = os.path.dirname(os.path.dirname(run_dir))  # runs_dir/run_id -> model_dir
        
        # Try to get the ticker from model parameters
        ticker = "unknown"  # Default value if we can't determine the ticker
        params_file = os.path.join(model_dir, "model_params.json")
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    model_params = json.load(f)
                    # Check if the model params have ticker information
                    if 'ticker' in model_params:
                        ticker = model_params['ticker']
            except Exception as e:
                log_warning(f"Warning: Could not load model parameters: {e}")
        
        # Also check if there's a ticker in the run directory name
        run_name = os.path.basename(run_dir)
        if '_' in run_name:
            # Format might be something like "AAPL_001" or "run_AAPL_001"
            parts = run_name.split('_')
            if len(parts) > 1 and not parts[0].isdigit():
                ticker = parts[0]
                
        # Initialize TrainingResultDTO
        result = TrainingResultDTO(
            ticker=ticker,
            run_id=run_id,
            model_dir=model_dir,
            run_dir=run_dir
        )
        
        # Load metrics
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    result.metrics = metrics_data
                    # Try to get ticker from metrics if available
                    if 'ticker' in metrics_data:
                        result.ticker = metrics_data['ticker']
            except Exception as e:
                log_warning(f"Warning: Could not load metrics from {metrics_path}: {e}")
        
        # Load history
        history_path = os.path.join(run_dir, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    result.history = json.load(f)
            except Exception as e:
                log_warning(f"Warning: Could not load history from {history_path}: {e}")
        
        # Load evaluation results
        eval_csv_path = os.path.join(run_dir, "evaluation_results.csv")
        if os.path.exists(eval_csv_path):
            try:
                result.evaluation_df = pd.read_csv(eval_csv_path)
            except Exception as e:
                log_warning(f"Warning: Could not load evaluation results from {eval_csv_path}: {e}")
        
        # Find the model file
        model_files = [f for f in os.listdir(run_dir) if f.endswith('.h5') or f.endswith('.keras')]
        if model_files:
            model_path = os.path.join(run_dir, model_files[0])
            result.best_model_path = model_path
            
            # Try to load the model
            try:
                result.model = ModelStorageManager.load_model(model_path)
            except Exception as e:
                log_warning(f"Warning: Could not load model from {model_path}: {e}")
        
        return result
    
    @staticmethod
    def load_model_from_directory(model_dir, custom_objects=None):
        """
        Load a model from a model directory.
        
        Args:
            model_dir (str): Path to the model directory
            custom_objects (dict, optional): Dictionary of custom objects for model loading
            
        Returns:
            tuple: (tf.keras.Model, dict) - The loaded model and its parameters
        """
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory '{model_dir}' does not exist")
            
        # Load model parameters
        params_file = os.path.join(model_dir, "model_params.json")
        if not os.path.exists(params_file):
            raise ValueError(f"Model parameters file not found in '{model_dir}'")
            
        with open(params_file, 'r') as f:
            model_params = json.load(f)
            
        # Look for model file in best model directory from summary
        best_model_info_path = os.path.join(model_dir, "best_model_info.json")
        if os.path.exists(best_model_info_path):
            with open(best_model_info_path, 'r') as f:
                best_model_info = json.load(f)
                model_path = best_model_info.get('model_path')
                if model_path and os.path.exists(model_path):
                    try:
                        log_info(f"Loading model from best model path: {model_path}")
                        model = ModelStorageManager.load_model(model_path, custom_objects)
                        return model, model_params
                    except Exception as e:
                        log_warning(f"Warning: Error loading best model: {e}")
        
        # Look for model.keras or model.h5 in the model directory
        for model_ext in ['.keras', '.h5']:
            model_path = os.path.join(model_dir, f"model{model_ext}")
            if os.path.exists(model_path):
                try:
                    log_info(f"Loading model from model directory: {model_path}")
                    model = ModelStorageManager.load_model(model_path, custom_objects)
                    return model, model_params
                except Exception as e:
                    log_warning(f"Warning: Error loading model from model directory: {e}")
        
        # Look for model in runs directories
        runs_dir = os.path.join(model_dir, "runs")
        if os.path.exists(runs_dir):
            # Find all run directories
            run_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                       if os.path.isdir(os.path.join(runs_dir, d))]
            
            if run_dirs:
                # Sort by modification time (newest first)
                run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                for run_dir in run_dirs:
                    # Look for model files in each run directory
                    model_files = [f for f in os.listdir(run_dir) 
                                  if os.path.isfile(os.path.join(run_dir, f)) and 
                                  (f.endswith('.h5') or f.endswith('.keras'))]
                    
                    if model_files:
                        # Use the first model file found
                        model_path = os.path.join(run_dir, model_files[0])
                        try:
                            model = ModelStorageManager.load_model(model_path, custom_objects)
                            return model, model_params
                        except Exception as e:
                            log_warning(f"Warning: Error loading model from {model_path}: {e}")
        
        # If no model file found, rebuild the model from parameters
        from model.model_builder import ModelBuilder
        
        # If we have feature info, load that too for proper input shape
        feature_info_path = os.path.join(model_dir, "feature_info.json")
        feature_count = None
        if os.path.exists(feature_info_path):
            try:
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    feature_count = feature_info.get('total_features', 0)
            except Exception as e:
                log_warning(f"Warning: Error loading feature info: {e}")
                
        # Get time window size from model parameters or use default
        n_steps = model_params.get('n_steps', 60)
        
        # Create a model builder instance
        builder = ModelBuilder()
        
        # Ensure model_params has all required parameters with defaults if missing
        default_params = ModelDefinition.get_default_model_parameters()
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        # Build the model with the right input shape
        model = builder.build_model(
            input_shape=(n_steps, feature_count or 10),  # Default to 10 features if unknown
            **model_params
        )
        return model, model_params
    
    @staticmethod
    def generate_model_summary(model_dir: str) -> Dict[str, Any]:
        """
        Generates a summary of all training runs for a model.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            Dictionary containing summary information
        """
        log_info(f"Creating model summary for {model_dir}")
        
        # Gather information from all run directories
        runs_data = []
        run_dirs = []
        
        # Find all run directories inside the "runs" subdirectory
        runs_dir = os.path.join(model_dir, "runs")
        if not os.path.exists(runs_dir):
            raise ValueError(f"No runs directory found in {model_dir}")
            
        # Look for numeric run directories inside the runs directory
        for item in os.listdir(runs_dir):
            item_path = os.path.join(runs_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid run directory
                if os.path.exists(os.path.join(item_path, "metrics.json")):
                    run_dirs.append(item_path)
        
        if not run_dirs:
            raise ValueError(f"No training runs found in {runs_dir}")
            
        log_info(f"Found {len(run_dirs)} training runs")
            
        # Load each training run
        for run_dir in run_dirs:
            run_result = ModelStorageManager.load_training_run(run_dir)
            if run_result:
                runs_data.append({
                    "run_number": int(run_result.run_id) if run_result.run_id and run_result.run_id.isdigit() else -1,
                    "metrics": run_result.metrics,
                    "history": run_result.history,
                    "model_path": run_result.best_model_path,
                    "run_dir": run_result.run_dir
                })
        
        # Create summary rows from run data
        summary_rows = []
        for run_data in runs_data:
            summary_row = ModelStorageManager._generate_summary_row(run_data)
            summary_rows.append(summary_row)
            
        # Find best run based on MSE
        valid_mse_runs = [row for row in summary_rows if not math.isnan(row['MSE'])]
        best_run_number = None
        best_mse = None
        
        if valid_mse_runs:
            best_run = min(valid_mse_runs, key=lambda x: x['MSE'])
            best_run_number = best_run['Run']
            best_mse = best_run['MSE']
            log_info(f"Best run: {best_run_number} with MSE: {best_mse}")
        else:
            log_info("No valid runs with MSE found")
            
        # Create comparative plots if we have multiple runs
        if len(runs_data) > 1:
            ModelStorageManager._create_comparative_plots(runs_data, model_dir)
            ModelStorageManager._create_training_progress_comparison(runs_data, model_dir)
            
        # Save summary to CSV
        summary_path = os.path.join(model_dir, "model_summary.csv")
        try:
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            log_info(f"Saved model summary to {summary_path}")
        except Exception as e:
            log_warning(f"Warning: Error saving model summary: {e}")
            
        # Return summary info
        return {
            "total_runs": len(runs_data),
            "valid_runs": len([r for r in runs_data if r.get('metrics') and 'evaluation' in r.get('metrics', {})]),
            "best_run_number": best_run_number,
            "best_mse": best_mse,
            "summary_path": summary_path,
            "model_dir": model_dir
        }
    
    @staticmethod
    def _generate_summary_row(run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary row for a particular run
        
        Args:
            run_data: Dictionary containing information about the run
            
        Returns:
            Dictionary containing summary metrics for the run
        """
        run_number = run_data['run_number']
        metrics = run_data.get('metrics', {})
        
        # Safely extract metrics, providing defaults for missing or None values
        final_train_loss = metrics.get('final_train_loss', float('nan'))
        final_val_loss = metrics.get('final_val_loss', float('nan'))
        
        # Extract MSE and ensure it's a float - check both possible keys
        evaluation = metrics.get('evaluation', {})
        # First try to get 'Overall_MSE' (used in our training code)
        mse = evaluation.get('Overall_MSE')
        # If not found, try 'mse' (lowercase version)
        if mse is None:
            mse = evaluation.get('mse')
        # If still not found, check for any key containing 'mse' or 'MSE'
        if mse is None:
            for key in evaluation:
                if 'mse' in key.lower():
                    mse = evaluation[key]
                    break
                    
        # Convert to float or handle None values
        if mse is None:
            mse = float('nan')
        else:
            try:
                mse = float(mse)
            except (ValueError, TypeError):
                mse = float('nan')
                
        # Extract best epoch safely
        best_epoch = metrics.get('best_epoch')
        if best_epoch is None:
            best_epoch = -1
        else:
            try:
                best_epoch = int(best_epoch)
            except (ValueError, TypeError):
                best_epoch = -1
        
        # Make sure we have the model path
        model_path = run_data.get('model_path', '')
        
        return {
            'Run': run_number,
            'Final Train Loss': final_train_loss,
            'Final Val Loss': final_val_loss,
            'MSE': mse,
            'Best Epoch': best_epoch,
            'Model Path': model_path
        }
    
    @staticmethod
    def _create_comparative_plots(runs_data: List[Dict[str, Any]], model_dir: str) -> None:
        """
        Create comparative plots across all runs.
        
        Args:
            runs_data: List of dictionaries containing run information
            model_dir: Path to the model directory
        """
        if not runs_data:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot MSE comparison
        mse_values = []
        run_numbers = []
        for run in runs_data:
            run_mse = run['metrics'].get('evaluation', {}).get('Overall_MSE')
            if run_mse is not None:  # Only include runs with valid MSE
                mse_values.append(run_mse)
                run_numbers.append(run['run_number'])
        
        if mse_values and run_numbers:  # Only plot if we have valid data
            plt.subplot(2, 2, 1)
            plt.bar(run_numbers, mse_values)
            plt.title('MSE Comparison Across Runs')
            plt.xlabel('Run Number')
            plt.ylabel('MSE')
            plt.xticks(run_numbers)
        
        # Plot final training loss comparison
        train_losses = []
        train_run_numbers = []
        for run in runs_data:
            train_loss = run['metrics'].get('final_train_loss')
            if train_loss is not None:  # Only include runs with valid loss
                train_losses.append(train_loss)
                train_run_numbers.append(run['run_number'])
        
        if train_losses and train_run_numbers:  # Only plot if we have valid data
            plt.subplot(2, 2, 2)
            plt.bar(train_run_numbers, train_losses)
            plt.title('Final Training Loss Comparison')
            plt.xlabel('Run Number')
            plt.ylabel('Training Loss')
            plt.xticks(train_run_numbers)
        
        # Plot final validation loss comparison
        val_losses = []
        val_run_numbers = []
        for run in runs_data:
            val_loss = run['metrics'].get('final_val_loss')
            if val_loss is not None:  # Only include runs with valid loss
                val_losses.append(val_loss)
                val_run_numbers.append(run['run_number'])
        
        if val_losses and val_run_numbers:  # Only plot if we have valid data
            plt.subplot(2, 2, 3)
            plt.bar(val_run_numbers, val_losses)
            plt.title('Final Validation Loss Comparison')
            plt.xlabel('Run Number')
            plt.ylabel('Validation Loss')
            plt.xticks(val_run_numbers)
        
        # Plot epochs comparison
        epochs_values = []
        epochs_run_numbers = []
        for run in runs_data:
            epochs = run['metrics'].get('best_epoch')
            if epochs is not None:  # Only include runs with valid epochs
                epochs_values.append(epochs)
                epochs_run_numbers.append(run['run_number'])
        
        if epochs_values and epochs_run_numbers:  # Only plot if we have valid data
            plt.subplot(2, 2, 4)
            plt.bar(epochs_run_numbers, epochs_values)
            plt.title('Training Epochs Comparison')
            plt.xlabel('Run Number')
            plt.ylabel('Epochs')
            plt.xticks(epochs_run_numbers)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "runs_comparison.png"))
        plt.close()
    
    @staticmethod
    def _create_training_progress_comparison(runs_data: List[Dict[str, Any]], model_dir: str) -> None:
        """
        Create training progress comparison plots across all runs.
        
        Args:
            runs_data: List of dictionaries containing run information
            model_dir: Path to the model directory
        """
        # Only plot runs with history data
        runs_with_history = [run for run in runs_data if run.get('history')]
        if not runs_with_history:
            return
            
        # Create loss progress comparison
        plt.figure(figsize=(14, 7))
        
        # Training loss plot
        has_training_loss = False
        plt.subplot(1, 2, 1)
        for run in runs_with_history:
            run_number = run['run_number']
            history = run.get('history', {})
            if 'loss' in history and history['loss']:  # Check if loss data exists and is not empty
                try:
                    epochs = range(1, len(history['loss']) + 1)
                    plt.plot(epochs, history['loss'], label=f'Run {run_number}')
                    has_training_loss = True
                except (TypeError, ValueError) as e:
                    log_warning(f"Warning: Error plotting loss for run {run_number}: {e}")
        
        if has_training_loss:
            plt.title('Training Loss Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Validation loss plot
        has_val_loss = False
        plt.subplot(1, 2, 2)
        for run in runs_with_history:
            run_number = run['run_number']
            history = run.get('history', {})
            if 'val_loss' in history and history['val_loss']:  # Check if val_loss data exists and is not empty
                try:
                    epochs = range(1, len(history['val_loss']) + 1)
                    plt.plot(epochs, history['val_loss'], label=f'Run {run_number}')
                    has_val_loss = True
                except (TypeError, ValueError) as e:
                    log_warning(f"Warning: Error plotting val_loss for run {run_number}: {e}")
        
        if has_val_loss:
            plt.title('Validation Loss Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Only save if we have at least one valid plot
        if has_training_loss or has_val_loss:
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "training_progress_comparison.png"))
        plt.close()
        
        # Create MSE progress comparison if metrics available
        mse_keys = ['mean_squared_error', 'mse', 'MSE']
        val_mse_keys = ['val_mean_squared_error', 'val_mse', 'val_MSE']
        
        # Find which key is used in the history
        mse_key = None
        val_mse_key = None
        
        for key in mse_keys:
            if any(key in run.get('history', {}) and run.get('history', {}).get(key) for run in runs_with_history):
                mse_key = key
                break
                
        for key in val_mse_keys:
            if any(key in run.get('history', {}) and run.get('history', {}).get(key) for run in runs_with_history):
                val_mse_key = key
                break
        
        if mse_key or val_mse_key:
            plt.figure(figsize=(14, 7))
            
            # Training MSE plot
            has_mse = False
            if mse_key:
                plt.subplot(1, 2, 1)
                for run in runs_with_history:
                    run_number = run['run_number']
                    history = run.get('history', {})
                    if mse_key in history and history[mse_key]:  # Check if MSE data exists and is not empty
                        try:
                            epochs = range(1, len(history[mse_key]) + 1)
                            plt.plot(epochs, history[mse_key], label=f'Run {run_number}')
                            has_mse = True
                        except (TypeError, ValueError) as e:
                            log_warning(f"Warning: Error plotting MSE for run {run_number}: {e}")
                
                if has_mse:
                    plt.title('Training MSE Progress')
                    plt.xlabel('Epoch')
                    plt.ylabel('MSE')
                    plt.legend()
                    plt.grid(True)
            
            # Validation MSE plot
            has_val_mse = False
            if val_mse_key:
                plt.subplot(1, 2, 2)
                for run in runs_with_history:
                    run_number = run['run_number']
                    history = run.get('history', {})
                    if val_mse_key in history and history[val_mse_key]:  # Check if val_MSE data exists and is not empty
                        try:
                            epochs = range(1, len(history[val_mse_key]) + 1)
                            plt.plot(epochs, history[val_mse_key], label=f'Run {run_number}')
                            has_val_mse = True
                        except (TypeError, ValueError) as e:
                            log_warning(f"Warning: Error plotting val_MSE for run {run_number}: {e}")
                
                if has_val_mse:
                    plt.title('Validation MSE Progress')
                    plt.xlabel('Epoch')
                    plt.ylabel('MSE')
                    plt.legend()
                    plt.grid(True)
            
            # Only save if we have at least one valid plot
            if has_mse or has_val_mse:
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, "mse_progress_comparison.png"))
            plt.close() 