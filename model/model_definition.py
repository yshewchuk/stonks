#!/usr/bin/env python
"""
ModelDefinition class for centralizing model parameter definitions and validation.
"""

from typing import Dict, List, Any, Set, Optional, Union

class ModelDefinition:
    """
    Centralizes the definition and validation of model parameters, training parameters, and features.
    
    This class serves as a single source of truth for:
    1. Available model parameters and their allowed values
    2. Available training parameters and their allowed values
    3. Parameter validation rules
    
    It's used by ModelBuilder, ModelIdentifier, and training scripts to ensure
    consistency in how models are defined and validated.
    """
    
    # Total number of available features
    TOTAL_FEATURE_COUNT = 330
    
    # Model parameter options and their allowed values
    MODEL_PARAMETER_OPTIONS = {
        # CNN Layer Parameters
        'cnn_layers': [0, 1, 2, 3],  # Number of CNN layers (0 for no CNN)
        'cnn_filters': [32, 64, 128, 256],  # Number of filters per CNN layer
        'cnn_kernel_size': [3, 5, 7, 9],  # Kernel size for CNN layers
        'cnn_pooling': [True, False],  # Whether to use pooling after CNN layers
        
        # LSTM Layer Parameters
        'lstm_layers': [1, 2, 3, 4],  # Number of LSTM layers
        'lstm_units': [32, 64, 128, 256],  # Number of units per LSTM layer
        
        # Regularization Parameters
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'recurrent_dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'l2_reg': [0.0, 0.001, 0.01, 0.1],
        'use_batch_norm': [True, False],
        
        # Training Parameters
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
        'activation': ['relu', 'tanh', 'sigmoid'],
        
        # Time Series Parameters
        'n_steps': [60]  # Number of time steps in the input sequence (only 60 supported)
    }
    
    # Training parameter options and their allowed values
    TRAINING_PARAMETER_OPTIONS = {
        'batch_size': [8, 16, 32, 64],
        'reduce_lr_patience': [3, 5, 7, 10]
    }
    
    @classmethod
    def get_default_model_parameters(cls) -> Dict[str, Any]:
        """
        Get default model parameters.
        
        Returns:
            Default model parameters with middle-of-the-road values
        """
        return {
            # CNN parameters
            'cnn_layers': 0,                    # No CNN by default
            'cnn_filters': 64,                  # Default filter size
            'cnn_kernel_size': 3,               # Default kernel size
            'cnn_pooling': False,               # No pooling by default
            
            # LSTM parameters
            'lstm_layers': 2,                   # Two LSTM layers
            'lstm_units': 128,                  # Units per LSTM layer
            
            # Regularization parameters
            'dropout_rate': 0.3,                # Dropout rate
            'recurrent_dropout_rate': 0.1,      # Recurrent dropout rate
            'l2_reg': 0.0,                      # L2 regularization
            'use_batch_norm': False,            # Batch normalization
            
            # Training parameters
            'learning_rate': 0.001,             # Learning rate
            'activation': 'relu',               # Activation function
            
            # Time Series parameters
            'n_steps': 60                        # Input window size
        }
    
    @classmethod
    def get_default_training_parameters(cls) -> Dict[str, Any]:
        """
        Get default training parameters.
        
        Returns:
            Default training parameters
        """
        return {
            'batch_size': 32,
            'reduce_lr_patience': 10
        }
    
    @classmethod
    def validate_model_parameters(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model parameters against allowed values.
        
        Args:
            parameters: Dictionary of model parameters
            
        Returns:
            Validated parameters (same as input if valid)
            
        Raises:
            ValueError: If any parameter is invalid
        """
        return cls._validate_parameters(parameters, cls.MODEL_PARAMETER_OPTIONS)
    
    @classmethod
    def validate_training_parameters(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training parameters against allowed values.
        
        Args:
            parameters: Dictionary of training parameters
            
        Returns:
            Validated parameters (same as input if valid)
            
        Raises:
            ValueError: If any parameter is invalid
        """
        return cls._validate_parameters(parameters, cls.TRAINING_PARAMETER_OPTIONS)
    
    @classmethod
    def validate_feature_indexes(cls, selected_indexes: Set[int]) -> Set[int]:
        """
        Validate that selected feature indexes are within the allowed range.
        
        Args:
            selected_indexes: Set of feature indexes to validate
            
        Returns:
            Validated indexes (same as input if valid)
            
        Raises:
            ValueError: If any selected index is not within the valid range
        """
        if not selected_indexes:
            raise ValueError("No feature indexes selected")
            
        invalid_indexes = [idx for idx in selected_indexes if idx < 0 or idx >= cls.TOTAL_FEATURE_COUNT]
        if invalid_indexes:
            raise ValueError(f"Invalid feature indexes (must be between 0 and {cls.TOTAL_FEATURE_COUNT-1}): {', '.join(map(str, invalid_indexes))}")
        return selected_indexes
    
    @classmethod
    def _validate_parameters(cls, parameters: Dict[str, Any], allowed_options: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Validate parameters against allowed options.
        
        Args:
            parameters: Dictionary of parameters to validate
            allowed_options: Dictionary of parameter names to allowed values
            
        Returns:
            Validated parameters (same as input if valid)
            
        Raises:
            ValueError: If any parameter is invalid
        """
        errors = []
        
        # Check for missing parameters
        missing_params = set(allowed_options.keys()) - set(parameters.keys())
        if missing_params:
            errors.append(f"Missing parameters: {', '.join(missing_params)}")
        
        # Check for invalid values
        for param_name, param_value in parameters.items():
            if param_name in allowed_options:
                allowed_values = allowed_options[param_name]
                if param_value not in allowed_values:
                    errors.append(
                        f"Invalid value for '{param_name}': {param_value}. "
                        f"Allowed values: {allowed_values}"
                    )
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return parameters 