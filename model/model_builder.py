import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any, Optional, Tuple

from .model_definition import ModelDefinition

class ModelBuilder:
    """
    A flexible builder class for constructing neural network models with various architectures.
    
    The ModelBuilder can create models with:
    - Optional CNN layers
    - Variable number of LSTM layers
    - Configurable activation functions
    - L2 regularization
    - Configurable dropout and recurrent dropout
    
    This class is responsible ONLY for building models, not storing or loading them.
    Use ModelStorageManager for storage operations.
    """
    
    def __init__(self):
        """Initialize the ModelBuilder class."""
        pass
    
    def build_model(
        self, 
        input_shape: Tuple[int, int],
        cnn_layers: int,
        cnn_filters: int,
        cnn_kernel_size: int,
        cnn_pooling: bool,
        lstm_layers: int, 
        lstm_units: int,
        dropout_rate: float,
        recurrent_dropout_rate: float,
        l2_reg: float,
        use_batch_norm: bool,
        learning_rate: float,
        activation: str,
        n_steps: int
    ) -> tf.keras.Model:
        """
        Build a price prediction model with the specified parameters.
        
        Args:
            input_shape: Tuple of (time steps, features) for input data
            cnn_layers: Number of CNN layers (0 for no CNN)
            cnn_filters: Number of filters for each CNN layer
            cnn_kernel_size: Kernel size for CNN layers
            cnn_pooling: Whether to use pooling after CNN layers
            lstm_layers: Number of LSTM layers
            lstm_units: Number of units per LSTM layer
            dropout_rate: Dropout rate for LSTM and Dense layers
            recurrent_dropout_rate: Recurrent dropout rate for LSTM layers
            l2_reg: L2 regularization factor
            use_batch_norm: Whether to use batch normalization
            learning_rate: Learning rate for optimizer
            activation: Activation function for layers
            n_steps: Number of time steps
                
        Returns:
            tf.keras.Model: A compiled TensorFlow model
        """
        # Validate parameters using ModelDefinition
        params = {
            'cnn_layers': cnn_layers,
            'cnn_filters': cnn_filters,
            'cnn_kernel_size': cnn_kernel_size,
            'cnn_pooling': cnn_pooling,
            'lstm_layers': lstm_layers,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'recurrent_dropout_rate': recurrent_dropout_rate,
            'l2_reg': l2_reg,
            'use_batch_norm': use_batch_norm,
            'learning_rate': learning_rate,
            'activation': activation,
            'n_steps': n_steps
        }
        ModelDefinition.validate_model_parameters(params)
        
        # Create the model
        model = tf.keras.Sequential()
        
        # Add the input layer
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Add CNN layers if specified
        if cnn_layers > 0:
            for _ in range(cnn_layers):
                model.add(Conv1D(
                    filters=cnn_filters,
                    kernel_size=cnn_kernel_size,
                    padding='same',
                    activation=activation,
                    kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
                ))
                
                # Add batch normalization if specified
                if use_batch_norm:
                    model.add(BatchNormalization())
                
                # Add pooling if specified
                if cnn_pooling:
                    model.add(MaxPooling1D(pool_size=2, padding='same'))
                
                # Add dropout if specified
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
        
        # Add LSTM layers
        for i in range(lstm_layers):
            return_sequences = i < lstm_layers - 1  # Return sequences for all but last LSTM layer
            model.add(LSTM(
                units=lstm_units,
                return_sequences=return_sequences,
                dropout=dropout_rate if dropout_rate > 0 else 0.0,
                recurrent_dropout=recurrent_dropout_rate if recurrent_dropout_rate > 0 else 0.0,
                kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
            ))
            
            # Add batch normalization after LSTM if specified
            if use_batch_norm:
                model.add(BatchNormalization())
        
        # Output layer - single unit with linear activation for regression
        model.add(Dense(
            units=21,
            activation='relu',
            kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
        ))
        
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error loss for regression
            metrics=['mse', 'mae']
        )
        
        return model
    
    @staticmethod
    def get_default_model_params() -> Dict[str, Any]:
        """
        Get default model parameters for a price prediction model.
        
        Returns:
            dict: Default model parameters
        """
        return ModelDefinition.get_default_model_parameters() 