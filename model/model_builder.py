import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

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
    
    @staticmethod
    def build_price_prediction_model(model_params):
        """
        Build a price prediction model with the specified parameters.
        
        Args:
            model_params (dict): A dictionary containing model configuration parameters:
                - n_steps (int): Number of time steps in input sequence
                - n_features_total (int): Number of features in input data
                - n_output_probabilities (int): Number of output probabilities
                - n_units (int): Base number of units for LSTM layers
                - dropout_rate (float): Dropout rate for LSTM and Dense layers
                - cnn_filters (list): List of filter counts for CNN layers. Empty list means no CNN layers.
                - cnn_kernel_sizes (list): List of kernel sizes for CNN layers. Must be same length as cnn_filters.
                - lstm_layers (int): Number of LSTM layers
                - dense_layers (list): List of units for Dense layers
                - activation (str): Activation function for Dense layers
                - recurrent_dropout_rate (float): Recurrent dropout rate for LSTM layers
                - l2_reg (float): L2 regularization factor
                - learning_rate (float): Learning rate for optimizer
                
        Returns:
            tf.keras.Model: A compiled TensorFlow model
            
        Raises:
            ValueError: If any required parameters are missing or invalid
        """
        # Required parameters - validate that they exist
        required_params = [
            'n_steps', 
            'n_features_total', 
            'n_units', 
            'n_output_probabilities',
            'dropout_rate', 
            'cnn_filters', 
            'cnn_kernel_sizes',
            'lstm_layers',
            'dense_layers', 
            'activation',
            'recurrent_dropout_rate',
            'l2_reg',
            'learning_rate'
        ]
        
        # Check for missing parameters
        missing_params = [param for param in required_params if param not in model_params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}. "
                            f"Use ModelBuilder.get_default_model_params() to get default values.")
        
        # Validate CNN parameters
        cnn_filters = model_params['cnn_filters']
        cnn_kernel_sizes = model_params['cnn_kernel_sizes']
        
        # Check if arrays are both empty or both non-empty
        if bool(cnn_filters) != bool(cnn_kernel_sizes):
            raise ValueError("Either both cnn_filters and cnn_kernel_sizes must be non-empty, or both must be empty")
            
        # If CNN layers will be used, validate that arrays have the same length
        if cnn_filters:
            if len(cnn_filters) != len(cnn_kernel_sizes):
                raise ValueError(f"cnn_filters and cnn_kernel_sizes must have the same length. "
                                f"Got cnn_filters length {len(cnn_filters)} and "
                                f"cnn_kernel_sizes length {len(cnn_kernel_sizes)}")
        
        # Extract parameters for readability
        n_steps = model_params['n_steps']
        n_features_total = model_params['n_features_total']
        n_units = model_params['n_units']
        n_output_probabilities = model_params['n_output_probabilities']
        dropout_rate = model_params['dropout_rate']
        lstm_layers = model_params['lstm_layers']
        dense_layers = model_params['dense_layers']
        activation = model_params['activation']
        recurrent_dropout_rate = model_params['recurrent_dropout_rate']
        l2_reg = model_params['l2_reg']
        learning_rate = model_params['learning_rate']
        use_batch_norm = model_params.get('use_batch_norm', False)  # Optional
        
        # Create the model
        model = tf.keras.Sequential()
        
        # Add the input layer
        model.add(tf.keras.layers.Input(shape=(n_steps, n_features_total)))
        
        # Optional CNN layers - use if arrays are non-empty
        if cnn_filters and cnn_kernel_sizes:
            for i in range(len(cnn_filters)):
                model.add(Conv1D(
                    filters=cnn_filters[i],
                    kernel_size=cnn_kernel_sizes[i],
                    padding='same',
                    activation=activation,
                    kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
                ))
                model.add(MaxPooling1D(pool_size=2, padding='same'))
                if dropout_rate > 0:
                    model.add(Dropout(dropout_rate))
        
        # LSTM layers
        for i in range(lstm_layers):
            return_sequences = i < lstm_layers - 1  # Return sequences for all but last LSTM layer
            model.add(LSTM(
                units=n_units,
                return_sequences=return_sequences,
                dropout=dropout_rate if dropout_rate > 0 else 0.0,
                recurrent_dropout=recurrent_dropout_rate if recurrent_dropout_rate > 0 else 0.0,
                kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
            ))
            
            # Optional batch normalization after LSTM layers
            if use_batch_norm:
                model.add(BatchNormalization())
        
        # Dense layers
        for units in dense_layers:
            model.add(Dense(
                units=units,
                activation=activation,
                kernel_regularizer=l2(l2_reg) if l2_reg > 0 else None
            ))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(
            units=n_output_probabilities,
            activation='sigmoid',  # Sigmoid for probability outputs
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
    def get_default_model_params():
        """
        Get default model parameters for a price prediction model.
        This is the single source of truth for model defaults.
        
        Returns:
            dict: Default model parameters
        """
        return {
            'n_steps': 60,                # Default window size
            'n_units': 128,               # Base number of LSTM units
            'dropout_rate': 0.2,          # Dropout rate
            'learning_rate': 0.001,       # Learning rate for Adam optimizer
            'n_output_probabilities': 21, # Number of output probabilities
            'cnn_filters': [],            # Empty means no CNN layers by default
            'cnn_kernel_sizes': [],       # Empty means no CNN layers by default
            'lstm_layers': 2,             # Use 2 LSTM layers by default
            'dense_layers': [64],         # One dense layer with 64 units
            'activation': 'relu',         # ReLU activation
            'recurrent_dropout_rate': 0.0, # No recurrent dropout by default
            'l2_reg': 0.0,                # No L2 regularization by default
            'use_batch_norm': False       # No batch normalization by default
        } 