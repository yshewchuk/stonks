import numpy as np
import base64
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import OrderedDict

class ModelIdentifier:
    """
    Handles the creation and parsing of deterministic model identifiers based on:
    1. Model architecture parameters
    2. Input features
    3. Training parameters
    
    The identifier is created by encoding bit arrays representing parameter choices,
    making it possible to uniquely identify models with specific configurations.
    """
    
    # Define the possible values for each model parameter
    MODEL_PARAM_OPTIONS = {
        'n_units': [32, 64, 96, 128, 160, 192, 224, 256],  # 3 bits
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 3 bits
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],  # 3 bits
        'lstm_layers': [1, 2, 3, 4],  # 2 bits
        'recurrent_dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # 3 bits
        'l2_reg': [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],  # 3 bits
        'use_batch_norm': [False, True],  # 1 bit
        'activation': ['relu', 'tanh', 'selu', 'elu']  # 2 bits
    }
    
    # Constant parameters that don't need encoding (only one value supported)
    CONSTANT_PARAMS = {
        'n_steps': 60
    }
    
    # Array parameters with variable-length arrays
    ARRAY_PARAM_OPTIONS = {
        'cnn_filters': list(range(16, 257, 16)),  # 4 bits per item (16 values)
        'cnn_kernel_sizes': [1, 2, 3, 5, 7, 9],  # 3 bits per item (6 values)
        'dense_layers': list(range(16, 257, 16))  # 4 bits per item (16 values)
    }
    
    # Training parameters options
    TRAINING_PARAM_OPTIONS = {
        'batch_size': [8, 16, 24, 32, 48, 64, 96, 128]  # 3 bits
    }
    
    def __init__(self):
        # Calculate bit sizes for each parameter
        self.param_bit_sizes = {
            param: self._calculate_bit_size(len(options))
            for param, options in self.MODEL_PARAM_OPTIONS.items()
        }
        
        self.array_param_bit_sizes = {
            param: self._calculate_bit_size(len(options))
            for param, options in self.ARRAY_PARAM_OPTIONS.items()
        }
        
        self.training_param_bit_sizes = {
            param: self._calculate_bit_size(len(options))
            for param, options in self.TRAINING_PARAM_OPTIONS.items()
        }
    
    def _calculate_bit_size(self, num_options: int) -> int:
        """
        Calculate the number of bits needed to represent a certain number of options.
        
        Args:
            num_options: Number of possible values for the parameter
            
        Returns:
            Number of bits needed to represent all options
        """
        return max(1, int(np.ceil(np.log2(num_options))))
    
    def _get_param_index(self, param: str, value: Any) -> int:
        """
        Get the index of a parameter value in its options array.
        
        Args:
            param: Parameter name
            value: Parameter value
            
        Returns:
            Index in the options array, or 0 if not found
        """
        if param in self.MODEL_PARAM_OPTIONS:
            try:
                return self.MODEL_PARAM_OPTIONS[param].index(value)
            except ValueError:
                # If the value isn't in the list, find the closest match
                options = self.MODEL_PARAM_OPTIONS[param]
                if isinstance(value, (int, float)):
                    closest_idx = min(range(len(options)), key=lambda i: abs(options[i] - value))
                    return closest_idx
                return 0
        
        elif param in self.TRAINING_PARAM_OPTIONS:
            try:
                return self.TRAINING_PARAM_OPTIONS[param].index(value)
            except ValueError:
                # Find closest value for numeric parameters
                options = self.TRAINING_PARAM_OPTIONS[param]
                if isinstance(value, (int, float)):
                    closest_idx = min(range(len(options)), key=lambda i: abs(options[i] - value))
                    return closest_idx
                return 0
                
        elif param in self.ARRAY_PARAM_OPTIONS:
            try:
                return self.ARRAY_PARAM_OPTIONS[param].index(value)
            except ValueError:
                # Find closest value for numeric parameters
                options = self.ARRAY_PARAM_OPTIONS[param]
                if isinstance(value, (int, float)):
                    closest_idx = min(range(len(options)), key=lambda i: abs(options[i] - value))
                    return closest_idx
                return 0
                
        return 0  # Default to first option if not found
    
    def _param_value_to_bits(self, param: str, value: Any) -> List[bool]:
        """
        Convert a parameter value to a bit array.
        
        Args:
            param: Parameter name
            value: Parameter value
            
        Returns:
            List of booleans representing the bits
        """
        index = self._get_param_index(param, value)
        
        if param in self.MODEL_PARAM_OPTIONS:
            bit_size = self.param_bit_sizes[param]
        elif param in self.TRAINING_PARAM_OPTIONS:
            bit_size = self.training_param_bit_sizes[param]
        elif param in self.ARRAY_PARAM_OPTIONS:
            bit_size = self.array_param_bit_sizes[param]
        else:
            bit_size = 1
            
        # Convert to binary and pad with zeros
        bits = [(index >> i) & 1 for i in range(bit_size)]
        return bits
    
    def _array_to_bits(self, param: str, values: List[Any]) -> List[bool]:
        """
        Convert an array parameter to a bit array.
        
        Args:
            param: Parameter name
            values: List of parameter values
            
        Returns:
            List of booleans representing the bits
        """
        # First byte encodes the length of the array (up to 255)
        length_bits = [(min(len(values), 255) >> i) & 1 for i in range(8)]
        
        # Rest of the bytes encode the values
        value_bits = []
        for value in values[:255]:  # Limit to 255 values
            value_bits.extend(self._param_value_to_bits(param, value))
            
        return length_bits + value_bits
    
    def _features_to_bits(self, included_features: Set[int], total_features: int) -> List[bool]:
        """
        Convert a set of included features to a bit array.
        
        Args:
            included_features: Set of indices of included features
            total_features: Total number of features available
            
        Returns:
            List of booleans representing the bits
        """
        return [i in included_features for i in range(total_features)]
    
    def _bits_to_bytes(self, bits: List[bool]) -> bytes:
        """
        Convert a bit array to bytes.
        
        Args:
            bits: List of booleans representing the bits
            
        Returns:
            Bytes object
        """
        # Pad the bits to a multiple of 8
        padded_bits = bits + [False] * ((8 - len(bits) % 8) % 8)
        
        # Convert to bytes
        result = bytearray()
        for i in range(0, len(padded_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(padded_bits) and padded_bits[i + j]:
                    byte |= (1 << j)
            result.append(byte)
            
        return bytes(result)
    
    def _bytes_to_base32(self, data: bytes) -> str:
        """
        Convert bytes to a base32 string.
        
        Args:
            data: Bytes to convert
            
        Returns:
            Base32 encoded string
        """
        return base64.b32encode(data).decode('ascii').rstrip('=')
    
    def create_model_identifier(self, 
                              model_params: Dict[str, Any], 
                              included_features: Set[int],
                              total_features: int,
                              training_params: Dict[str, Any]) -> str:
        """
        Create a model identifier from model parameters, features, and training parameters.
        
        Args:
            model_params: Dictionary of model parameters
            included_features: Set of indices of included features
            total_features: Total number of features available
            training_params: Dictionary of training parameters
            
        Returns:
            Model identifier string
        """
        # Encode model parameters
        model_param_bits = []
        for param, options in self.MODEL_PARAM_OPTIONS.items():
            if param in model_params:
                model_param_bits.extend(self._param_value_to_bits(param, model_params[param]))
            else:
                # Use default value (first option) if parameter is not provided
                model_param_bits.extend([False] * self.param_bit_sizes[param])
        
        # Encode array parameters
        array_param_bits = []
        for param in self.ARRAY_PARAM_OPTIONS:
            if param in model_params and model_params[param]:
                array_param_bits.extend(self._array_to_bits(param, model_params[param]))
            else:
                # Empty array (length 0)
                array_param_bits.extend([False] * 8)
        
        # Encode features
        feature_bits = self._features_to_bits(included_features, total_features)
        
        # Encode training parameters
        training_param_bits = []
        for param, options in self.TRAINING_PARAM_OPTIONS.items():
            if param in training_params:
                training_param_bits.extend(self._param_value_to_bits(param, training_params[param]))
            else:
                # Use default value (first option) if parameter is not provided
                training_param_bits.extend([False] * self.training_param_bit_sizes[param])
        
        # Convert bits to bytes
        model_param_bytes = self._bits_to_bytes(model_param_bits)
        array_param_bytes = self._bits_to_bytes(array_param_bits)
        feature_bytes = self._bits_to_bytes(feature_bits)
        training_param_bytes = self._bits_to_bytes(training_param_bits)
        
        # Convert bytes to base32
        model_param_base32 = self._bytes_to_base32(model_param_bytes)
        array_param_base32 = self._bytes_to_base32(array_param_bytes)
        feature_base32 = self._bytes_to_base32(feature_bytes)
        training_param_base32 = self._bytes_to_base32(training_param_bytes)
        
        # Combine and return
        return f"{model_param_base32}_{array_param_base32}_{feature_base32}_{training_param_base32}"
    
    def _base32_to_bytes(self, base32_str: str) -> bytes:
        """
        Convert a base32 string to bytes.
        
        Args:
            base32_str: Base32 encoded string
            
        Returns:
            Decoded bytes
        """
        # Add padding if necessary
        padding = '=' * ((8 - len(base32_str) % 8) % 8)
        return base64.b32decode(base32_str + padding)
    
    def _bytes_to_bits(self, data: bytes) -> List[bool]:
        """
        Convert bytes to a bit array.
        
        Args:
            data: Bytes to convert
            
        Returns:
            List of booleans representing the bits
        """
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1 == 1)
        return bits
    
    def decode_model_identifier(self, identifier: str) -> Dict[str, Any]:
        """
        Decode a model identifier into model parameters, features, and training parameters.
        
        Args:
            identifier: Model identifier string
            
        Returns:
            Dictionary containing decoded parameters
        """
        try:
            # Split the identifier
            parts = identifier.split('_')
            if len(parts) != 4:
                raise ValueError("Invalid identifier format")
                
            model_param_base32, array_param_base32, feature_base32, training_param_base32 = parts
            
            # Convert to bytes and then to bits
            model_param_bits = self._bytes_to_bits(self._base32_to_bytes(model_param_base32))
            array_param_bits = self._bytes_to_bits(self._base32_to_bytes(array_param_base32))
            feature_bits = self._bytes_to_bits(self._base32_to_bytes(feature_base32))
            training_param_bits = self._bytes_to_bits(self._base32_to_bytes(training_param_base32))
            
            # Decode model parameters
            model_params = {}
            bit_index = 0
            for param, options in self.MODEL_PARAM_OPTIONS.items():
                bit_size = self.param_bit_sizes[param]
                
                if bit_index + bit_size <= len(model_param_bits):
                    # Extract bits for this parameter
                    param_bits = model_param_bits[bit_index:bit_index+bit_size]
                    bit_index += bit_size
                    
                    # Convert bits to index
                    index = 0
                    for i, bit in enumerate(param_bits):
                        if bit:
                            index |= (1 << i)
                    
                    # Get parameter value
                    if index < len(options):
                        model_params[param] = options[index]
                
            # Decode array parameters
            array_params = {}
            bit_index = 0
            for param, options in self.ARRAY_PARAM_OPTIONS.items():
                # First byte is the length
                if bit_index + 8 <= len(array_param_bits):
                    length_bits = array_param_bits[bit_index:bit_index+8]
                    bit_index += 8
                    
                    # Convert bits to length
                    length = 0
                    for i, bit in enumerate(length_bits):
                        if bit:
                            length |= (1 << i)
                    
                    # Get values
                    values = []
                    bit_size = self.array_param_bit_sizes[param]
                    for _ in range(length):
                        if bit_index + bit_size <= len(array_param_bits):
                            value_bits = array_param_bits[bit_index:bit_index+bit_size]
                            bit_index += bit_size
                            
                            # Convert bits to index
                            index = 0
                            for i, bit in enumerate(value_bits):
                                if bit:
                                    index |= (1 << i)
                            
                            # Get value
                            if index < len(options):
                                values.append(options[index])
                    
                    if values:
                        array_params[param] = values
            
            # Merge array parameters into model parameters
            model_params.update(array_params)
            
            # Add constant parameters
            model_params.update(self.CONSTANT_PARAMS)
            
            # Decode features
            included_features = set()
            for i, bit in enumerate(feature_bits):
                if bit:
                    included_features.add(i)
            
            # Decode training parameters
            training_params = {}
            bit_index = 0
            for param, options in self.TRAINING_PARAM_OPTIONS.items():
                bit_size = self.training_param_bit_sizes[param]
                
                if bit_index + bit_size <= len(training_param_bits):
                    # Extract bits for this parameter
                    param_bits = training_param_bits[bit_index:bit_index+bit_size]
                    bit_index += bit_size
                    
                    # Convert bits to index
                    index = 0
                    for i, bit in enumerate(param_bits):
                        if bit:
                            index |= (1 << i)
                    
                    # Get parameter value
                    if index < len(options):
                        training_params[param] = options[index]
            
            return {
                'model_params': model_params,
                'included_features': included_features,
                'training_params': training_params
            }
            
        except Exception as e:
            raise ValueError(f"Error decoding model identifier: {e}")
    
    @staticmethod
    def extract_features_from_data(feature_data, include_all=False):
        """
        Extract feature indices from feature data.
        
        Args:
            feature_data: Data containing features (DataFrame, numpy array, etc.)
            include_all: Whether to include all features or make a selection
            
        Returns:
            Tuple of (included_features, total_features)
        """
        # Determine the total number of features
        if hasattr(feature_data, 'shape'):
            if len(feature_data.shape) >= 2:
                total_features = feature_data.shape[1]
            else:
                total_features = 1
        else:
            # Fallback for other data types
            total_features = len(feature_data)
        
        # If including all features, create a set of all indices
        if include_all:
            included_features = set(range(total_features))
        else:
            # This is where you'd implement feature selection logic
            # For now, just include all features
            included_features = set(range(total_features))
        
        return included_features, total_features 