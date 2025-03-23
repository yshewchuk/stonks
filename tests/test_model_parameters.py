#!/usr/bin/env python
"""
Test the ModelParameters class for managing multiple parameter encoders.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_identifier.model_parameters import ModelParameters
from model.model_identifier.bit_array import BitArray

class TestModelParameters(unittest.TestCase):
    """Test cases for the ModelParameters class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_options = {
            'cnn_layers': [0, 1, 2, 3],
            'cnn_filters': [32, 64, 96, 128],
            'cnn_kernel_size': [2, 3, 4, 5],
            'cnn_pooling': [False, True],
            'lstm_layers': [1, 2, 3, 4],
            'lstm_units': [32, 64, 96, 128],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            'recurrent_dropout_rate': [0.0, 0.1, 0.2, 0.3],
            'l2_reg': [0.0, 0.0001, 0.001, 0.01],
            'use_batch_norm': [False, True],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'activation': ['relu', 'tanh', 'selu', 'elu']
        }
        self.model_params = ModelParameters(self.parameter_options)
    
    def test_init(self):
        """Test initialization."""
        # Test with valid options
        params = ModelParameters(self.parameter_options)
        self.assertEqual(len(params.encoders), 12)
        
        # Verify encoder bit sizes
        self.assertEqual(params.encoders['cnn_layers'].bit_size, 2)      # 4 values
        self.assertEqual(params.encoders['cnn_filters'].bit_size, 2)     # 4 values
        self.assertEqual(params.encoders['cnn_kernel_size'].bit_size, 2) # 4 values
        self.assertEqual(params.encoders['cnn_pooling'].bit_size, 1)     # 2 values
        self.assertEqual(params.encoders['lstm_layers'].bit_size, 2)     # 4 values
        self.assertEqual(params.encoders['lstm_units'].bit_size, 2)      # 4 values
        self.assertEqual(params.encoders['dropout_rate'].bit_size, 2)    # 4 values
        self.assertEqual(params.encoders['recurrent_dropout_rate'].bit_size, 2) # 4 values
        self.assertEqual(params.encoders['l2_reg'].bit_size, 2)          # 4 values
        self.assertEqual(params.encoders['use_batch_norm'].bit_size, 1)  # 2 values
        self.assertEqual(params.encoders['learning_rate'].bit_size, 2)   # 4 values
        self.assertEqual(params.encoders['activation'].bit_size, 2)      # 4 values
        
        # Test with empty options
        with self.assertRaises(ValueError):
            ModelParameters({'param': []})
    
    def test_encode(self):
        """Test encoding parameters."""
        # Test valid parameters
        parameters = {
            'cnn_layers': 2,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'l2_reg': 0.001,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu'
        }
        bits = self.model_params.encode(parameters)
        
        # Total bits should be sum of individual encoder bit sizes
        expected_size = sum(encoder.bit_size for encoder in self.model_params.encoders.values())
        self.assertEqual(len(bits), expected_size)
        
        # Test missing parameter
        invalid_params = parameters.copy()
        del invalid_params['cnn_layers']
        with self.assertRaises(ValueError):
            self.model_params.encode(invalid_params)
        
        # Test invalid value
        invalid_params = parameters.copy()
        invalid_params['cnn_layers'] = 42
        with self.assertRaises(ValueError):
            self.model_params.encode(invalid_params)
    
    def test_decode(self):
        """Test decoding parameters."""
        # First encode some parameters
        parameters = {
            'cnn_layers': 2,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'l2_reg': 0.001,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu'
        }
        bits = self.model_params.encode(parameters)
        
        # Test decoding
        decoded = self.model_params.decode(bits)
        self.assertEqual(decoded, parameters)
        
        # Test wrong number of bits
        with self.assertRaises(ValueError):
            self.model_params.decode(BitArray([True]))
    
    def test_round_trip(self):
        """Test round trip encoding and decoding."""
        test_cases = [
            {
                'cnn_layers': 0,
                'cnn_filters': 32,
                'cnn_kernel_size': 2,
                'cnn_pooling': False,
                'lstm_layers': 1,
                'lstm_units': 32,
                'dropout_rate': 0.1,
                'recurrent_dropout_rate': 0.0,
                'l2_reg': 0.0,
                'use_batch_norm': False,
                'learning_rate': 0.0001,
                'activation': 'relu'
            },
            {
                'cnn_layers': 3,
                'cnn_filters': 128,
                'cnn_kernel_size': 5,
                'cnn_pooling': True,
                'lstm_layers': 4,
                'lstm_units': 128,
                'dropout_rate': 0.4,
                'recurrent_dropout_rate': 0.3,
                'l2_reg': 0.01,
                'use_batch_norm': True,
                'learning_rate': 0.1,
                'activation': 'elu'
            }
        ]
        
        for parameters in test_cases:
            bits = self.model_params.encode(parameters)
            decoded = self.model_params.decode(bits)
            self.assertEqual(parameters, decoded)
    
    def test_deterministic_ordering(self):
        """Test that parameter ordering is deterministic."""
        parameters1 = {
            'cnn_layers': 2,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'l2_reg': 0.001,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu'
        }
        
        # Create parameters in different order
        parameters2 = {
            'activation': 'relu',
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'l2_reg': 0.001,
            'recurrent_dropout_rate': 0.1,
            'dropout_rate': 0.2,
            'lstm_units': 128,
            'lstm_layers': 2,
            'cnn_pooling': True,
            'cnn_kernel_size': 3,
            'cnn_filters': 64,
            'cnn_layers': 2
        }
        
        bits1 = self.model_params.encode(parameters1)
        bits2 = self.model_params.encode(parameters2)
        
        # Both should produce identical bit patterns
        self.assertEqual(bits1.bits, bits2.bits)

if __name__ == '__main__':
    unittest.main() 