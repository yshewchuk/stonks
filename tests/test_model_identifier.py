#!/usr/bin/env python
"""
Tests for the model identifier system.
"""

import sys
import os
import unittest
from pathlib import Path
from model.model_identifier.bit_array import BitArray

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_identifier import ModelIdentifier
from model.model_definition import ModelDefinition

class TestModelIdentifier(unittest.TestCase):
    def setUp(self):
        # Now we use feature indexes instead of feature names
        self.feature_indexes = {0, 1, 2, 3, 4, 5}  # Using the first 6 feature indexes for testing
        self.identifier = ModelIdentifier()  # No parameters needed for constructor
    
    def test_create_and_decode_identifier(self):
        """Test creating and decoding a model identifier."""
        # Test parameters
        model_params = {
            'cnn_layers': 2,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'l2_reg': 0.01,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu',
            'n_steps': 60  # Required parameter
        }
        
        training_params = {
            'batch_size': 32,
            'reduce_lr_patience': 10
        }
        
        selected_feature_indexes = {0, 2, 4}  # Using indexes instead of names
        
        # Create identifier
        model_id = self.identifier.create_model_identifier(
            model_parameters=model_params,
            training_parameters=training_params,
            selected_feature_indexes=selected_feature_indexes
        )
        
        # Decode identifier
        decoded = self.identifier.decode_model_identifier(model_id)
        
        # Verify decoded values
        self.assertEqual(decoded['version'], 1)
        self.assertEqual(decoded['model_parameters'], model_params)
        self.assertEqual(decoded['training_parameters'], training_params)
        self.assertEqual(decoded['feature_indexes'], selected_feature_indexes)
    
    def test_invalid_features(self):
        """Test that invalid feature indexes raise an error."""
        model_params = {
            'cnn_layers': 1,
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'l2_reg': 0.01,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu',
            'n_steps': 60
        }
        
        training_params = {
            'batch_size': 32,
            'reduce_lr_patience': 10
        }
        
        # Test with invalid feature index (greater than total feature count)
        invalid_feature_index = ModelDefinition.TOTAL_FEATURE_COUNT + 10
        with self.assertRaises(ValueError):
            self.identifier.create_model_identifier(
                model_parameters=model_params,
                training_parameters=training_params,
                selected_feature_indexes={invalid_feature_index}
            )
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise an error."""
        model_params = {
            'cnn_layers': 5,  # Invalid value
            'cnn_filters': 64,
            'cnn_kernel_size': 3,
            'cnn_pooling': True,
            'lstm_layers': 2,
            'lstm_units': 128,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'l2_reg': 0.01,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'activation': 'relu',
            'n_steps': 60
        }
        
        training_params = {
            'batch_size': 32,
            'reduce_lr_patience': 10
        }
        
        selected_feature_indexes = {0, 1, 2}
        
        # Test with invalid parameter
        with self.assertRaises(ValueError):
            self.identifier.create_model_identifier(
                model_parameters=model_params,
                training_parameters=training_params,
                selected_feature_indexes=selected_feature_indexes
            )
    
    def test_invalid_identifier(self):
        """Test that invalid identifiers raise an error."""
        with self.assertRaises(ValueError):
            self.identifier.decode_model_identifier("invalid_identifier")
    
    def test_version_handling(self):
        """Test that unsupported versions raise an error."""
        # Create a bit array with an unsupported version
        bits = [True] * 8  # Version 255
        bits.extend([False] * (self.identifier.total_size - 8))
        
        # Convert bits to BitArray
        bit_array = BitArray(bits)
        
        # Convert to base32
        invalid_id = bit_array.to_base32()
        
        with self.assertRaises(ValueError):
            self.identifier.decode_model_identifier(invalid_id)

if __name__ == '__main__':
    unittest.main() 