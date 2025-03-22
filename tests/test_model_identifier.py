#!/usr/bin/env python
"""
Test the ModelIdentifier class for creating deterministic model identifiers.
"""

import sys
import os
import numpy as np
from pprint import pprint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_identifier import ModelIdentifier
from model.model_builder import ModelBuilder

def test_model_identifier():
    """Test the ModelIdentifier class."""
    print("Testing ModelIdentifier...")
    
    # Create a ModelIdentifier
    identifier = ModelIdentifier()
    
    # Get default model parameters
    model_params = ModelBuilder.get_default_model_params()
    model_params['n_features_total'] = 330  # Add a required parameter
    
    # Create some sample feature data
    # Normally this would be the actual feature data, but we'll use a placeholder
    fake_feature_data = np.zeros((100, 330))  # 100 samples, 330 features
    
    # Extract features
    included_features, total_features = identifier.extract_features_from_data(
        fake_feature_data, include_all=True
    )
    
    print(f"Total features: {total_features}")
    print(f"Included features: {len(included_features)}")
    
    # Create a sample training params dict
    training_params = {
        'batch_size': 32,
    }
    
    # Generate an identifier
    model_id = identifier.create_model_identifier(
        model_params=model_params,
        included_features=included_features,
        total_features=total_features,
        training_params=training_params
    )
    
    print(f"Generated model ID: {model_id}")
    print(f"Model ID length: {len(model_id)}")
    
    # Test decoding
    decoded = identifier.decode_model_identifier(model_id)
    
    print("\nDecoded model parameters:")
    pprint(decoded['model_params'])
    
    print(f"\nDecoded included features count: {len(decoded['included_features'])}")
    
    print("\nDecoded training parameters:")
    pprint(decoded['training_params'])
    
    # Test with different parameters
    model_params2 = model_params.copy()
    model_params2['n_units'] = 192
    model_params2['dropout_rate'] = 0.5
    model_params2['lstm_layers'] = 3
    model_params2['cnn_filters'] = [32, 64]
    model_params2['cnn_kernel_sizes'] = [3, 5]
    
    model_id2 = identifier.create_model_identifier(
        model_params=model_params2,
        included_features=included_features,
        total_features=total_features,
        training_params=training_params
    )
    
    print(f"\nGenerated model ID with different parameters: {model_id2}")
    
    # Check if they're different
    print(f"Are IDs different? {model_id != model_id2}")
    
    # Test with a subset of features
    subset_features = set(range(0, 330, 2))  # Every other feature
    
    model_id3 = identifier.create_model_identifier(
        model_params=model_params,
        included_features=subset_features,
        total_features=total_features,
        training_params=training_params
    )
    
    print(f"\nGenerated model ID with subset of features: {model_id3}")
    print(f"Are feature-different IDs different? {model_id != model_id3}")
    
    # Test with different training parameters
    training_params2 = {
        'batch_size': 64,
    }
    
    model_id4 = identifier.create_model_identifier(
        model_params=model_params,
        included_features=included_features,
        total_features=total_features,
        training_params=training_params2
    )
    
    print(f"\nGenerated model ID with different training parameters: {model_id4}")
    print(f"Are training-different IDs different? {model_id != model_id4}")
    
    print("\nTest completed successfully!")
    
if __name__ == "__main__":
    test_model_identifier() 