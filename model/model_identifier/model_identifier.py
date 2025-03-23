"""
Creates and decodes unique identifiers for model configurations.
"""

from typing import Dict, List, Any, Set, Optional
from .model_parameters import ModelParameters
from .bit_array import BitArray
from ..model_definition import ModelDefinition

class ModelIdentifier:
    """
    Creates and decodes unique identifiers for models based on their parameters and features.
    
    The identifier encodes:
    1. Version number (for future compatibility)
    2. Model parameters (e.g., number of units, dropout rate, layer configuration)
    3. Training parameters (e.g., batch size, learning rate)
    4. Selected feature indexes
    """
    
    # Current version of the identifier format
    VERSION = 1
    
    # Number of bits used for version
    VERSION_BITS = 8
    
    def __init__(self):
        """
        Initialize ModelIdentifier.
        """
        self.model_params = ModelParameters(ModelDefinition.MODEL_PARAMETER_OPTIONS)
        self.training_params = ModelParameters(ModelDefinition.TRAINING_PARAMETER_OPTIONS)
        
        # Calculate total bit size for validation
        self.param_size = sum(encoder.bit_size for encoder in self.model_params.encoders.values())
        self.training_param_size = sum(encoder.bit_size for encoder in self.training_params.encoders.values())
        self.feature_size = ModelDefinition.TOTAL_FEATURE_COUNT
        self.total_size = (
            self.VERSION_BITS + 
            self.param_size + 
            self.training_param_size + 
            self.feature_size
        )
    
    def create_model_identifier(
        self,
        model_parameters: Dict[str, Any],
        training_parameters: Dict[str, Any],
        selected_feature_indexes: Set[int]
    ) -> str:
        """
        Create a unique identifier for a model configuration.
        
        Args:
            model_parameters: Dictionary of model parameter names to their values
            training_parameters: Dictionary of training parameter names to their values
            selected_feature_indexes: Set of selected feature indexes
            
        Returns:
            Base32 encoded string uniquely identifying the model configuration
            
        Raises:
            ValueError: If parameters are invalid or feature indexes are out of range
        """
        # Validate parameters
        model_parameters = ModelDefinition.validate_model_parameters(model_parameters)
        training_parameters = ModelDefinition.validate_training_parameters(training_parameters)
        selected_feature_indexes = ModelDefinition.validate_feature_indexes(selected_feature_indexes)
        
        # Create bit array starting with version
        bits = BitArray()
        
        # Add version bits
        version_bits = BitArray([bool(self.VERSION & (1 << i)) for i in range(self.VERSION_BITS)])
        bits.extend(version_bits)
        
        # Add model parameter bits
        model_param_bits = self.model_params.encode(model_parameters)
        bits.extend(model_param_bits)
        
        # Add training parameter bits
        training_param_bits = self.training_params.encode(training_parameters)
        bits.extend(training_param_bits)
        
        # Add feature selection bits (1 bit per feature index)
        feature_bits = [False] * self.feature_size
        for idx in selected_feature_indexes:
            feature_bits[idx] = True
        bits.extend(BitArray(feature_bits))
        
        # Verify total size
        if len(bits) != self.total_size:
            raise ValueError(f"Internal error: Expected {self.total_size} bits, got {len(bits)}")
        
        # Convert to base32 string
        return bits.to_base32()
    
    def decode_model_identifier(self, identifier: str) -> Dict[str, Any]:
        """
        Decode a model identifier back into its components.
        
        Args:
            identifier: Base32 encoded model identifier string
            
        Returns:
            Dictionary containing:
                'version': Version number of the identifier
                'model_parameters': Dictionary of model parameter names to values
                'training_parameters': Dictionary of training parameter names to values
                'feature_indexes': Set of selected feature indexes
                
        Raises:
            ValueError: If identifier format is invalid
        """
        try:
            # Convert from base32
            bits = BitArray.from_base32(identifier)
            
            # Extract version
            if len(bits) < self.VERSION_BITS:
                raise ValueError("Identifier too short")
            version_bits = bits[:self.VERSION_BITS]
            version = sum((1 << i) for i, bit in enumerate(version_bits) if bit)
            
            if version != self.VERSION:
                raise ValueError(f"Unsupported version: {version}")
            
            # Extract model parameters
            model_param_start = self.VERSION_BITS
            model_param_bits = bits[model_param_start:model_param_start + self.param_size]
            model_parameters = self.model_params.decode(model_param_bits)
            
            # Extract training parameters
            training_param_start = model_param_start + self.param_size
            training_param_bits = bits[training_param_start:training_param_start + self.training_param_size]
            training_parameters = self.training_params.decode(training_param_bits)
            
            # Extract features
            feature_start = training_param_start + self.training_param_size
            feature_bits = bits[feature_start:]
            selected_feature_indexes = {
                idx for idx, selected in enumerate(feature_bits)
                if selected
            }
            
            return {
                'version': version,
                'model_parameters': model_parameters,
                'training_parameters': training_parameters,
                'feature_indexes': selected_feature_indexes
            }
            
        except Exception as e:
            raise ValueError(f"Invalid identifier format: {str(e)}")
    
    @staticmethod
    def _bits_to_int(bits: List[bool]) -> int:
        """Convert a list of bits to an integer."""
        return sum((1 << i) for i, bit in enumerate(bits) if bit) 