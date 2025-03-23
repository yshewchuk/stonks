"""
Manages multiple parameter encoders for model configuration.
"""

from typing import Dict, List, Any
from .parameter_encoder import ParameterEncoder
from .bit_array import BitArray

class ModelParameters:
    """
    Manages encoding and decoding of multiple model parameters.
    
    Each parameter has its own encoder that handles converting values
    to and from bits based on the allowed options for that parameter.
    """
    
    def __init__(self, parameter_options: Dict[str, List[Any]]):
        """
        Initialize with parameter options.
        
        Args:
            parameter_options: Dictionary mapping parameter names to their possible values
            
        Raises:
            ValueError: If any parameter has an empty options list
        """
        # Create encoders for each parameter
        self.encoders = {
            name: ParameterEncoder(options)
            for name, options in sorted(parameter_options.items())
        }
    
    def encode(self, parameters: Dict[str, Any]) -> BitArray:
        """
        Encode multiple parameters into a single bit array.
        
        Args:
            parameters: Dictionary of parameter names to values
            
        Returns:
            BitArray containing all encoded parameters
            
        Raises:
            ValueError: If any parameter is missing or invalid
        """
        # Check for missing parameters
        missing = set(self.encoders.keys()) - set(parameters.keys())
        if missing:
            raise ValueError(f"Missing parameters: {', '.join(missing)}")
        
        # Encode each parameter in sorted order
        bits = BitArray()
        for name, encoder in self.encoders.items():
            param_bits = encoder.encode(parameters[name])
            bits.extend(param_bits)
        
        return bits
    
    def decode(self, bits: BitArray) -> Dict[str, Any]:
        """
        Decode a bit array back into parameter values.
        
        Args:
            bits: BitArray containing encoded parameters
            
        Returns:
            Dictionary mapping parameter names to their values
            
        Raises:
            ValueError: If bit array size doesn't match expected size
        """
        # Calculate total expected size
        expected_size = sum(encoder.bit_size for encoder in self.encoders.values())
        if len(bits) != expected_size:
            raise ValueError(f"Expected {expected_size} bits, got {len(bits)}")
        
        # Decode each parameter
        parameters = {}
        pos = 0
        for name, encoder in self.encoders.items():
            param_bits = bits[pos:pos + encoder.bit_size]
            parameters[name] = encoder.decode(param_bits)
            pos += encoder.bit_size
        
        return parameters 