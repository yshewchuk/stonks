"""
Handles encoding and decoding of individual parameter values.
"""

from typing import List, Any
import math
from .bit_array import BitArray

class ParameterEncoder:
    """
    Encodes and decodes parameter values using a fixed set of options.
    
    Each parameter value is encoded using the minimum number of bits needed
    to represent all possible values in the options list.
    """
    
    def __init__(self, options: List[Any]):
        """
        Initialize encoder with a list of possible values.
        
        Args:
            options: List of possible values for this parameter
            
        Raises:
            ValueError: If options list is empty
        """
        if not options:
            raise ValueError("Options list cannot be empty")
        
        self.options = list(options)  # Make a copy to prevent modification
        self.bit_size = max(1, math.ceil(math.log2(len(options))))
    
    def encode(self, value: Any) -> BitArray:
        """
        Encode a parameter value into bits.
        
        Args:
            value: Value to encode
            
        Returns:
            BitArray containing the encoded value
            
        Raises:
            ValueError: If value is not in options list
        """
        try:
            index = self.options.index(value)
        except ValueError:
            raise ValueError(f"Value {value} not in options list: {self.options}")
        
        # Convert index to bits
        return BitArray([bool(index & (1 << i)) for i in range(self.bit_size)])
    
    def decode(self, bits: BitArray) -> Any:
        """
        Decode bits back into a parameter value.
        
        Args:
            bits: BitArray containing the encoded value
            
        Returns:
            Decoded parameter value
            
        Raises:
            ValueError: If bits have wrong size or represent invalid index
        """
        if len(bits) != self.bit_size:
            raise ValueError(f"Expected {self.bit_size} bits, got {len(bits)}")
        
        # Convert bits to index
        index = sum((1 << i) for i, bit in enumerate(bits) if bit)
        
        if index >= len(self.options):
            raise ValueError(f"Invalid index {index} for options list of size {len(self.options)}")
        
        return self.options[index] 