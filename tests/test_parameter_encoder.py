#!/usr/bin/env python
"""
Test the ParameterEncoder class for encoding/decoding parameter values.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_identifier.parameter_encoder import ParameterEncoder
from model.model_identifier.bit_array import BitArray

class TestParameterEncoder(unittest.TestCase):
    """Test cases for the ParameterEncoder class."""
    
    def test_init(self):
        """Test initialization."""
        # Test with valid options
        encoder = ParameterEncoder([1, 2, 3, 4])
        self.assertEqual(encoder.bit_size, 2)  # Need 2 bits for 4 values
        
        encoder = ParameterEncoder([True, False])
        self.assertEqual(encoder.bit_size, 1)  # Need 1 bit for 2 values
        
        encoder = ParameterEncoder(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        self.assertEqual(encoder.bit_size, 3)  # Need 3 bits for 8 values
        
        # Test with empty options
        with self.assertRaises(ValueError):
            ParameterEncoder([])
    
    def test_encode_numeric(self):
        """Test encoding numeric values."""
        encoder = ParameterEncoder([32, 64, 96, 128])
        
        # Test each value
        self.assertEqual(encoder.encode(32).bits, [False, False])  # Index 0
        self.assertEqual(encoder.encode(64).bits, [True, False])   # Index 1
        self.assertEqual(encoder.encode(96).bits, [False, True])   # Index 2
        self.assertEqual(encoder.encode(128).bits, [True, True])   # Index 3
        
        # Test invalid value
        with self.assertRaises(ValueError):
            encoder.encode(42)
    
    def test_encode_boolean(self):
        """Test encoding boolean values."""
        encoder = ParameterEncoder([False, True])
        
        # Test each value
        self.assertEqual(encoder.encode(False).bits, [False])  # Index 0
        self.assertEqual(encoder.encode(True).bits, [True])    # Index 1
    
    def test_encode_string(self):
        """Test encoding string values."""
        encoder = ParameterEncoder(['relu', 'tanh', 'selu'])
        
        # Test each value
        self.assertEqual(encoder.encode('relu').bits, [False, False])  # Index 0
        self.assertEqual(encoder.encode('tanh').bits, [True, False])   # Index 1
        self.assertEqual(encoder.encode('selu').bits, [False, True])   # Index 2
        
        # Test invalid value
        with self.assertRaises(ValueError):
            encoder.encode('invalid')
    
    def test_decode_numeric(self):
        """Test decoding numeric values."""
        encoder = ParameterEncoder([32, 64, 96, 128])
        
        # Test each bit pattern
        self.assertEqual(encoder.decode(BitArray([False, False])), 32)   # Index 0
        self.assertEqual(encoder.decode(BitArray([True, False])), 64)    # Index 1
        self.assertEqual(encoder.decode(BitArray([False, True])), 96)    # Index 2
        self.assertEqual(encoder.decode(BitArray([True, True])), 128)    # Index 3
        
        # Test wrong number of bits
        with self.assertRaises(ValueError):
            encoder.decode(BitArray([True]))
        
        # Test invalid index
        with self.assertRaises(ValueError):
            encoder.decode(BitArray([True, True, True]))
    
    def test_decode_boolean(self):
        """Test decoding boolean values."""
        encoder = ParameterEncoder([False, True])
        
        # Test each bit pattern
        self.assertEqual(encoder.decode(BitArray([False])), False)  # Index 0
        self.assertEqual(encoder.decode(BitArray([True])), True)    # Index 1
    
    def test_decode_string(self):
        """Test decoding string values."""
        encoder = ParameterEncoder(['relu', 'tanh', 'selu'])
        
        # Test each bit pattern
        self.assertEqual(encoder.decode(BitArray([False, False])), 'relu')  # Index 0
        self.assertEqual(encoder.decode(BitArray([True, False])), 'tanh')   # Index 1
        self.assertEqual(encoder.decode(BitArray([False, True])), 'selu')   # Index 2
    
    def test_round_trip(self):
        """Test round trip encoding and decoding."""
        # Test with numeric values
        encoder = ParameterEncoder([32, 64, 96, 128])
        for value in [32, 64, 96, 128]:
            bits = encoder.encode(value)
            decoded = encoder.decode(bits)
            self.assertEqual(value, decoded)
        
        # Test with boolean values
        encoder = ParameterEncoder([False, True])
        for value in [False, True]:
            bits = encoder.encode(value)
            decoded = encoder.decode(bits)
            self.assertEqual(value, decoded)
        
        # Test with string values
        encoder = ParameterEncoder(['relu', 'tanh', 'selu'])
        for value in ['relu', 'tanh', 'selu']:
            bits = encoder.encode(value)
            decoded = encoder.decode(bits)
            self.assertEqual(value, decoded)

if __name__ == '__main__':
    unittest.main() 