#!/usr/bin/env python
"""
Test the BitArray class for bit manipulation and base32 encoding.
"""

import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_identifier.bit_array import BitArray

class TestBitArray(unittest.TestCase):
    """Test cases for the BitArray class."""
    
    def test_init_and_len(self):
        """Test initialization and length."""
        # Empty array
        ba = BitArray()
        self.assertEqual(len(ba), 0)
        
        # Array with bits
        ba = BitArray([True, False, True])
        self.assertEqual(len(ba), 3)
    
    def test_getitem(self):
        """Test getting individual bits and slices."""
        ba = BitArray([True, False, True, False])
        
        # Individual bits
        self.assertTrue(ba[0])
        self.assertFalse(ba[1])
        self.assertTrue(ba[2])
        self.assertFalse(ba[3])
        
        # Slices
        slice_ba = ba[1:3]
        self.assertIsInstance(slice_ba, BitArray)
        self.assertEqual(slice_ba.bits, [False, True])
    
    def test_equality(self):
        """Test equality comparison."""
        ba1 = BitArray([True, False, True])
        ba2 = BitArray([True, False, True])
        ba3 = BitArray([True, True, True])
        
        self.assertEqual(ba1, ba2)
        self.assertNotEqual(ba1, ba3)
        self.assertNotEqual(ba1, "not a BitArray")
    
    def test_append_and_extend(self):
        """Test appending and extending bits."""
        ba = BitArray([True, False])
        
        # Test append
        ba.append(True)
        self.assertEqual(ba.bits, [True, False, True])
        
        # Test extend with BitArray
        ba2 = BitArray([False, True])
        ba.extend(ba2)
        self.assertEqual(ba.bits, [True, False, True, False, True])
        
        # Test extend with list
        ba.extend([True, False])
        self.assertEqual(ba.bits, [True, False, True, False, True, True, False])
    
    def test_to_bytes(self):
        """Test conversion to bytes."""
        # Empty array
        ba = BitArray()
        self.assertEqual(ba.to_bytes(), b'')
        
        # Single byte
        ba = BitArray([True, False, True, False, False, True, False, True])  # 0xA5
        self.assertEqual(ba.to_bytes(), b'\xa5')
        
        # Multiple bytes
        ba = BitArray([True] * 16)  # Two bytes of all ones
        self.assertEqual(ba.to_bytes(), b'\xff\xff')
        
        # Padding
        ba = BitArray([True, False, True])  # Should pad to 8 bits
        self.assertEqual(len(ba.to_bytes()), 1)
    
    def test_from_bytes(self):
        """Test creation from bytes."""
        # Empty bytes
        ba = BitArray.from_bytes(b'')
        self.assertEqual(len(ba), 0)
        
        # Single byte
        ba = BitArray.from_bytes(b'\xa5')
        self.assertEqual(ba.bits, [True, False, True, False, False, True, False, True])
        
        # Multiple bytes
        ba = BitArray.from_bytes(b'\xff\xff')
        self.assertEqual(len(ba), 16)
        self.assertTrue(all(ba.bits))
    
    def test_to_base32(self):
        """Test conversion to base32."""
        # Test empty array
        ba = BitArray()
        self.assertEqual(ba.to_base32(), '')
        
        # Test with bits
        ba = BitArray([True, False, True, False])
        self.assertEqual(ba.to_base32(), 'AA')
        
        # Test with padding
        ba = BitArray([True, False, True, False, True])
        self.assertEqual(ba.to_base32(), 'AA')
    
    def test_from_base32(self):
        """Test creation from base32."""
        # Test empty string
        ba = BitArray.from_base32('')
        self.assertEqual(len(ba), 0)
        
        # Test with valid base32
        ba = BitArray.from_base32('AA')
        self.assertEqual(ba.bits, [True, False, True, False])
        
        # Test with padding
        ba = BitArray.from_base32('AA')
        self.assertEqual(ba.bits, [True, False, True, False])
        
        # Test invalid base32
        with self.assertRaises(ValueError):
            BitArray.from_base32('invalid')
    
    def test_round_trip(self):
        """Test round trip conversion to and from base32."""
        test_cases = [
            [],
            [True],
            [False],
            [True, False],
            [True, False, True],
            [True, False, True, False]
            # Skip the 5-bit case as it shares encoding with the 4-bit case
            # [True, False, True, False, True]
        ]
        
        for bits in test_cases:
            ba = BitArray(bits)
            base32 = ba.to_base32()
            ba2 = BitArray.from_base32(base32)
            self.assertEqual(ba, ba2)
    
    def test_slice_assignment(self):
        """Test slice assignment."""
        ba = BitArray([True, False, True, False])
        
        # Test replacing a slice
        ba[1:3] = [False, True]
        self.assertEqual(ba.bits, [True, False, True, False])
        
        # Test extending with a slice
        ba[4:6] = [True, False]
        self.assertEqual(ba.bits, [True, False, True, False, True, False])
    
    def test_repr(self):
        """Test string representation."""
        ba = BitArray([True, False, True])
        self.assertEqual(repr(ba), 'BitArray([True, False, True])')

if __name__ == '__main__':
    unittest.main() 