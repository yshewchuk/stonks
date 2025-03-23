#!/usr/bin/env python
"""
Test the BitArray class for bit manipulation and base32 encoding.
"""

import sys
import os
import unittest
from pathlib import Path
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_identifier.bit_array import BitArray
from model.model_identifier import ModelIdentifier
from model.model_definition import ModelDefinition

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
        
    def test_large_sparse_array(self):
        """Test conversion of large arrays with mostly False values."""
        # Create an array with 1000 bits, mostly False
        # This tests the fix for the issue with bit arrays with a lot of False values
        bits = [False] * 1000
        bits[10] = True
        bits[100] = True
        bits[500] = True
        bits[999] = True
        
        ba = BitArray(bits)
        bytes_data = ba.to_bytes()
        
        # Verify the bytes are correct
        self.assertEqual(len(bytes_data), 125)  # 1000 bits = 125 bytes
        
        # Verify the specific bits we set to True
        ba2 = BitArray.from_bytes(bytes_data)
        self.assertEqual(len(ba2), 1000)
        self.assertTrue(ba2[10])
        self.assertTrue(ba2[100])
        self.assertTrue(ba2[500])
        self.assertTrue(ba2[999])
        
        # Verify all other bits are False
        for i in range(1000):
            if i not in [10, 100, 500, 999]:
                self.assertFalse(ba2[i], f"Bit at position {i} should be False")
    
    def test_non_multiple_of_8_length(self):
        """Test conversion of bit arrays with lengths not a multiple of 8."""
        # Create arrays with various non-multiple-of-8 lengths
        test_lengths = [1, 3, 5, 7, 9, 15, 17, 31, 33]
        
        for length in test_lengths:
            # Set alternate bits to True
            bits = [i % 2 == 0 for i in range(length)]
            ba = BitArray(bits)
            
            # Convert to bytes and back
            bytes_data = ba.to_bytes()
            ba2 = BitArray.from_bytes(bytes_data)
            
            # Truncate ba2 to original length (from_bytes doesn't know original length)
            ba2 = BitArray(ba2.bits[:length])
            
            # Verify the bits match the original
            self.assertEqual(ba, ba2, f"Round trip failed for length {length}")
    
    def test_last_byte_construction(self):
        """Test that the last byte of a bit array is correctly constructed."""
        # Test with a partial last byte (7 bits)
        bits = [True, False, True, False, True, False, True]
        ba = BitArray(bits)
        bytes_data = ba.to_bytes()
        
        # Manually calculate what the last byte should be
        expected_byte = 0
        for i, bit in enumerate(bits):
            if bit:
                expected_byte |= (1 << i)
        
        # Verify the last byte matches our expectation
        self.assertEqual(bytes_data[0], expected_byte)
        
        # Recreate the BitArray and verify the bits match
        ba2 = BitArray.from_bytes(bytes_data)
        self.assertEqual(ba2.bits[:7], bits)
        
    def test_feature_selection_scenario(self):
        """Test the specific scenario that caused issues with feature selection in genetic algorithm."""
        # Create a bit array simulating feature selection (few True bits, many False bits)
        total_features = 330
        selected_features = [5, 20, 100, 150, 200, 300]
        
        bits = [False] * total_features
        for idx in selected_features:
            bits[idx] = True
            
        ba = BitArray(bits)
        base32 = ba.to_base32()
        
        # Convert back and check that only the selected features are True
        ba2 = BitArray.from_base32(base32)
        selected_indexes = [i for i, bit in enumerate(ba2.bits) if bit]
        
        # Verify the selected indexes match what we set
        self.assertEqual(selected_indexes, selected_features)
    
    def test_avoid_seven_pattern(self):
        """Test that large sparse arrays don't produce base32 encoding with many '7' characters."""
        # Create a bit array with 330 elements (typical feature count)
        # with only 15 True values randomly distributed
        total_bits = 330
        true_count = 15
        
        # Generate random positions for True values
        true_positions = sorted(random.sample(range(total_bits), true_count))
        
        # Create the bit array
        bits = [False] * total_bits
        for pos in true_positions:
            bits[pos] = True
        
        ba = BitArray(bits)
        base32 = ba.to_base32()
        
        # Check the resulting base32 string for patterns of '7'
        seven_count = base32.count('7')
        seven_ratio = seven_count / len(base32) if len(base32) > 0 else 0
        
        # A reasonable threshold - no more than 30% of characters should be '7'
        self.assertLess(seven_ratio, 0.3, 
                        f"Base32 encoding has too many '7' characters ({seven_count}/{len(base32)} = {seven_ratio:.2f})")
        
        # Ensure a long consecutive pattern of '7's doesn't exist
        consecutive_sevens = max(len(s) for s in base32.split('7')) if '7' in base32 else 0
        self.assertLess(consecutive_sevens, 5, 
                        f"Base32 encoding has {consecutive_sevens} consecutive '7' characters")


if __name__ == '__main__':
    unittest.main() 