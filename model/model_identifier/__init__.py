"""
Model identifier module for creating and decoding unique model identifiers.

This module provides classes for:
- Bit array manipulation (BitArray)
- Parameter encoding (ParameterEncoder)
- Model parameter management (ModelParameters)
- Model identification (ModelIdentifier)
"""

from .bit_array import BitArray
from .parameter_encoder import ParameterEncoder
from .model_parameters import ModelParameters
from .model_identifier import ModelIdentifier

__all__ = ['BitArray', 'ParameterEncoder', 'ModelParameters', 'ModelIdentifier'] 