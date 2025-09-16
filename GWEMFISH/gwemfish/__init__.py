"""
GWEMFISH: Gravitational Wave and Electromagnetic Fisher Information Analysis

A comprehensive package for automated gravitational lensing Fisher matrix analysis
combining electromagnetic and gravitational wave observations.
"""

from .input.lens_input import LensInput
from .input.parameter_input import ParameterInput
from .models.lens_models import SupportedLensModels

__version__ = "0.1.0"
__author__ = "GWEMFISH Team"

__all__ = [
    "LensInput",
    "ParameterInput", 
    "SupportedLensModels"
]
