"""
Cosmological Simulations Access Module
======================================

This module provides access to real data from major cosmological simulations:
- IllustrisTNG (https://www.tng-project.org/)
- EAGLE (http://icc.dur.ac.uk/Eagle/)
- FIRE-2 (https://fire.northwestern.edu/)
- SIMBA (http://simba.roe.ac.uk/)
- Flathub FIRE (https://flathub.flatironinstitute.org/)
- BAHAMAS, Auriga, APOSTLE, Magneticum

Each simulation requires different access methods - see individual modules.
"""

from .illustris_tng import IllustrisTNGAccess
from .eagle_database import EAGLEDatabase
from .fire_data import FIREDataAccess
from .simba_data import SIMBADataAccess
from .flathub_fire import FlathubFIREAccess
from .additional_simulations import (
    BAHAMASAccess, 
    AurigaAccess, 
    APOSTLEAccess, 
    MagneticumAccess,
    get_all_additional_simulations
)
from .download_all import download_all_simulations, get_dwarf_stripping_data

__all__ = [
    # Main simulations
    'IllustrisTNGAccess',
    'EAGLEDatabase', 
    'FIREDataAccess',
    'SIMBADataAccess',
    'FlathubFIREAccess',
    # Additional simulations
    'BAHAMASAccess',
    'AurigaAccess',
    'APOSTLEAccess',
    'MagneticumAccess',
    'get_all_additional_simulations',
    # Download utilities
    'download_all_simulations',
    'get_dwarf_stripping_data'
]
