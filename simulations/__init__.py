"""
Simulations Package
===================

This package contains all simulation-related code for the CGC project:

Subpackages
-----------
cgc : Core CGC (Casimir-Gravity Crossover) module
    - Physics implementation
    - MCMC sampling
    - Data loading
    - Analysis tools

class_cgc : Modified CLASS Boltzmann solver
    - CGC modifications to cosmological perturbations

LaCE : Lyman-alpha Cosmology Emulator
    - Flux power spectrum emulation

stripping_models : Tidal stripping calibrations
    - IllustrisTNG, EAGLE, SIMBA calibrations

mcmc_config : MCMC configuration files
    - Cobaya YAML configurations

Usage
-----
>>> import sys
>>> sys.path.insert(0, '/path/to/MCMC_cgc')
>>> from simulations.cgc import CGCPhysics, run_mcmc
"""

__version__ = "1.0.0"
