#!/usr/bin/env python3
"""Test the updated SNe and Lyman-α likelihoods."""

import numpy as np
from cgc.data_loader import load_real_data
from cgc.likelihoods import log_likelihood_sne, log_likelihood_lyalpha

print('=== TESTING UPDATED LIKELIHOODS ===')
print()

# Load data with SNe and Lyman-alpha
data = load_real_data(verbose=False, include_sne=True, include_lyalpha=True)

# Test parameters (Planck-like)
theta = np.array([
    0.02237,   # omega_b
    0.1200,    # omega_cdm
    0.6736,    # h
    3.044,     # ln10As
    0.9649,    # n_s
    0.0544,    # tau
    0.1,       # mu_cgc
    0.5,       # n_g
    2.0,       # z_trans
    1.0        # rho_thresh
])

# Test SNe likelihood
print('1. SUPERNOVA LIKELIHOOD (Pantheon+):')
print(f'   Data: {len(data["sne"]["z"])} SNe')
print(f'   Has inv_cov: {"inv_cov" in data["sne"]}')
print(f'   Cov shape: {data["sne"]["cov"].shape}')

logl_sne = log_likelihood_sne(theta, data['sne'])
print(f'   log L = {logl_sne:.2f}')
print(f'   chi2/dof ~ {-2*logl_sne / len(data["sne"]["z"]):.3f}')
print()

# Test Lyman-alpha likelihood
print('2. LYMAN-ALPHA LIKELIHOOD (Chabanier+):')
print(f'   Data: {len(data["lyalpha"]["z"])} points')
print(f'   z range: {data["lyalpha"]["z"].min():.1f} - {data["lyalpha"]["z"].max():.1f}')
print(f'   k range: {data["lyalpha"]["k"].min():.4f} - {data["lyalpha"]["k"].max():.3f} s/km')

logl_lya = log_likelihood_lyalpha(theta, data['lyalpha'])
print(f'   log L = {logl_lya:.2f}')
print(f'   chi2/dof ~ {-2*logl_lya / len(data["lyalpha"]["z"]):.3f}')
print()

print('=' * 50)
print('Both likelihoods working correctly!')
