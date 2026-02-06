#!/usr/bin/env python3
"""
Verify CGC formula consistency and physics basis.
"""
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from simulations.cgc.cgc_physics import CGCPhysics, apply_cgc_to_bao, apply_cgc_to_growth, CGC_COUPLINGS

print('=' * 70)
print('FORMULA CONSISTENCY CHECK (Thesis v12 Parameters)')
print('=' * 70)

# Test CGC parameters - Thesis v12: μ_fit = 0.47, n_g = 0.0125, z_trans = 1.67
cgc = CGCPhysics(mu=0.47, z_trans=1.67, rho_thresh=200.0)
print(f'\nCGC Parameters (Thesis v12):')
print(f'  μ_fit = {cgc.mu} (fundamental MCMC best-fit)')
print(f'  n_g = {cgc.n_g} (FIXED: β₀²/4π²)')
print(f'  z_trans = {cgc.z_trans}')

# Test BAO formula at different redshifts
print(f'\nBAO Formula: (D_V/r_d)_CGC = (D_V/r_d)_LCDM x [1 + mu x (1+z)^(-n_g)]')
all_match = True
for z in [0.5, 1.0, 1.5, 2.0, 3.0]:
    factor = 1 + cgc.mu * (1 + z)**(-cgc.n_g)
    bao_ratio = apply_cgc_to_bao(1.0, z, cgc)
    match = np.isclose(factor, bao_ratio)
    all_match = all_match and match
    print(f'  z={z}: factor = {factor:.4f}, code = {bao_ratio:.4f}, match = {match}')
print(f'  All BAO tests pass: {all_match}')

# Test Growth formula at different redshifts
print(f'\nGrowth Formula: fs8_CGC = fs8_LCDM x [1 + alpha x mu x (1+z)^(-n_g)]')
print(f'  alpha_growth = {CGC_COUPLINGS["growth"]}')
all_match = True
for z in [0.5, 1.0, 1.5, 2.0, 3.0]:
    alpha = CGC_COUPLINGS['growth']
    factor = 1 + alpha * cgc.mu * (1 + z)**(-cgc.n_g)
    growth_ratio = apply_cgc_to_growth(1.0, z, cgc)
    match = np.isclose(factor, growth_ratio)
    all_match = all_match and match
    print(f'  z={z}: factor = {factor:.4f}, code = {growth_ratio:.4f}, match = {match}')
print(f'  All Growth tests pass: {all_match}')

# Check redshift variation
print(f'\nRedshift Variation of (1+z)^(-n_g) with n_g={cgc.n_g}:')
for z in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    factor = (1 + z)**(-cgc.n_g)
    print(f'  z={z}: (1+z)^(-{cgc.n_g}) = {factor:.4f}')

print(f'\n' + '=' * 70)
print('CGC COUPLINGS')
print('=' * 70)
for probe, alpha in CGC_COUPLINGS.items():
    print(f'  {probe}: {alpha}')

# Check H0 and sigma8 modifications
print(f'\n' + '=' * 70)
print('DERIVED PARAMETER MODIFICATIONS')
print('=' * 70)

# H0 modification
alpha_h0 = CGC_COUPLINGS['h0']
H0_planck = 67.4
H0_cgc = H0_planck * (1 + alpha_h0 * cgc.mu)
print(f'\nH0 modification:')
print(f'  Formula: H0_CGC = H0_Planck x (1 + alpha_h0 x mu)')
print(f'  alpha_h0 = {alpha_h0} (calibrated from Planck->SH0ES gap)')
print(f'  H0_Planck = {H0_planck} km/s/Mpc')
print(f'  H0_CGC = {H0_planck} x (1 + {alpha_h0} x {cgc.mu}) = {H0_cgc:.2f} km/s/Mpc')
print(f'  Target: ~70.5 km/s/Mpc, Match: {abs(H0_cgc - 70.5) < 0.5}')

# sigma8 modification
alpha_s8 = CGC_COUPLINGS['sigma8']
sigma8_planck = 0.811
sigma8_cgc = sigma8_planck * (1 + alpha_s8 * cgc.mu)
print(f'\nsigma8 modification:')
print(f'  Formula: sigma8_CGC = sigma8_Planck x (1 + alpha_s8 x mu)')
print(f'  alpha_s8 = {alpha_s8} (negative = enhanced gravity reduces CMB sigma8)')
print(f'  Physics: Faster structure growth -> same clustering today with lower initial sigma8')
print(f'  sigma8_Planck = {sigma8_planck}')
print(f'  sigma8_CGC = {sigma8_planck} x (1 + {alpha_s8} x {cgc.mu}) = {sigma8_cgc:.4f}')

# S8 calculation
Omega_m = 0.315
S8_cgc = sigma8_cgc * np.sqrt(Omega_m / 0.3)
print(f'\nS8 calculation:')
print(f'  S8 = sigma8 x sqrt(Omega_m / 0.3)')
print(f'  S8_CGC = {sigma8_cgc:.4f} x sqrt({Omega_m}/0.3) = {S8_cgc:.4f}')
print(f'  Target: ~0.78, Match: {abs(S8_cgc - 0.78) < 0.02}')

print(f'\n' + '=' * 70)
print('PHYSICS CONSISTENCY SUMMARY')
print('=' * 70)
print('\nFormula Structure:')
print('  BAO:    [1 + mu x (1+z)^(-n_g)]       - Distance ladder modification')
print('  Growth: [1 + 0.1 x mu x (1+z)^(-n_g)] - RSD enhancement')
print('  H0:     [1 + 0.31 x mu]               - Local H0 enhancement')
print('  sigma8: [1 - 0.40 x mu]               - CMB sigma8 reduction')

print('\nPhysical Interpretation:')
print('  - Enhanced gravity (G_eff > G_N) at cosmological scales')
print('  - (1+z)^(-n_g) captures redshift evolution: effect stronger at low z')
print('  - H0 enhancement: Local expansion rate increased by enhanced G')
print('  - sigma8 reduction: Same present clustering with faster growth')
print('    requires lower initial amplitude from CMB')
