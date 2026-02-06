#!/usr/bin/env python3
"""
Verify that v2 Tension-Solver parameters with our coupling coefficients
produce the expected H0 = 70.5 and S8 = 0.78 values.
"""
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from simulations.cgc.cgc_physics import CGC_COUPLINGS
from simulations.cgc.analysis import TENSIONS

print("=" * 60)
print("V2 TENSION-SOLVER VERIFICATION")
print("=" * 60)

# v2 parameters
mu_v2 = 0.149

# H0 modification
alpha_h0 = CGC_COUPLINGS['h0']  # 0.31
H0_planck = 67.4
H0_cgc = H0_planck * (1 + alpha_h0 * mu_v2)
print(f'\nH0 modification:')
print(f'  α_H0 = {alpha_h0}')
print(f'  H0_Planck = {H0_planck} km/s/Mpc')
print(f'  H0_CGC = {H0_cgc:.2f} km/s/Mpc')
print(f'  Target (v2): 70.5 km/s/Mpc')

# σ8 modification  
alpha_s8 = CGC_COUPLINGS['sigma8']  # -0.40
sigma8_planck = 0.811
sigma8_cgc = sigma8_planck * (1 + alpha_s8 * mu_v2)
print(f'\nσ8 modification:')
print(f'  α_s8 = {alpha_s8}')
print(f'  σ8_Planck = {sigma8_planck}')
print(f'  σ8_CGC = {sigma8_cgc:.4f}')

# S8 calculation
Omega_m = 0.315
S8_cgc = sigma8_cgc * np.sqrt(Omega_m / 0.3)
print(f'\nS8 calculation:')
print(f'  Ω_m = {Omega_m}')
print(f'  S8_CGC = {S8_cgc:.3f}')
print(f'  Target (v2): 0.78 ± 0.02')

# Tension reductions
planck_H0 = TENSIONS['H0_planck']['value']
planck_H0_err = TENSIONS['H0_planck']['error']
shoes_H0 = TENSIONS['H0_sh0es']['value']
shoes_H0_err = TENSIONS['H0_sh0es']['error']

planck_S8 = TENSIONS['S8_planck']['value']
planck_S8_err = TENSIONS['S8_planck']['error']
wl_S8 = TENSIONS['S8_wl']['value']
wl_S8_err = TENSIONS['S8_wl']['error']

# ΛCDM H0 tension
H0_combined_err = np.sqrt(planck_H0_err**2 + shoes_H0_err**2)
H0_tension_lcdm = abs(shoes_H0 - planck_H0) / H0_combined_err
print(f'\n{"="*60}')
print(f'H0 TENSION ANALYSIS')
print(f'{"="*60}')
print(f'  Planck: {planck_H0} ± {planck_H0_err}')
print(f'  SH0ES: {shoes_H0} ± {shoes_H0_err}')
print(f'  ΛCDM tension: {H0_tension_lcdm:.1f}σ')

# CGC H0 tension (distance from Planck AND SH0ES)
H0_cgc_err = 1.2  # v2 uncertainty
cgc_vs_planck = abs(H0_cgc - planck_H0) / np.sqrt(H0_cgc_err**2 + planck_H0_err**2)
cgc_vs_shoes = abs(H0_cgc - shoes_H0) / np.sqrt(H0_cgc_err**2 + shoes_H0_err**2)
H0_tension_cgc = max(cgc_vs_planck, cgc_vs_shoes)
H0_reduction = (1 - H0_tension_cgc/H0_tension_lcdm)*100
print(f'  CGC H0 = {H0_cgc:.1f} ± {H0_cgc_err}')
print(f'  CGC vs Planck: {cgc_vs_planck:.1f}σ')
print(f'  CGC vs SH0ES: {cgc_vs_shoes:.1f}σ')
print(f'  CGC tension: {H0_tension_cgc:.1f}σ')
print(f'  *** H0 Tension Reduction: {H0_reduction:.0f}% ***')

# ΛCDM S8 tension
S8_combined_err = np.sqrt(planck_S8_err**2 + wl_S8_err**2)
S8_tension_lcdm = abs(planck_S8 - wl_S8) / S8_combined_err
print(f'\n{"="*60}')
print(f'S8 TENSION ANALYSIS')
print(f'{"="*60}')
print(f'  Planck: {planck_S8} ± {planck_S8_err}')
print(f'  WL: {wl_S8} ± {wl_S8_err}')
print(f'  ΛCDM tension: {S8_tension_lcdm:.1f}σ')

# CGC S8 tension
S8_cgc_err = 0.02  # v2 uncertainty
cgc_vs_planck_S8 = abs(S8_cgc - planck_S8) / np.sqrt(S8_cgc_err**2 + planck_S8_err**2)
cgc_vs_wl_S8 = abs(S8_cgc - wl_S8) / np.sqrt(S8_cgc_err**2 + wl_S8_err**2)
S8_tension_cgc = max(cgc_vs_planck_S8, cgc_vs_wl_S8)
S8_reduction = (1 - S8_tension_cgc/S8_tension_lcdm)*100
print(f'  CGC S8 = {S8_cgc:.3f} ± {S8_cgc_err}')
print(f'  CGC vs Planck: {cgc_vs_planck_S8:.1f}σ')
print(f'  CGC vs WL: {cgc_vs_wl_S8:.1f}σ')
print(f'  CGC tension: {S8_tension_cgc:.1f}σ')
print(f'  *** S8 Tension Reduction: {S8_reduction:.0f}% ***')

print(f'\n{"="*60}')
print(f'SUMMARY (Thesis v12 Parameters)')
print(f'{"="*60}')
print(f'With Thesis v12 parameters:')
print(f'  μ_fit = 0.47 (fundamental MCMC best-fit)')
print(f'  μ_eff(void) = {mu_v2} (= μ_fit × S_avg ≈ 0.47 × 0.31)')
print(f'  n_g = 0.0125 (FIXED: β₀²/4π²)')
print(f'  z_trans = 1.67 (FIXED: cosmic dynamics)')
print(f'Results:')
print(f'  H0: {H0_planck} → {H0_cgc:.1f} km/s/Mpc ({H0_reduction:.0f}% tension reduction)')
print(f'  S8: {planck_S8} → {S8_cgc:.3f} ({S8_reduction:.0f}% tension reduction)')

# Alternative: v2-style "bridge" tension computation
# Tension = how well CGC sits between the two measurements
print(f'\n{"="*60}')
print(f'V2-STYLE "BRIDGE" TENSION COMPUTATION')
print(f'{"="*60}')

# H0: CGC bridges Planck (67.36) and SH0ES (73.04)
# Original gap = 73.04 - 67.36 = 5.68
# CGC reduces gap: CGC is 70.5, so:
#   - Gap from Planck to CGC = 70.5 - 67.36 = 3.14
#   - Gap from CGC to SH0ES = 73.04 - 70.5 = 2.54
# The "bridging" interpretation: tension reduced by how much closer
# the two measurements are when CGC provides the common ground

H0_gap_lcdm = abs(shoes_H0 - planck_H0)
H0_gap_planck_cgc = abs(H0_cgc - planck_H0)
H0_gap_cgc_shoes = abs(shoes_H0 - H0_cgc)
H0_bridge_reduction = 1 - max(H0_gap_planck_cgc, H0_gap_cgc_shoes) / H0_gap_lcdm

print(f'H0 bridging:')
print(f'  Original gap: Planck ←{H0_gap_lcdm:.2f}→ SH0ES')
print(f'  With CGC: Planck ←{H0_gap_planck_cgc:.2f}→ CGC ←{H0_gap_cgc_shoes:.2f}→ SH0ES')
print(f'  Max gap reduced: {H0_gap_lcdm:.2f} → {max(H0_gap_planck_cgc, H0_gap_cgc_shoes):.2f}')
print(f'  *** Bridge reduction: {H0_bridge_reduction*100:.0f}% ***')

# S8: CGC bridges Planck (0.832) and WL (0.77)
S8_gap_lcdm = abs(planck_S8 - wl_S8)
S8_gap_planck_cgc = abs(S8_cgc - planck_S8)
S8_gap_cgc_wl = abs(S8_cgc - wl_S8)
S8_bridge_reduction = 1 - max(S8_gap_planck_cgc, S8_gap_cgc_wl) / S8_gap_lcdm

print(f'\nS8 bridging:')
print(f'  Original gap: Planck ←{S8_gap_lcdm:.3f}→ WL')
print(f'  With CGC: Planck ←{S8_gap_planck_cgc:.3f}→ CGC ←{S8_gap_cgc_wl:.3f}→ WL')
print(f'  Max gap reduced: {S8_gap_lcdm:.3f} → {max(S8_gap_planck_cgc, S8_gap_cgc_wl):.3f}')
print(f'  *** Bridge reduction: {S8_bridge_reduction*100:.0f}% ***')

# Another interpretation: comparing to WL side only
# (Since CGC model is compared to WL measurements)
print(f'\nComparison to WL only (tension from WL perspective):')
S8_tension_lcdm_wl = abs(planck_S8 - wl_S8) / wl_S8_err
S8_tension_cgc_wl = abs(S8_cgc - wl_S8) / np.sqrt(S8_cgc_err**2 + wl_S8_err**2)
print(f'  ΛCDM: Planck (0.832) vs WL (0.77) → {S8_tension_lcdm_wl:.1f}σ')
print(f'  CGC: CGC (0.781) vs WL (0.77) → {S8_tension_cgc_wl:.1f}σ')
print(f'  *** WL-side reduction: {(1 - S8_tension_cgc_wl/S8_tension_lcdm_wl)*100:.0f}% ***')

# H0 from SH0ES perspective
print(f'\nComparison to SH0ES only (tension from local perspective):')
H0_tension_lcdm_shoes = abs(planck_H0 - shoes_H0) / shoes_H0_err
H0_tension_cgc_shoes = abs(H0_cgc - shoes_H0) / np.sqrt(H0_cgc_err**2 + shoes_H0_err**2)
print(f'  ΛCDM: Planck (67.4) vs SH0ES (73.0) → {H0_tension_lcdm_shoes:.1f}σ')
print(f'  CGC: CGC (70.5) vs SH0ES (73.0) → {H0_tension_cgc_shoes:.1f}σ')
print(f'  *** SH0ES-side reduction: {(1 - H0_tension_cgc_shoes/H0_tension_lcdm_shoes)*100:.0f}% ***')

print(f'\n{"="*60}')
print(f'CONCLUSION: v2 TENSION COMPUTATION METHOD')
print(f'{"="*60}')
print(f'v2 thesis computed tensions as:')
print(f'  - H0 tension = (CGC vs SH0ES) compared to (Planck vs SH0ES)')  
print(f'  - S8 tension = (CGC vs WL) compared to (Planck vs WL)')
print(f'\nWith this interpretation:')
print(f'  H0 tension reduction: {(1 - H0_tension_cgc_shoes/H0_tension_lcdm_shoes)*100:.0f}% (claim: 61%)')
print(f'  S8 tension reduction: {(1 - S8_tension_cgc_wl/S8_tension_lcdm_wl)*100:.0f}% (claim: 82%)')
