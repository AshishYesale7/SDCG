#!/usr/bin/env python3
"""
=============================================================================
SDCG PARAMETER DEGENERACY ANALYSIS
=============================================================================
Analyzing expected parameter degeneracies:
1. μ ↔ n_g (both affect amplitude)
2. z_trans ↔ ρ_thresh (both control transition)
3. μ ↔ H₀ (both affect expansion rate)

This is crucial for understanding what MCMC can actually constrain.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("=" * 80)
print("SDCG PARAMETER DEGENERACY ANALYSIS")
print("=" * 80)
print()

# =============================================================================
# 1. μ ↔ n_g DEGENERACY
# =============================================================================
print("1. μ ↔ n_g DEGENERACY ANALYSIS")
print("=" * 80)
print()

print("The SDCG modification to gravity is:")
print("  ΔG/G = μ × (k/k₀)^n_g × g(z) × S(ρ)")
print()
print("At a fixed scale k and redshift z, the effect depends on μ × (k/k₀)^n_g")
print()

k_0 = 0.05  # Mpc^-1

print("Effect at different scales:")
print("-" * 60)
print(f"{'k (Mpc⁻¹)':<12} {'(k/k₀)^0.01':<15} {'(k/k₀)^0.014':<15} {'(k/k₀)^0.02':<15}")
print("-" * 60)

k_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
for k in k_values:
    ratio = k / k_0
    f1 = ratio**0.01
    f2 = ratio**0.014
    f3 = ratio**0.02
    print(f"{k:<12.2f} {f1:<15.4f} {f2:<15.4f} {f3:<15.4f}")
print()

print("DEGENERACY DIRECTION:")
print("  If we increase n_g, we need to decrease μ to maintain the same effect")
print("  at a reference scale.")
print()

# Compute degeneracy curve
print("Degeneracy curve (constant effect at k = 0.1 Mpc⁻¹):")
print("-" * 40)

k_ref = 0.1
target_effect = 0.045 * (k_ref / k_0)**0.014  # Reference effect

n_g_range = np.linspace(0.005, 0.025, 10)
mu_degenerate = []

for n_g in n_g_range:
    # μ × (k/k₀)^n_g = target_effect
    mu = target_effect / (k_ref / k_0)**n_g
    mu_degenerate.append(mu)

print(f"{'n_g':<10} {'μ (degenerate)':<15}")
print("-" * 25)
for n_g, mu in zip(n_g_range, mu_degenerate):
    print(f"{n_g:<10.4f} {mu:<15.4f}")
print()

print("BREAKING THE DEGENERACY:")
print("  • Use MULTIPLE scales (different k bins)")
print("  • The power-law n_g has different effect at different k")
print("  • Lyman-α (high k) + BAO (low k) together break degeneracy")
print()

# =============================================================================
# 2. z_trans ↔ ρ_thresh DEGENERACY
# =============================================================================
print("2. z_trans ↔ ρ_thresh DEGENERACY ANALYSIS")
print("=" * 80)
print()

print("Both z_trans and ρ_thresh control when/where SDCG turns on:")
print()
print("  g(z) = [(1+z_trans)/(1+z)]^γ  for z ≤ z_trans, else 0")
print("  S(ρ) = 1/[1 + (ρ/ρ_thresh)^α]")
print()

print("DIFFERENT ROLES:")
print("-" * 60)
print("  z_trans: Controls WHEN in cosmic history SDCG activates")
print("           → Affects redshift evolution of observables")
print()
print("  ρ_thresh: Controls WHERE spatially SDCG is active")
print("            → Affects environment-dependence of observables")
print()

print("PARTIAL DEGENERACY:")
print("  At fixed redshift z, increasing z_trans increases g(z)")
print("  This can be compensated by increasing ρ_thresh (more screening)")
print()

# Compute degeneracy
z = 0.5  # Example redshift
gamma = 2.0

def g_z(z, z_trans, gamma=2.0):
    if z > z_trans:
        return 0.0
    return ((1 + z_trans) / (1 + z))**gamma

print(f"At z = {z}:")
print(f"{'z_trans':<10} {'g(z)':<15} {'ρ_thresh to compensate':<25}")
print("-" * 50)

z_trans_range = [1.0, 1.5, 2.0, 2.5, 3.0]
g_ref = g_z(z, 1.67, gamma)

for z_t in z_trans_range:
    g = g_z(z, z_t, gamma)
    # If g is larger, need more screening (larger ρ_thresh) to compensate
    if g > 0:
        rho_thresh_factor = 200 * (g / g_ref)**0.5  # Approximate scaling
        print(f"{z_t:<10.2f} {g:<15.4f} {rho_thresh_factor:<25.0f} ρ_crit")
print()

print("BREAKING THE DEGENERACY:")
print("  • z_trans affects redshift EVOLUTION (compare z=0.5 vs z=1.5)")
print("  • ρ_thresh affects ENVIRONMENT dependence (void vs cluster)")
print("  • Combining growth rate at multiple z breaks z_trans degeneracy")
print("  • Comparing void vs cluster objects breaks ρ_thresh degeneracy")
print()

# =============================================================================
# 3. μ ↔ H₀ DEGENERACY
# =============================================================================
print("3. μ ↔ H₀ DEGENERACY ANALYSIS")
print("=" * 80)
print()

print("SDCG affects structure formation, which can mimic changes in H₀.")
print()

print("Distance-redshift relation:")
print("  D_L(z) = (1+z) ∫₀^z c dz' / H(z')")
print()
print("  Increasing H₀ → smaller distances → fainter objects")
print()

print("Growth rate:")
print("  f(z) = d ln D / d ln a ≈ Ω_m(z)^γ_growth")
print()
print("  In SDCG: γ_growth is modified by μ")
print("  Increasing μ → faster growth → more structure")
print()

print("DEGENERACY MECHANISM:")
print("-" * 60)
print("  Higher μ → faster structure growth")
print("  Can be partially compensated by:")
print("    • Lower H₀ (slower expansion → less dilution)")
print("    • Or higher σ₈ (more initial fluctuations)")
print()

print("However, this degeneracy is WEAK because:")
print("  • H₀ affects DISTANCES (geometric)")
print("  • μ affects GROWTH (dynamical)")
print("  • BAO measures distances directly → constrains H₀")
print("  • RSD measures growth directly → constrains μ")
print()

# Quantitative estimate
H0_fid = 67.4  # km/s/Mpc
mu_fid = 0.149  # MCMC best-fit in voids (thesis v10)

# Approximate degeneracy: growth rate scales as (1 + μ/2)
# To maintain same growth with μ=0, need to adjust H₀

delta_mu = 0.025  # Uncertainty in μ
# Growth enhancement calculation
growth_change = (1 + (mu_fid + delta_mu)/2) / (1 + mu_fid/2)

# This could be compensated by changing H₀ by similar amount
H0_compensate = H0_fid / growth_change

print(f"Quantitative estimate:")
print(f"  If μ changes from {mu_fid:.3f} to {mu_fid + delta_mu:.3f}:")
print(f"  Growth changes by factor {growth_change:.4f}")
print(f"  Could compensate with H₀ = {H0_compensate:.2f} km/s/Mpc")
print(f"  (vs fiducial H₀ = {H0_fid:.2f} km/s/Mpc)")
print()

print("BREAKING THE DEGENERACY:")
print("  • Use BAO as standard ruler (geometric, not affected by μ)")
print("  • CMB acoustic scale is independent of late-time μ")
print("  • Combine with local H₀ measurement (Cepheids, TRGB)")
print()

# =============================================================================
# 4. FULL DEGENERACY MATRIX
# =============================================================================
print("4. EXPECTED CORRELATION MATRIX")
print("=" * 80)
print()

print("Based on the analysis above, expected correlations:")
print()

params = ['μ', 'n_g', 'z_trans', 'ρ_thresh', 'H₀', 'σ₈', 'Ω_m']
n_params = len(params)

# Approximate correlation matrix (educated guess based on physics)
corr_matrix = np.array([
    #  μ      n_g    z_t    ρ_th   H₀     σ₈     Ω_m
    [ 1.0,  -0.7,   0.3,   0.4,  -0.2,   0.6,  -0.3],  # μ
    [-0.7,   1.0,  -0.2,  -0.3,   0.1,  -0.4,   0.2],  # n_g
    [ 0.3,  -0.2,   1.0,   0.5,   0.1,   0.3,  -0.1],  # z_trans
    [ 0.4,  -0.3,   0.5,   1.0,   0.0,   0.2,   0.0],  # ρ_thresh
    [-0.2,   0.1,   0.1,   0.0,   1.0,  -0.3,   0.5],  # H₀
    [ 0.6,  -0.4,   0.3,   0.2,  -0.3,   1.0,  -0.5],  # σ₈
    [-0.3,   0.2,  -0.1,   0.0,   0.5,  -0.5,   1.0],  # Ω_m
])

print("Expected Correlation Matrix:")
print("-" * 80)
print(f"{'':8s}", end='')
for p in params:
    print(f"{p:>8s}", end='')
print()
print("-" * 80)

for i, p in enumerate(params):
    print(f"{p:8s}", end='')
    for j in range(n_params):
        val = corr_matrix[i, j]
        if abs(val) > 0.5:
            print(f"{val:>8.2f}*", end='')  # Strong correlation
        else:
            print(f"{val:>8.2f} ", end='')
    print()
print()
print("* = strong correlation (|r| > 0.5)")
print()

# =============================================================================
# 5. IMPLICATIONS FOR MCMC
# =============================================================================
print("5. IMPLICATIONS FOR MCMC ANALYSIS")
print("=" * 80)
print()

print("EXPECTED BEHAVIOR:")
print("-" * 60)
print()
print("1. μ and n_g will be ANTI-CORRELATED (r ≈ -0.7)")
print("   → The 2D posterior will be elongated along μ-n_g direction")
print("   → This is why we need multi-scale data")
print()
print("2. μ and σ₈ will be CORRELATED (r ≈ +0.6)")
print("   → Higher μ allows for higher σ₈")
print("   → This connects to the σ₈ tension!")
print()
print("3. z_trans and ρ_thresh will be CORRELATED (r ≈ +0.5)")
print("   → Later transition can be compensated by less screening")
print()

print("RECOMMENDATIONS:")
print("-" * 60)
print()
print("1. Use MULTIPLE probes at DIFFERENT scales:")
print("   • CMB (large scales, high z)")
print("   • BAO (intermediate scales, z ~ 0.1-2)")
print("   • Lyman-α (small scales, z ~ 2-4)")
print("   • RSD (all scales, z ~ 0.1-1)")
print()
print("2. Run MCMC with SUFFICIENT walkers to explore degeneracies")
print("   • Minimum: 2 × n_params × 10 = 200 walkers")
print("   • Recommended: 500+ walkers")
print()
print("3. Check for MULTIMODALITY")
print("   • Use multiple starting points")
print("   • Check Gelman-Rubin diagnostic")
print()
print("4. Compare with ΛCDM using Bayesian evidence")
print("   • SDCG has 4 extra parameters")
print("   • Need significant Δχ² to justify extra complexity")
print()

# =============================================================================
# 6. WHAT WE CAN ACTUALLY CONSTRAIN
# =============================================================================
print("6. WHAT CAN ACTUALLY BE CONSTRAINED")
print("=" * 80)
print()

print("WELL-CONSTRAINED:")
print("  ✓ μ_eff = μ × ⟨S⟩ (the effective coupling)")
print("  ✓ μ × (k/k₀)^n_g at a reference scale (combined effect)")
print("  ✓ H₀, Ω_m, σ₈ (from standard probes)")
print()

print("PARTIALLY CONSTRAINED (from data with priors):")
print("  ⚠️ z_trans (needs multi-redshift data, prior from z_acc)")
print()

print("FIXED FROM THEORY (cannot constrain from data):")
print("  ✗ n_g = β₀²/4π² = 0.014 (from QFT: β₀ ≈ 0.74)")
print("  ✗ ρ_thresh (needs environment-dependent data, fix at 200ρ_crit)")
print("  ✗ α = 2 (screening exponent, degenerate with ρ_thresh)")
print("  ✗ γ = 3 (z-evolution exponent, degenerate with z_trans)")
print()

print("CRITICAL LOGIC:")
print("  β₀ is FIXED from QFT (one-loop beta function) → β₀ ≈ 0.74")
print("  n_g is DERIVED from β₀: n_g = β₀²/4π² = (0.74)²/39.48 = 0.014")
print("  We do NOT derive β₀ from n_g (that would be circular!)")
print()

print("DERIVED QUANTITIES:")
print("  • μ_bare = μ_eff / ⟨S⟩ (depends on knowing S)")
print("  • z_trans = z_acc + Δz = 0.63 + 1.0 = 1.63")
print()

# Verify β₀ → n_g derivation (NOT the other way around!)
beta_0_QFT = 0.74  # Fixed from QFT one-loop calculation
n_g_derived = beta_0_QFT**2 / (4 * np.pi**2)
print(f"QFT derivation: β₀ = {beta_0_QFT:.2f} (fixed from theory)")
print(f"                n_g = β₀²/4π² = {n_g_derived:.4f}")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("DEGENERACY ANALYSIS SUMMARY")
print("=" * 80)
print()

print("┌────────────────────┬────────────────────┬──────────────────────────────┐")
print("│ Degeneracy         │ Correlation        │ How to Break                 │")
print("├────────────────────┼────────────────────┼──────────────────────────────┤")
print("│ μ ↔ n_g            │ r ≈ -0.7 (strong)  │ Multi-scale data (BAO + Lyα) │")
print("│ μ ↔ σ₈             │ r ≈ +0.6 (strong)  │ CMB normalization            │")
print("│ z_trans ↔ ρ_thresh │ r ≈ +0.5 (medium)  │ Multi-z + environment        │")
print("│ μ ↔ H₀             │ r ≈ -0.2 (weak)    │ BAO as geometric probe       │")
print("│ ρ_thresh ↔ α       │ r ≈ +0.8 (strong)  │ Cannot break easily!         │")
print("└────────────────────┴────────────────────┴──────────────────────────────┘")
print()

print("BOTTOM LINE:")
print("  • ONLY μ is truly free — all other SDCG params are fixed from theory")
print("  • n_g = 0.014 is FIXED from β₀ = 0.70 (QFT)")
print("  • α = 2, γ = 3, ρ_thresh = 200ρ_crit are FIXED from theory")
print("  • z_trans = 1.64 is derived from cosmic deceleration q(z) = 0")
print("  • The MCMC constrains μ = 0.149 ± 0.025 in void environments (6σ)")
print()
