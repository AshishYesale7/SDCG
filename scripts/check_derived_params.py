#!/usr/bin/env python3
"""
CRITICAL QUESTION: Are ρ_thresh, z_trans, α_CGC Derived or Fitted?
===================================================================
Checking the theory derivations for these parameters.
"""

import numpy as np

print("="*80)
print("THEORY-DERIVED VALUES vs MCMC-FITTED VALUES")
print("="*80)

# =============================================================================
# 1. ρ_thresh: THEORY DERIVATION
# =============================================================================
print("\n" + "="*80)
print("1. ρ_thresh (Screening Threshold)")
print("="*80)

print("""
THEORY DERIVATION:
  Screening activates when scalar force F_φ ~ gravitational force F_G
  This happens at the virial overdensity of halos.
  
  For standard cosmology: Δ_vir ≈ 200 (Bryan & Norman 1998)
  
  Therefore: ρ_thresh = Δ_vir × ρ_crit ≈ 200 ρ_crit
""")

rho_thresh_theory = 200  # From virial theorem
rho_thresh_mcmc = 242.5  # From MCMC fit

print(f"  THEORY PREDICTION:  ρ_thresh = {rho_thresh_theory} ρ_crit")
print(f"  MCMC FIT:           ρ_thresh = {rho_thresh_mcmc} ρ_crit")
print(f"  DIFFERENCE:         {(rho_thresh_mcmc - rho_thresh_theory)/rho_thresh_theory*100:.0f}%")
print(f"\n  VERDICT: MCMC agrees with theory within ~20%!")
print(f"           We COULD fix ρ_thresh = 200 and not fit it.")

# =============================================================================
# 2. z_trans: THEORY DERIVATION
# =============================================================================
print("\n" + "="*80)
print("2. z_trans (Transition Redshift)")
print("="*80)

print("""
THEORY DERIVATION:
  CGC activates when dark energy becomes dynamically important.
  
  Step 1: Matter-DE equality
    z_eq = (Ω_Λ/Ω_m)^(1/3) - 1
         = (0.685/0.315)^(1/3) - 1 = 0.30
  
  Step 2: Scalar response time (one e-fold delay)
    Δz ≈ 1
  
  Result: z_trans = z_eq + Δz ≈ 1.30
  
  Alternative: deceleration q(z) = 0 occurs at:
    q(z) = Ω_m(1+z)³/(2[Ω_m(1+z)³ + Ω_Λ]) - Ω_Λ/[Ω_m(1+z)³ + Ω_Λ] = 0
    Solving: z_q=0 ≈ 0.64
    
    With delay: z_trans = z_q=0 + 1 ≈ 1.64
""")

Omega_m = 0.315
Omega_L = 0.685

z_eq = (Omega_L / Omega_m)**(1/3) - 1
z_trans_v1 = z_eq + 1.0  # With 1 e-fold delay

# Solve for q(z) = 0
z_q0 = (2*Omega_L/Omega_m)**(1/3) - 1
z_trans_v2 = z_q0 + 1.0  # With 1 e-fold delay

z_trans_mcmc = 2.14

print(f"  THEORY (method 1): z_trans = z_eq + 1 = {z_trans_v1:.2f}")
print(f"  THEORY (method 2): z_trans = z_q=0 + 1 = {z_trans_v2:.2f}")
print(f"  MCMC FIT:          z_trans = {z_trans_mcmc:.2f}")
print(f"\n  VERDICT: MCMC value is higher than theory predictions!")
print(f"           Theory says 1.3-1.6, MCMC says 2.1")
print(f"           This suggests either:")
print(f"           - The delay is ~2 e-folds, not 1")
print(f"           - Or this parameter genuinely needs fitting")

# =============================================================================
# 3. α_CGC (phenomenological exponent in MCMC)
# =============================================================================
print("\n" + "="*80)
print("3. α_CGC (Scale Exponent in MCMC)")
print("="*80)

print("""
IMPORTANT: This is where there's CONFUSION!

In MCMC, the parameter called 'n_g' is used as:
  D_ℓ^CGC = D_ℓ^ΛCDM × [1 + μ × (ℓ/1000)^(n_g/2)]
  
This is a PHENOMENOLOGICAL power-law, NOT the EFT n_g!

THEORY n_g (EFT):
  n_g^(EFT) = β₀²/(4π²) = 0.70²/39.48 = 0.0124
  
  This describes G_eff(k) = G_N × [1 + n_g × ln(k/k_*)]
  
MCMC "n_g" (phenomenological):
  n_g^(phenom) = 0.906 ± 0.063
  
  This is just a power-law index that fits the data.
""")

n_g_eft = 0.70**2 / (4 * np.pi**2)
n_g_mcmc = 0.906

print(f"  EFT THEORY:  n_g = {n_g_eft:.5f}")
print(f"  MCMC FIT:    n_g = {n_g_mcmc:.3f}")
print(f"\n  VERDICT: These are COMPLETELY DIFFERENT quantities!")
print(f"           The MCMC 'n_g' should be renamed to α_scale or p_CGC")

# =============================================================================
# 4. μ: CAN IT BE DERIVED?
# =============================================================================
print("\n" + "="*80)
print("4. μ (CGC Coupling) - Can It Be Derived?")
print("="*80)

print("""
THEORY ATTEMPT:
  From QFT: μ_bare = β₀² × ln(M_Pl/H₀) / (16π²)
                   = 0.70² × 138 / (16π²)
                   = 0.43
  
  This is CLOSE to the MCMC value of 0.467!
""")

ln_Mpl_H0 = 138
mu_theory = 0.70**2 * ln_Mpl_H0 / (16 * np.pi**2)
mu_mcmc = 0.467

print(f"  THEORY (QFT):  μ = β₀² × ln(M_Pl/H₀) / (16π²) = {mu_theory:.3f}")
print(f"  MCMC FIT:      μ = {mu_mcmc:.3f}")
print(f"  DIFFERENCE:    {(mu_mcmc - mu_theory)/mu_theory*100:.0f}%")
print(f"\n  VERDICT: Theory and MCMC agree to ~10%!")

# =============================================================================
# SUMMARY: WHAT'S TRULY DERIVED vs FITTED?
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: DERIVED vs FITTED")
print("="*80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ Parameter   │ Theory Value │ MCMC Value    │ Agreement │ Status           │
├─────────────┼──────────────┼───────────────┼───────────┼──────────────────┤
│ β₀          │ 0.70         │ (not fitted)  │ N/A       │ ✓ FULLY DERIVED  │
│ n_g (EFT)   │ 0.0124       │ (not fitted)  │ N/A       │ ✓ FULLY DERIVED  │
├─────────────┼──────────────┼───────────────┼───────────┼──────────────────┤
│ μ           │ 0.43         │ 0.467 ± 0.027 │ ~10%      │ ≈ DERIVED        │
│ ρ_thresh    │ 200          │ 242.5 ± 98.2  │ ~20%      │ ≈ DERIVED        │
│ z_trans     │ 1.3-1.6      │ 2.14 ± 0.52   │ ~50%      │ ⚠ FITTED         │
│ α_scale     │ N/A*         │ 0.906 ± 0.063 │ N/A       │ ⚠ FITTED         │
└─────────────────────────────────────────────────────────────────────────────┘

* The MCMC α_scale is a phenomenological exponent, NOT related to n_g^(EFT)

CONCLUSION:
  
  ✓ β₀, n_g^(EFT): FULLY DERIVED from particle physics (no fitting)
  
  ≈ μ, ρ_thresh: CAN BE DERIVED from theory, MCMC confirms to ~20%
                 We COULD fix these to theory values and still fit!
  
  ⚠ z_trans: Theory predicts ~1.5, MCMC prefers ~2.1 
             Either theory needs refinement or this is genuinely free
  
  ⚠ α_scale: This is PURELY PHENOMENOLOGICAL in our MCMC
             It should ideally be connected to n_g^(EFT) somehow
""")

# =============================================================================
# REVISED PARAMETER COUNT
# =============================================================================
print("\n" + "="*80)
print("REVISED FREE PARAMETER COUNT")
print("="*80)

print("""
IF WE FIX theory-derivable parameters:

  FIXED (from theory):
    β₀ = 0.70            ← From particle physics
    n_g = 0.0124         ← From β₀
    μ = 0.43             ← From β₀² × ln(M_Pl/H₀) / 16π²
    ρ_thresh = 200       ← From virial theorem
    
  FITTED (with priors):
    z_trans ~ 1.5-2.5    ← Theory gives ~1.5, but needs some freedom
    α_scale ~ 0.5-1.0    ← Phenomenological (should connect to n_g)
    + 6 cosmological     ← Standard ΛCDM
    
  TOTAL: 8 free parameters (6 cosmology + 2 SDCG)
         vs 10 in current MCMC
         vs 6 for ΛCDM

HONEST ANSWER TO YOUR QUESTION:
  YES, ρ_thresh and μ CAN be derived from theory!
  z_trans is approximately derivable but needs some freedom.
  α_scale is phenomenological and should ideally be eliminated.
""")

print("="*80)
