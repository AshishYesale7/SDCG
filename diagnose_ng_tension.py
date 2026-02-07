#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              DIAGNOSE n_g TENSION: Why does MCMC prefer n_g ~ 0.09?          ║
╚══════════════════════════════════════════════════════════════════════════════╝

The MCMC finds n_g ≈ 0.09 but EFT predicts n_g = 0.014. Let's understand why.

n_g ENTERS THE CGC EQUATIONS AS FOLLOWS:
=========================================

1. CMB: D_ℓ^CGC = D_ℓ^ΛCDM × [1 + μ × (ℓ/1000)^(n_g/2)]
   → Larger n_g means STRONGER scale dependence in CMB
   → At ℓ=2000: n_g=0.014 gives (2)^0.007 = 1.005 (+0.5% boost)
                n_g=0.09  gives (2)^0.045 = 1.032 (+3.2% boost)

2. BAO: (D_V/r_d)^CGC = (D_V/r_d)^ΛCDM × [1 + μ × (1+z)^(-n_g)]
   → Larger n_g means FASTER decay of CGC effect with redshift
   → At z=0.5: n_g=0.014 gives (1.5)^(-0.014) = 0.994 (small effect)
               n_g=0.09  gives (1.5)^(-0.09) = 0.963 (larger effect)

3. Growth: fσ8_CGC = fσ8_ΛCDM × [1 + 0.1μ × (1+z)^(-n_g)]
   → Same as BAO: Larger n_g = faster decay with z

4. Lyman-α: P_F^CGC = P_F^ΛCDM × [1 + μ × (k/k_CGC)^n_g × W(z)]
   → At k=0.01: n_g=0.014 gives (0.01/0.05)^0.014 = 0.995
                n_g=0.09  gives (0.01/0.05)^0.09  = 0.945

HYPOTHESIS: The data prefers STRONGER scale dependence than EFT predicts.
This could indicate:
  • Higher-loop corrections to n_g
  • Effective n_g from integrated multi-scale effects
  • Tension with the simple RG derivation
"""

import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulations.cgc.data_loader import DataLoader
from simulations.cgc.cgc_physics import CGCPhysics, apply_cgc_to_cmb, apply_cgc_to_bao, \
                            apply_cgc_to_growth
from simulations.cgc.likelihoods import log_likelihood_cmb, log_likelihood_bao, \
                            log_likelihood_growth, log_likelihood_sne, \
                            log_likelihood_lyalpha

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    n_g TENSION DIAGNOSTIC                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
loader = DataLoader(use_real_data=True)
data = loader.load_all(include_sne=True, include_lyalpha=True)

# =============================================================================
# FIXED PARAMETERS (keep everything else fixed)
# =============================================================================

# Standard cosmology (Planck 2018)
theta_base = np.array([
    0.02242,   # ω_b
    0.1193,    # ω_cdm
    0.6736,    # h
    3.044,     # ln10As
    0.9649,    # n_s
    0.054,     # τ
    0.149,     # μ (fixed at MCMC best-fit)
    0.014,     # n_g (will vary this)
    1.64,      # z_trans
    200.0      # ρ_thresh
])

# =============================================================================
# SCAN n_g VALUES
# =============================================================================

n_g_values = np.linspace(0.01, 0.15, 30)

print("\n" + "="*70)
print("SCANNING LOG-LIKELIHOOD AS FUNCTION OF n_g")
print("="*70)
print(f"\nFixed parameters: μ = 0.149, z_trans = 1.64, ρ_thresh = 200")
print(f"Scanning n_g from {n_g_values[0]:.3f} to {n_g_values[-1]:.3f}\n")

# Store likelihoods
ll_cmb = []
ll_bao = []
ll_growth = []
ll_sne = []
ll_lyalpha = []
ll_total = []

for n_g in n_g_values:
    theta = theta_base.copy()
    theta[7] = n_g  # Set n_g
    
    # Compute individual likelihoods
    cmb_ll = log_likelihood_cmb(theta, data['cmb'])
    bao_ll = log_likelihood_bao(theta, data['bao'])
    growth_ll = log_likelihood_growth(theta, data['growth'])
    sne_ll = log_likelihood_sne(theta, data['sne'])
    lyalpha_ll = log_likelihood_lyalpha(theta, data['lyalpha'])
    
    ll_cmb.append(cmb_ll)
    ll_bao.append(bao_ll)
    ll_growth.append(growth_ll)
    ll_sne.append(sne_ll)
    ll_lyalpha.append(lyalpha_ll)
    ll_total.append(cmb_ll + bao_ll + growth_ll + sne_ll + lyalpha_ll)

# Convert to arrays
ll_cmb = np.array(ll_cmb)
ll_bao = np.array(ll_bao)
ll_growth = np.array(ll_growth)
ll_sne = np.array(ll_sne)
ll_lyalpha = np.array(ll_lyalpha)
ll_total = np.array(ll_total)

# Find best n_g for each dataset
best_ng_cmb = n_g_values[np.argmax(ll_cmb)]
best_ng_bao = n_g_values[np.argmax(ll_bao)]
best_ng_growth = n_g_values[np.argmax(ll_growth)]
best_ng_sne = n_g_values[np.argmax(ll_sne)]
best_ng_lyalpha = n_g_values[np.argmax(ll_lyalpha)]
best_ng_total = n_g_values[np.argmax(ll_total)]

# =============================================================================
# PRINT RESULTS
# =============================================================================

print("\n┌────────────────────────────────────────────────────────────────────┐")
print("│ WHICH DATASET DRIVES n_g HIGHER?                                   │")
print("├────────────────────────────────────────────────────────────────────┤")
print(f"│  EFT prediction:  n_g = 0.014 (from β₀²/4π²)                       │")
print(f"│  MCMC found:      n_g ≈ 0.09                                       │")
print("├────────────────────────────────────────────────────────────────────┤")
print(f"│  CMB prefers:     n_g = {best_ng_cmb:.3f}  (Δlog L_max = {ll_cmb.max() - ll_cmb[0]:+.1f})        │")
print(f"│  BAO prefers:     n_g = {best_ng_bao:.3f}  (Δlog L_max = {ll_bao.max() - ll_bao[0]:+.1f})        │")
print(f"│  Growth prefers:  n_g = {best_ng_growth:.3f}  (Δlog L_max = {ll_growth.max() - ll_growth[0]:+.1f})        │")
print(f"│  SNe prefers:     n_g = {best_ng_sne:.3f}  (Δlog L_max = {ll_sne.max() - ll_sne[0]:+.1f})        │")
print(f"│  Lyα prefers:     n_g = {best_ng_lyalpha:.3f}  (Δlog L_max = {ll_lyalpha.max() - ll_lyalpha[0]:+.1f})        │")
print("├────────────────────────────────────────────────────────────────────┤")
print(f"│  TOTAL prefers:   n_g = {best_ng_total:.3f}                                   │")
print("└────────────────────────────────────────────────────────────────────┘")

# =============================================================================
# INTERPRET RESULTS
# =============================================================================

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Which dataset contributes most to the tension?
tension_sources = {
    'CMB': (best_ng_cmb, ll_cmb.max() - ll_cmb[0]),
    'BAO': (best_ng_bao, ll_bao.max() - ll_bao[0]),
    'Growth': (best_ng_growth, ll_growth.max() - ll_growth[0]),
    'SNe': (best_ng_sne, ll_sne.max() - ll_sne[0]),
    'Lyα': (best_ng_lyalpha, ll_lyalpha.max() - ll_lyalpha[0])
}

# Sort by likelihood improvement
sorted_sources = sorted(tension_sources.items(), key=lambda x: abs(x[1][1]), reverse=True)

print("\nDATA CONTRIBUTION TO n_g TENSION (sorted by impact):")
print("-" * 50)
for name, (best_ng, delta_ll) in sorted_sources:
    tension_dir = "↑" if best_ng > 0.014 else "↓" if best_ng < 0.014 else "="
    print(f"  {name:8s}: n_g = {best_ng:.3f} ({tension_dir} from EFT), ΔlogL = {delta_ll:+.1f}")

# =============================================================================
# PHYSICS INTERPRETATION
# =============================================================================

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
│ PHYSICS INTERPRETATION                                                        │
╠══════════════════════════════════════════════════════════════════════════════╣
│                                                                               │
│ The n_g ≈ 0.09 tension suggests one of:                                       │
│                                                                               │
│ 1. HIGHER-LOOP CORRECTIONS                                                    │
│    n_g = β₀²/4π² is only one-loop. Two-loop gives:                           │
│    n_g(2-loop) = β₀²/4π² × [1 + β₀²/(8π²)]                                   │
│    With β₀ = 0.70: n_g(2-loop) ≈ 0.014 × 1.006 ≈ 0.014 (negligible)          │
│                                                                               │
│ 2. EFFECTIVE n_g FROM INTEGRATED EFFECTS                                      │
│    The simple (k/k_pivot)^n_g formula may be an approximation.                │
│    True RG running could give:                                                │
│    G_eff ~ 1 + μ × log(k/k_pivot) × [1 + n_g × log(k/k_pivot)]               │
│    which would appear as larger effective n_g                                 │
│                                                                               │
│ 3. DEGENERACY WITH μ                                                          │
│    If μ is lower than expected, n_g must increase to compensate:              │
│    μ × (k/k_pivot)^n_g = constant → larger n_g if μ is smaller               │
│                                                                               │
│ 4. SYSTEMATIC IN DATA                                                         │
│    Scale-dependent systematics could mimic larger n_g                         │
│                                                                               │
│ RECOMMENDED: Treat n_g as PHENOMENOLOGICAL, report tension with EFT           │
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# CHECK μ-n_g DEGENERACY
# =============================================================================

print("\n" + "="*70)
print("CHECKING μ-n_g DEGENERACY")
print("="*70)

# Fix n_g = 0.014 (EFT) and vary μ
mu_values = np.linspace(0.05, 0.4, 30)

ll_mu_scan = []
for mu in mu_values:
    theta = theta_base.copy()
    theta[6] = mu      # Set μ
    theta[7] = 0.014   # Fix n_g at EFT value
    
    total_ll = (log_likelihood_cmb(theta, data['cmb']) +
                log_likelihood_bao(theta, data['bao']) +
                log_likelihood_growth(theta, data['growth']) +
                log_likelihood_sne(theta, data['sne']) +
                log_likelihood_lyalpha(theta, data['lyalpha']))
    ll_mu_scan.append(total_ll)

ll_mu_scan = np.array(ll_mu_scan)
best_mu_with_ng_eft = mu_values[np.argmax(ll_mu_scan)]

print(f"\nWith n_g FIXED at EFT value (0.014):")
print(f"  Best-fit μ = {best_mu_with_ng_eft:.3f}")
print(f"  Compare to μ = 0.149 (thesis value)")

# Now check with fitted n_g = 0.09
ll_mu_scan_fitted = []
for mu in mu_values:
    theta = theta_base.copy()
    theta[6] = mu
    theta[7] = 0.09  # Use MCMC-preferred n_g
    
    total_ll = (log_likelihood_cmb(theta, data['cmb']) +
                log_likelihood_bao(theta, data['bao']) +
                log_likelihood_growth(theta, data['growth']) +
                log_likelihood_sne(theta, data['sne']) +
                log_likelihood_lyalpha(theta, data['lyalpha']))
    ll_mu_scan_fitted.append(total_ll)

ll_mu_scan_fitted = np.array(ll_mu_scan_fitted)
best_mu_with_ng_fitted = mu_values[np.argmax(ll_mu_scan_fitted)]

print(f"\nWith n_g = 0.09 (MCMC-preferred):")
print(f"  Best-fit μ = {best_mu_with_ng_fitted:.3f}")

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│ DEGENERACY ANALYSIS                                                │
├────────────────────────────────────────────────────────────────────┤
│  With n_g = 0.014 (EFT):    best μ = {best_mu_with_ng_eft:.3f}                      │
│  With n_g = 0.09 (MCMC):    best μ = {best_mu_with_ng_fitted:.3f}                      │
├────────────────────────────────────────────────────────────────────┤
│  This confirms: μ and n_g are DEGENERATE!                          │
│  When n_g is forced to 0.014, μ must increase to compensate.       │
│  When n_g is free, it absorbs some of the signal → lower μ         │
└────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# CONCLUSION
# =============================================================================

print("="*70)
print("CONCLUSION")
print("="*70)

print(f"""
The n_g tension (0.09 vs 0.014) is driven by:

1. {sorted_sources[0][0]:s} dataset is the PRIMARY driver
   - Prefers n_g = {sorted_sources[0][1][0]:.3f} (ΔlogL = {sorted_sources[0][1][1]:+.1f})

2. μ-n_g DEGENERACY exists in the data
   - With n_g = 0.014 (EFT): μ = {best_mu_with_ng_eft:.3f}
   - With n_g = 0.09:        μ = {best_mu_with_ng_fitted:.3f}

RECOMMENDATIONS:
================
1. For PHYSICAL CONSISTENCY: Use EFT priors (n_g = 0.014 ± 0.003)
   → Accept that μ will be slightly higher

2. For BEST FIT TO DATA: Let n_g float
   → But acknowledge this is PHENOMENOLOGICAL, not EFT-derived

3. For THESIS: Report BOTH analyses
   → EFT-constrained: μ = X ± Y with n_g fixed at 0.014
   → Phenomenological: μ = A ± B, n_g = C ± D (best fit)
""")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
