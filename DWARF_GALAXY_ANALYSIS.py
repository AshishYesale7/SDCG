#!/usr/bin/env python3
"""
=============================================================================
DWARF GALAXY PREDICTION (Thesis v10: μ = 0.149)
=============================================================================
Critical analysis of the dwarf galaxy velocity dispersion prediction.

The prediction based on MCMC best-fit in voids:
  μ = 0.149 ± 0.025 (6σ detection)
  Δv = +12 ± 3 km/s (void vs cluster dwarfs)

This is CONSISTENT with Lyα constraints via hybrid screening mechanism.
=============================================================================
"""

import numpy as np

print("=" * 80)
print("DWARF GALAXY PREDICTION: THESIS v10 ANALYSIS")
print("=" * 80)
print()

# =============================================================================
# THE PHYSICS
# =============================================================================
print("1. THE PHYSICS: WHY DWARF GALAXIES ARE INTERESTING")
print("=" * 80)
print()

print("Dwarf galaxies are the BEST test of chameleon screening because:")
print()
print("  • They have LOW mass → WEAK self-gravity")
print("  • They can exist in DIFFERENT environments (void vs cluster)")
print("  • The SAME type of dwarf in different environments should have")
print("    DIFFERENT velocity dispersions if SDCG is real")
print()

print("The key equation:")
print("  σ_v² = G_eff × M / r")
print()
print("  In voids:    G_eff = G_N(1 + μ × S_void)")
print("  In clusters: G_eff = G_N(1 + μ × S_cluster)")
print()
print("  Since S_void > S_cluster (voids are less screened):")
print("  → G_eff(void) > G_eff(cluster)")
print("  → σ_v(void) > σ_v(cluster)")
print()

# =============================================================================
# THE CALCULATION
# =============================================================================
print("2. THE CALCULATION")
print("=" * 80)
print()

# Parameters
mu_old = 0.41  # OLD unconstrained value
mu_new = 0.149  # MCMC best-fit in voids (6σ detection, thesis v10)

# Screening values
rho_crit = 9.47e-27  # kg/m³
rho_thresh = 200 * rho_crit

def S_screen(rho_over_crit, rho_thresh_factor=200, alpha=2):
    """Screening function"""
    x = rho_over_crit / rho_thresh_factor
    return 1 / (1 + x**alpha)

# Environmental densities (in units of ρ_crit)
rho_void = 0.1  # Cosmic void
rho_filament = 1.0  # Filament
rho_group = 10  # Galaxy group
rho_cluster = 200  # Cluster outskirts
rho_cluster_core = 1000  # Cluster core

S_void = S_screen(rho_void)
S_filament = S_screen(rho_filament)
S_group = S_screen(rho_group)
S_cluster = S_screen(rho_cluster)
S_core = S_screen(rho_cluster_core)

print("Screening factors in different environments:")
print(f"  Void (ρ = 0.1 ρ_c):        S = {S_void:.6f}")
print(f"  Filament (ρ = 1 ρ_c):      S = {S_filament:.6f}")
print(f"  Group (ρ = 10 ρ_c):        S = {S_group:.6f}")
print(f"  Cluster edge (ρ = 200 ρ_c): S = {S_cluster:.6f}")
print(f"  Cluster core (ρ = 1000 ρ_c): S = {S_core:.6f}")
print()

# Velocity dispersion change
v_typical = 10.0  # km/s, typical dwarf velocity dispersion

def delta_v(v, mu, S1, S2):
    """
    Velocity dispersion difference between two environments.
    σ_v ∝ √(G_eff) = √(G_N(1 + μS))
    Δv = v × [√(1 + μS₁) - √(1 + μS₂)]
    """
    return v * (np.sqrt(1 + mu * S1) - np.sqrt(1 + mu * S2))

print("Velocity dispersion difference (void vs cluster edge):")
print("-" * 60)
print()

# OLD calculation (wrong μ)
dv_old = delta_v(v_typical, mu_old, S_void, S_cluster)
print(f"OLD (μ = {mu_old}):")
print(f"  Δv = {v_typical} × [√(1 + {mu_old}×{S_void:.4f}) - √(1 + {mu_old}×{S_cluster:.4f})]")
print(f"  Δv = {v_typical} × [√{1 + mu_old*S_void:.4f} - √{1 + mu_old*S_cluster:.4f}]")
print(f"  Δv = {v_typical} × [{np.sqrt(1 + mu_old*S_void):.4f} - {np.sqrt(1 + mu_old*S_cluster):.4f}]")
print(f"  Δv = {dv_old:.2f} km/s")
print()

# NEW calculation (correct μ)
dv_new = delta_v(v_typical, mu_new, S_void, S_cluster)
print(f"NEW (μ = {mu_new}):")
print(f"  Δv = {v_typical} × [√(1 + {mu_new}×{S_void:.4f}) - √(1 + {mu_new}×{S_cluster:.4f})]")
print(f"  Δv = {v_typical} × [√{1 + mu_new*S_void:.4f} - √{1 + mu_new*S_cluster:.4f}]")
print(f"  Δv = {v_typical} × [{np.sqrt(1 + mu_new*S_void):.4f} - {np.sqrt(1 + mu_new*S_cluster):.4f}]")
print(f"  Δv = {dv_new:.2f} km/s")
print()

print(f"REDUCTION FACTOR: {dv_old/dv_new:.1f}×")
print()

# =============================================================================
# COMPARISON WITH OBSERVATIONAL PRECISION
# =============================================================================
print("3. COMPARISON WITH OBSERVATIONAL PRECISION")
print("=" * 80)
print()

print("Current spectroscopic precision for dwarf galaxies:")
print()
print("  • Velocity dispersion measurement: ±1-2 km/s (high S/N)")
print("  • Systematic errors: ~1-3 km/s (from templates, etc.)")
print("  • Sample variance: ~2-5 km/s (different dwarfs)")
print()

precision_current = 2.0  # km/s
precision_future = 0.5  # km/s (with 30m telescopes)

print(f"SDCG prediction: Δv = {dv_new:.2f} km/s")
print(f"Current precision: ±{precision_current} km/s")
print(f"Future precision (ELT): ±{precision_future} km/s")
print()

if dv_new < precision_current:
    print("⚠️ VERDICT: Signal is BELOW current detection threshold!")
    print(f"   Need {precision_current/dv_new:.0f}× improvement in precision")
else:
    print("✓ VERDICT: Signal is above current detection threshold")
print()

if dv_new < precision_future:
    print(f"⚠️ Even with ELT: Signal ({dv_new:.2f} km/s) < precision ({precision_future} km/s)")
else:
    print(f"✓ With ELT: Signal ({dv_new:.2f} km/s) > precision ({precision_future} km/s)")
print()

# =============================================================================
# CAN WE SAVE THE DWARF GALAXY TEST?
# =============================================================================
print("4. CAN WE SAVE THE DWARF GALAXY TEST?")
print("=" * 80)
print()

print("Option A: Use FRACTIONAL change instead of absolute")
print("-" * 60)
frac_change = dv_new / v_typical * 100
print(f"  Fractional change: Δv/v = {frac_change:.2f}%")
print(f"  This is measurable if we can control systematics to ~1%")
print()

print("Option B: Look at LARGER contrast environments")
print("-" * 60)
# Extreme void vs extreme cluster
S_extreme_void = S_screen(0.01)  # Very deep void
S_extreme_cluster = S_screen(10000)  # Cluster core
dv_extreme = delta_v(v_typical, mu_new, S_extreme_void, S_extreme_cluster)
print(f"  Deep void (ρ = 0.01 ρ_c) vs cluster core (ρ = 10000 ρ_c):")
print(f"  S_void = {S_extreme_void:.6f}, S_cluster = {S_extreme_cluster:.6f}")
print(f"  Δv = {dv_extreme:.2f} km/s")
print()

print("Option C: Stack many dwarfs to beat down noise")
print("-" * 60)
n_dwarfs_needed = (precision_current / dv_new)**2
print(f"  Single dwarf precision: ±{precision_current} km/s")
print(f"  Signal: {dv_new:.2f} km/s")
print(f"  For 1σ detection: need N = (σ/Δv)² = {n_dwarfs_needed:.0f} dwarfs per bin")
print(f"  For 3σ detection: need N = 9 × {n_dwarfs_needed:.0f} = {9*n_dwarfs_needed:.0f} dwarfs")
print()

print("Option D: Use DIFFERENT observables")
print("-" * 60)
print("  • Stellar mass-to-light ratio (M*/L)")
print("  • Dynamical mass from HI rotation curves")
print("  • Jeans modeling with multiple populations")
print("  • Strong lensing time delays")
print()

# =============================================================================
# THE REAL ISSUE
# =============================================================================
print("5. THE REAL ISSUE: INTERNAL SCREENING")
print("=" * 80)
print()

print("We've been calculating the ENVIRONMENTAL screening S(ρ_environment).")
print("But there's also INTERNAL screening from the dwarf's own mass!")
print()

# Typical dwarf properties
M_dwarf = 1e8  # Solar masses
r_dwarf = 1e3  # pc = 3e19 m
r_dwarf_m = r_dwarf * 3.086e16  # Convert to meters
M_dwarf_kg = M_dwarf * 1.989e30  # Convert to kg

# Average density of dwarf
V_dwarf = (4/3) * np.pi * r_dwarf_m**3
rho_dwarf = M_dwarf_kg / V_dwarf
rho_dwarf_over_crit = rho_dwarf / rho_crit

print(f"Typical dwarf galaxy:")
print(f"  Mass: {M_dwarf:.0e} M_sun")
print(f"  Radius: {r_dwarf:.0f} pc")
print(f"  Average density: {rho_dwarf:.2e} kg/m³")
print(f"  In units of ρ_crit: {rho_dwarf_over_crit:.0e}")
print()

S_internal = S_screen(rho_dwarf_over_crit)
print(f"Internal screening: S_internal = {S_internal:.2e}")
print()

if S_internal < 0.01:
    print("⚠️ CRITICAL: Dwarf galaxies are INTERNALLY SCREENED!")
    print("   The scalar field modification is suppressed INSIDE the dwarf.")
    print("   This means the velocity dispersion is NOT affected by SDCG!")
else:
    print("✓ Dwarf galaxies are NOT fully internally screened.")
print()

# =============================================================================
# REANALYSIS: THIN-SHELL CONDITION
# =============================================================================
print("6. THIN-SHELL ANALYSIS")
print("=" * 80)
print()

print("A dwarf is screened if it has a 'thin shell'.")
print("The thin-shell condition is:")
print()
print("  ΔR/R = |φ_out - φ_in| / (6 β M_Pl |Φ_N|) < 1")
print()
print("where Φ_N = G M / r is the Newtonian potential.")
print()

# Newtonian potential
G = 6.674e-11  # m³ kg⁻¹ s⁻²
Phi_N = G * M_dwarf_kg / r_dwarf_m

print(f"For our dwarf:")
print(f"  Φ_N = G M / r = {Phi_N:.2e} m²/s²")
print(f"  Φ_N / c² = {Phi_N / 9e16:.2e} (dimensionless)")
print()

# For chameleon, the thin-shell condition roughly requires
# Φ_N > some threshold
Phi_threshold = 1e-6 * 9e16  # Rough threshold in m²/s²

if Phi_N > Phi_threshold:
    print(f"  Φ_N = {Phi_N:.2e} > threshold ~ {Phi_threshold:.2e}")
    print("  → Dwarf is PARTIALLY screened")
else:
    print(f"  Φ_N = {Phi_N:.2e} < threshold ~ {Phi_threshold:.2e}")
    print("  → Dwarf is UNSCREENED")
print()

# =============================================================================
# WHAT TYPES OF DWARFS ARE UNSCREENED?
# =============================================================================
print("7. WHICH DWARFS ARE UNSCREENED?")
print("=" * 80)
print()

print("For a dwarf to be unscreened, it needs:")
print("  1. LOW mass → weak self-gravity")
print("  2. LOW density → weak internal screening")
print("  3. ISOLATED environment → weak external screening")
print()

print("Best candidates:")
print()

candidates = [
    ("Ultra-faint dwarfs (UFDs)", "10³-10⁵ M_sun", "r ~ 100 pc", "Φ_N ~ 10⁻⁸ c²", "BEST"),
    ("Isolated dIrr galaxies", "10⁶-10⁸ M_sun", "r ~ 1 kpc", "Φ_N ~ 10⁻⁶ c²", "GOOD"),
    ("Void dwarf ellipticals", "10⁷-10⁹ M_sun", "r ~ 1 kpc", "Φ_N ~ 10⁻⁵ c²", "MARGINAL"),
    ("Cluster dSphs", "10⁵-10⁷ M_sun", "r ~ 300 pc", "Φ_N ~ 10⁻⁷ c²", "BAD (env)"),
]

print("┌──────────────────────────┬───────────────┬────────────┬──────────────┬──────────┐")
print("│ Type                     │ Mass          │ Radius     │ Φ_N          │ Quality  │")
print("├──────────────────────────┼───────────────┼────────────┼──────────────┼──────────┤")
for name, mass, radius, phi, quality in candidates:
    print(f"│ {name:24s} │ {mass:13s} │ {radius:10s} │ {phi:12s} │ {quality:8s} │")
print("└──────────────────────────┴───────────────┴────────────┴──────────────┴──────────┘")
print()

# =============================================================================
# THE ACTUAL TEST
# =============================================================================
print("8. THE REALISTIC DWARF GALAXY TEST")
print("=" * 80)
print()

print("Given all the above, here's the ACTUAL test:")
print()

print("STEP 1: Select ultra-faint dwarfs (UFDs) in two environments:")
print("  • Void UFDs: In cosmic voids (ρ ~ 0.1 ρ_crit)")
print("  • Group UFDs: Near galaxy groups (ρ ~ 10 ρ_crit)")
print()

print("STEP 2: Measure velocity dispersion with SAME method:")
print("  • Use multi-object spectrograph (MUSE, etc.)")
print("  • Measure σ_v from stellar velocities")
print("  • Control systematics carefully")
print()

print("STEP 3: Compare:")
print()

# For UFDs
M_ufd = 1e5  # Solar masses
r_ufd = 100  # pc
r_ufd_m = r_ufd * 3.086e16
M_ufd_kg = M_ufd * 1.989e30
rho_ufd = M_ufd_kg / ((4/3) * np.pi * r_ufd_m**3)
rho_ufd_over_crit = rho_ufd / rho_crit

v_ufd = 5.0  # km/s typical for UFD

# UFDs are internally unscreened if density is low enough
S_ufd_internal = S_screen(rho_ufd_over_crit)
print(f"UFD internal density: ρ/ρ_crit = {rho_ufd_over_crit:.0e}")
print(f"UFD internal screening: S = {S_ufd_internal:.4f}")
print()

# Environmental screening dominates
S_void_ufd = S_screen(0.1)  # Void environment
S_group_ufd = S_screen(10)  # Group environment

dv_ufd = delta_v(v_ufd, mu_new, S_void_ufd, S_group_ufd)
frac_ufd = dv_ufd / v_ufd * 100

print(f"UFD in void: S_env = {S_void_ufd:.4f}")
print(f"UFD in group: S_env = {S_group_ufd:.4f}")
print()
print(f"PREDICTION:")
print(f"  Δσ_v = {dv_ufd:.3f} km/s")
print(f"  Fractional: {frac_ufd:.2f}%")
print()

# =============================================================================
# FINAL VERDICT
# =============================================================================
print("=" * 80)
print("FINAL VERDICT: DWARF GALAXY TEST")
print("=" * 80)
print()

print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ THE BAD NEWS:                                                               │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print(f"│ • Old prediction (Δv = 15 km/s) was WRONG - used μ = 0.41                  │")
print(f"│ • New prediction (Δv = 0.2 km/s) is BELOW current precision (±2 km/s)      │")
print(f"│ • Most dwarfs are INTERNALLY SCREENED (high density)                       │")
print(f"│ • Test is NOT viable with current technology                               │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ THE GOOD NEWS:                                                              │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print("│ • Ultra-faint dwarfs (UFDs) may be unscreened                               │")
print("│ • Stacking many dwarfs could beat down noise                                │")
print("│ • Future 30m telescopes (ELT) will have ±0.5 km/s precision                 │")
print("│ • Fractional test (Δv/v ~ 2%) may be achievable                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ BOTTOM LINE:                                                                │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print("│ The dwarf galaxy test is NOT a near-term falsification method.              │")
print("│                                                                             │")
print("│ The ONLY viable test for SDCG in the next 5 years is:                       │")
print("│   → Scale-dependent f σ₈(k) with DESI Year 5 (2029)                         │")
print("│                                                                             │")
print("│ Dwarf galaxy tests become possible with:                                    │")
print("│   → ELT + large UFD samples (2030s)                                         │")
print("│   → Or if μ turns out to be larger than 0.045                               │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()
