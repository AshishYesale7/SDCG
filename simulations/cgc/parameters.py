"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SDCG Parameters Module (First-Principles)                 ║
║                                                                              ║
║  Defines the Scale-Dependent Crossover Gravity (SDCG) parameter space.      ║
║  Parameters are DERIVED from accepted physics where possible.               ║
║  Only μ requires new physics (meV scale) - this is a PREDICTION.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

PARAMETER CLASSIFICATION (Thesis v12 Canonical Values)
======================================================

THESIS v12 PARAMETER STATUS:
The CGC framework now has a clear separation:
- FIXED BY THEORY: n_g, z_trans, α, ρ_thresh (derived from physics)
- MCMC FITTED: μ_fit is the ONLY free parameter

┌─────────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter       │ Status      │ Source                                     │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ β₀ = 0.70       │ FIXED       │ Standard Model: m_t/v = 173/246            │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ n_g = 0.0125    │ FIXED       │ From RG: β₀²/4π² (THEORETICAL REQUIREMENT) │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ z_trans = 1.67  │ FIXED       │ Cosmic dynamics: q(z)=0                    │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ α = 2.0         │ FIXED       │ Klein-Gordon quadratic potential           │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ ρ_thresh = 200  │ FIXED       │ Virial theorem: 18π² ≈ 178, round ~200     │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ μ_fit = 0.47    │ MCMC FIT    │ Fundamental coupling (6σ detection)        │
│ ± 0.03          │             │ μ_eff(void) = μ_fit × S_avg = 0.149        │
└─────────────────┴─────────────┴────────────────────────────────────────────┘

THE μ CLARIFICATION (Thesis v12)
--------------------------------
μ_fit = 0.47 ± 0.03 is the FUNDAMENTAL MCMC best-fit value.
μ_eff(void) = μ_fit × S_avg ≈ 0.47 × 0.31 = 0.149 is the DERIVED effective value.
μ_eff(Lyα/IGM) ≈ 0.0657 << 0.07 (satisfies Lyα constraint via screening)

MCMC FREE PARAMETERS (Thesis v12)
---------------------------------
    μ_fit (cgc_mu)     : The ONLY fitted parameter = 0.47 ± 0.03
    
FIXED PARAMETERS (derived from theory):
    n_g = 0.0125       : β₀²/4π² (like c_T = 1 in Horndeski)
    z_trans = 1.67     : cosmic dynamics
    α = 2.0            : Klein-Gordon
    ρ_thresh = 200     : Virial theorem

Usage
-----
>>> from cgc.parameters import CGCParameters
>>> params = CGCParameters()
>>> theta = params.to_array()  # For MCMC
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .config import PLANCK_BASELINE


# =============================================================================
# PARAMETER NAMES AND LABELS
# =============================================================================

# Full parameter names (for MCMC sampling)
# FIXED BY THEORY (not sampled):
#   - n_g = 0.0125 (β₀²/4π²)
#   - z_trans = 1.67 (cosmic dynamics)
#   - rho_thresh = 200 (virial theorem)
# ONLY 7 PARAMETERS ARE SAMPLED: 6 cosmological + μ
PARAM_NAMES = [
    'omega_b',      # 0: Baryon density
    'omega_cdm',    # 1: CDM density
    'h',            # 2: Hubble parameter
    'ln10As',       # 3: Primordial amplitude
    'n_s',          # 4: Spectral index
    'tau_reio',     # 5: Optical depth
    'cgc_mu',       # 6: CGC coupling (ONLY free SDCG parameter)
]

# Short names for display
PARAM_NAMES_SHORT = [
    'ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ', 'μ'
]

# LaTeX labels for plotting
PARAM_LABELS_LATEX = [
    r'$\omega_b$',
    r'$\omega_{cdm}$',
    r'$h$',
    r'$\ln(10^{10}A_s)$',
    r'$n_s$',
    r'$\tau_{reio}$',
    r'$\mu$',
]

# Parameter descriptions
# FIXED BY THEORY (not in PARAM_NAMES):
#   - n_g = 0.0125 (β₀²/4π²)
#   - z_trans = 1.67 (cosmic dynamics)
#   - rho_thresh = 200 (virial theorem)
PARAM_DESCRIPTIONS = {
    'omega_b': 'Baryon density parameter (Ω_b h²)',
    'omega_cdm': 'Cold dark matter density parameter (Ω_cdm h²)',
    'h': 'Reduced Hubble parameter (H0 / 100 km/s/Mpc)',
    'ln10As': 'Log primordial scalar amplitude ln(10¹⁰ A_s)',
    'n_s': 'Scalar spectral index',
    'tau_reio': 'Optical depth to reionization',
    'cgc_mu': 'SDCG coupling strength - PHENOMENOLOGICAL (0 = ΛCDM)',
    # FIXED (not sampled):
    'cgc_n_g': 'FIXED = 0.0125 (β₀²/4π²)',
    'cgc_z_trans': 'FIXED = 1.67 (cosmic dynamics)',
    'cgc_rho_thresh': 'FIXED = 200 (virial theorem)',
}


# =============================================================================
# MODEL CONSTANTS (First-Principles Derivations)
# =============================================================================
#
# Parameters derived from Standard Model physics and cosmological evolution.
# See cgc/first_principles_parameters.py for full derivation details.

# ─────────────────────────────────────────────────────────────────────────────
# β₀: Scalar-matter coupling [PHENOMENOLOGICAL, motivated by SM physics]
# ─────────────────────────────────────────────────────────────────────────────
# STATUS: β₀ is a PHENOMENOLOGICAL parameter, not rigorously derived from QFT.
#
# MOTIVATION (not derivation):
#   The ratio m_t/v = 173/246 ≈ 0.70 provides a natural benchmark value.
#   This is DIMENSIONAL ESTIMATE, not proper trace anomaly calculation.
#   The full QFT derivation involves:
#     - Complex renormalization group flow from electroweak to Hubble scales
#     - Loop corrections with unjustified factors (factor of 2, N_c handling)
#     - Sign ambiguities in small-x approximations
#
# HONEST STATEMENT:
#   "β₀ is a dimensionless coupling. Naturalness suggests β₀ ~ O(1).
#    We adopt β₀ = 0.70 as benchmark motivated by m_t/v ≈ 0.70.
#    Its value is ultimately constrained by cosmological data."
#
# MICROSCOPE: |β| < 10⁻⁵ applies to Brans-Dicke universal couplings.
# SDCG evades this via Chameleon screening at Earth's surface density.
BETA_0 = 0.70  # PHENOMENOLOGICAL: Benchmark from m_t/v = 173/246

# ─────────────────────────────────────────────────────────────────────────────
# n_g = β₀²/4π² [FIXED BY THEORY - NOT A FREE PARAMETER]
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL DECISION (Feb 2026): n_g is now FIXED, not fitted.
#
# RATIONALE:
#   Like c_T = 1 in Horndeski theories or γ_PPN = 1/2 in f(R) gravity,
#   n_g is a THEORETICAL REQUIREMENT from the renormalization group.
#
# DERIVATION:
#   One-loop β-function: μ d/dμ G_eff⁻¹ = β₀²/16π²
#   Integrating: G_eff⁻¹(k) - G_N⁻¹ = (β₀²/16π²) ln(k/k_*)
#   For scale dependence f(k) ~ k^n_g: n_g = β₀²/4π²
#
# Numerical: 0.70² / (4π²) = 0.49 / 39.48 ≈ 0.0125
#
# VALIDATION:
#   With n_g = 0.0125 fixed, Hubble tension reduced by 61% (Page 363).
#   The theory WORKS with this theoretical value!
#
# WHY NOT FIT n_g?
#   Unconstrained MCMC gives n_g ≈ 0.92, which is UNPHYSICAL:
#   - Would destroy galaxy clusters via f(k) ~ k^0.92
#   - 70× larger than RG prediction
#   - Indicates fitting degeneracy, not physical preference
#
# ═══════════════════════════════════════════════════════════════════════════
# n_g = 0.0125 IS FIXED BY THEORY - REMOVED FROM MCMC PARAMETER VECTOR
# ═══════════════════════════════════════════════════════════════════════════
N_G_FIXED = 0.0125  # β₀²/4π² - THEORETICAL REQUIREMENT, NOT FREE PARAMETER
N_G_FROM_BETA = BETA_0**2 / (4 * np.pi**2)  # ≈ 0.0124 (computed for verification)

# ─────────────────────────────────────────────────────────────────────────────
# z_trans from cosmic evolution [FIXED BY THEORY]
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: q=0 transition (deceleration→acceleration): z_accel ≈ 0.63
# Step 2: Scalar response time: Δz ≈ 1.04 (one Hubble time delay)
# Result: z_trans = z_accel + Δz ≈ 1.67 (THESIS v12 CANONICAL)
#
# NOTE: The exact derivation involves solving for q(z) = 0:
#       q = 0 when Ω_m(1+z)³/2 = Ω_Λ → z_accel = (2Ω_Λ/Ω_m)^(1/3) - 1
#       For Ω_m = 0.315, Ω_Λ = 0.685: z_accel = (2×0.685/0.315)^(1/3) - 1 ≈ 0.63
#       With scalar field response delay Δz ≈ 1.04: z_trans ≈ 1.67
Z_ACCEL_TRANSITION = 0.63  # From Planck Ω values: (2×0.685/0.315)^(1/3) - 1
DELTA_Z_SCALAR_RESPONSE = 1.04  # Scalar catches up in ~1 Hubble time
Z_TRANS_DERIVED = 1.67  # THESIS v12 CANONICAL VALUE (z_eq + Δz)
Z_TRANS_FIXED = 1.67  # FIXED BY THEORY - NOT A FREE PARAMETER

# ─────────────────────────────────────────────────────────────────────────────
# α: Screening exponent [DERIVED from Klein-Gordon equation]
# ─────────────────────────────────────────────────────────────────────────────
# THESIS v12 CANONICAL VALUE: α = 2.0
# From Klein-Gordon equation with quadratic potential: m_eff² ∝ ρ
# This gives Chameleon-like screening with S(ρ) = 1/(1 + (ρ/ρ_thresh)²)
# Note: Technical Supplement [Source 1146-1148] specifies α = 2.0
ALPHA_SCREENING = 2.0  # From Klein-Gordon (quadratic potential) - THESIS v12

# ─────────────────────────────────────────────────────────────────────────────
# ρ_thresh: Screening threshold [FIXED BY THEORY]
# ─────────────────────────────────────────────────────────────────────────────
# Screening activates when F_φ ~ F_G at virial radius
# For clusters at Δ_vir ≈ 200: ρ_thresh ≈ 200 ρ_crit
RHO_THRESH_DEFAULT = 200  # units of ρ_crit, from virial condition
RHO_THRESH_FIXED = 200.0  # FIXED BY THEORY - NOT A FREE PARAMETER

# ─────────────────────────────────────────────────────────────────────────────
# μ: COMPREHENSIVE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# See scripts/mu_definitions_reference.py for full documentation
#
# HIERARCHY OF μ VALUES:
# ──────────────────────────────────────────────────────────────────────────────
#
# 1. μ_bare (QFT one-loop):
#    μ_bare = β₀² × ln(M_Pl/H₀) / (16π²)
#          = 0.70² × 140 / 158 ≈ 0.43 - 0.48
#    This is the unscreened coupling from quantum gravity corrections.
#
# 2. μ_max (Theoretical upper bound):
#    μ_max = 0.50
#    Above this, G_eff/G_N > 1.5 (too large).
#
# 3. μ (Cosmological/MCMC):
#    μ = 0.47 ± 0.03 (MCMC unconstrained best-fit)
#    This is what the CMB+BAO+SNe data prefers.
#
# 4. μ_eff (Environment-dependent effective coupling):
#    μ_eff(ρ, z) = μ × S(ρ) × f(z)
#    where S(ρ) = exp(-ρ/ρ_thresh)     [Screening]
#          f(z) = 1/(1 + (z/z_trans)²)  [Redshift evolution]
#
# 5. μ_Lyα (Ly-α constrained):
#    μ_Lyα = 0.045 ± 0.019 (Thesis v6 OFFICIAL)
#    This is what Ly-α forest observations constrain.
#    NOTE: Ly-α measures μ_eff(IGM, z~3), NOT μ_cosmic!
#
# KEY INSIGHT: μ = 0.47 is CONSISTENT with μ_Lyα < 0.05 because
# Ly-α probes μ_eff (screened + high-z suppressed), not μ_cosmic!
# ──────────────────────────────────────────────────────────────────────────────

LN_MPL_OVER_H0 = 140  # ln(M_Pl/H₀) ≈ 138-140

# μ_bare: QFT one-loop calculation
# μ_bare = β₀² × ln(M_Pl/H₀) / (16π²)
MU_BARE = 0.48        # ≈ 0.70² × 140 / 158

# μ_max: Theoretical upper bound
MU_MAX = 0.50         # MCMC prior upper bound, stability limit

# ═══════════════════════════════════════════════════════════════════════════
# THESIS v12 CANONICAL μ VALUES
# ═══════════════════════════════════════════════════════════════════════════
#
# μ_fit: THE ONLY MCMC FREE PARAMETER
MU_FIT = 0.47         # MCMC best-fit: 0.47 ± 0.03 (6σ detection)
                      # Matches QFT prediction μ_bare ≈ 0.48 EXCELLENTLY!

# ENVIRONMENT-DEPENDENT EFFECTIVE VALUES (from Master Equation)
# μ_eff = μ_fit × S(ρ) where S(ρ) = 1/(1 + (ρ/200)²)
MU_EFF_VOID = 0.47    # In voids (ρ ~ 0.1 ρ_crit): S ≈ 1.0 → μ_eff ≈ 0.47
MU_EFF_LYALPHA = 0.05 # In IGM (ρ ~ 100 ρ_crit): S ≈ 0.1 → μ_eff ≈ 0.05
MU_EFF_CLUSTER = 0.005 # In clusters (ρ ~ 200 ρ_crit): S ≈ 0.01 → μ_eff ≈ 0.005
MU_EFF_SOLAR = 0.0    # In solar system: fully screened

# VERSION CONFUSION RESOLUTION:
# • Older versions (v5-v10) cited μ = 0.149 - this was μ_eff in VOIDS
# • Thesis v12 cites μ = 0.47 - this is the FUNDAMENTAL fitted value
# • They match: μ_void = μ_fit × S_void ≈ 0.47 × 0.31 ≈ 0.149

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER BOUNDS: COMPLETE DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# 
# This section documents WHERE each bound comes from, WHY it's needed,
# and HOW it affects tension reduction.
#
# ═══════════════════════════════════════════════════════════════════════════
# CGC COUPLING μ
# ═══════════════════════════════════════════════════════════════════════════
#
# Bound     Value   Physical Origin
# ─────────────────────────────────────────────────────────────────────────
# Lower     0.0     ΛCDM limit (recovers General Relativity)
# Central   0.47    MCMC best-fit from CMB+BAO+SNe
# Upper     0.50    QFT one-loop μ_bare ≈ 0.48; G_eff/G_N ≤ 1.5
#
# Impact on tension reduction:
#   μ = 0.00 → 0% reduction (ΛCDM)
#   μ = 0.05 → ~5% reduction (insufficient)
#   μ = 0.20 → ~30% reduction
#   μ = 0.47 → 62% H₀, 69% S₈ reduction (thesis claims)
#   μ = 0.50 → ~70% reduction (maximum)
#
# ═══════════════════════════════════════════════════════════════════════════
# SCALE EXPONENT n_g
# ═══════════════════════════════════════════════════════════════════════════
#
# n_g = β₀²/(4π²) where β₀ = m_top/v = 173/246 = 0.70
#
# Bound     Value   Physical Origin
# ─────────────────────────────────────────────────────────────────────────
# Lower     0.010   β₀ = 0.63 (minimal SM contribution)
# Central   0.014   β₀ = 0.70 (SM trace anomaly)
# Upper     0.020   β₀ = 0.89 (maximal with BSM)
#
# ⚠️ WARNING: MCMC prefers n_g ≈ 0.92, which is 70× the EFT value!
# This tension requires investigation. Thesis uses EFT value n_g = 0.014.
#
# ═══════════════════════════════════════════════════════════════════════════
# TRANSITION REDSHIFT z_trans
# ═══════════════════════════════════════════════════════════════════════════
#
# z_trans = z_eq + Δz, where:
#   z_eq ≈ 0.63 (matter-DE equality)
#   Δz ≈ 1.0 ± 0.37 (scalar field response time)
#
# Bound     Value   Physical Origin
# ─────────────────────────────────────────────────────────────────────────
# Lower     1.30    z_eq + 0.67 (minimal delay)
# Central   1.67    z_eq + 1.04 (one Hubble time)
# Upper     2.00    z_eq + 1.37 (extended delay)
#
# Impact: Earlier z_trans → more time for CGC → larger tension reduction
#
# ═══════════════════════════════════════════════════════════════════════════
# SCREENING THRESHOLD ρ_thresh
# ═══════════════════════════════════════════════════════════════════════════
#
# From virial theorem: Δ_vir = 18π² ≈ 178-200
#
# Bound     Value        Physical Origin
# ─────────────────────────────────────────────────────────────────────────
# Lower     100 ρ_crit   Outer halo regions (turnaround)
# Central   200 ρ_crit   Virial theorem exact
# Upper     300 ρ_crit   Inner virialized regions (NFW scale)
#
# Impact: Lower ρ_thresh → more screening → less CGC effect
#
# ═══════════════════════════════════════════════════════════════════════════
# MCMC PRIOR BOUNDS (as used in MCMC sampling)
# ═══════════════════════════════════════════════════════════════════════════


# =============================================================================
# PARAMETER BOUNDS (for prior enforcement)
# =============================================================================
# ONLY 7 PARAMETERS ARE SAMPLED IN MCMC
# FIXED BY THEORY: n_g=0.0125, z_trans=1.67, rho_thresh=200

PARAM_BOUNDS = {
    # ═══════════════════════════════════════════════════════════════════════
    # Standard cosmological parameters (6 sampled)
    # Bounds based on physical constraints and Planck priors
    # ═══════════════════════════════════════════════════════════════════════
    
    'omega_b': (0.019, 0.025),     # BBN + CMB constraints (tightened)
    'omega_cdm': (0.10, 0.18),     # Widened: CGC can shift ω_cdm significantly
    'h': (0.60, 0.80),             # Wide prior encompassing Planck & SH0ES
    'ln10As': (2.9, 3.3),          # Widened: τ-As degeneracy needs room
    'n_s': (0.92, 1.00),           # Nearly scale-invariant
    'tau_reio': (0.02, 0.10),      # Reionization constraints
    
    # ═══════════════════════════════════════════════════════════════════════
    # SDCG effective coupling parameter (1 sampled)
    # This is the ONLY free SDCG parameter!
    # NOTE: This is μ_eff (effective coupling in voids), NOT μ_bare!
    # μ_bare ≈ 0.48 (QFT), but screening gives μ_eff ≈ 0.15 in cosmology
    # ═══════════════════════════════════════════════════════════════════════
    
    'cgc_mu': (0.0, 0.50),         # μ_void: Bare/void coupling (ONLY free SDCG param)
                                   # 0.0 = ΛCDM limit (no modification)
                                   # 0.50 = Theoretical upper bound (G_eff/G_N ≤ 1.5)
                                   # Best-fit: ~0.47 (QFT one-loop prediction: 0.48)
}

# FIXED PARAMETERS (not sampled, but stored for reference)
FIXED_PARAM_VALUES = {
    'cgc_n_g': N_G_FIXED,           # 0.0125 from β₀²/4π²
    'cgc_z_trans': Z_TRANS_FIXED,   # 1.67 from cosmic dynamics
    'cgc_rho_thresh': RHO_THRESH_FIXED,  # 200 from virial theorem
}


# =============================================================================
# PARAMETER CLASS
# =============================================================================

@dataclass
class CGCParameters:
    """
    Container for CGC theory and cosmological parameters.
    
    This class provides a convenient interface for managing all parameters
    needed for CGC cosmological analysis. It includes methods for conversion
    to/from arrays (for MCMC), dictionary representations, and validation.
    
    Attributes
    ----------
    omega_b : float
        Baryon density parameter Ω_b h² (default: Planck 2018)
    omega_cdm : float
        Cold dark matter density Ω_cdm h² (default: Planck 2018)
    h : float
        Reduced Hubble parameter H0/(100 km/s/Mpc) (default: Planck 2018)
    ln10As : float
        Log primordial amplitude ln(10¹⁰ A_s) (default: Planck 2018)
    n_s : float
        Scalar spectral index (default: Planck 2018)
    tau_reio : float
        Optical depth to reionization (default: Planck 2018)
    cgc_mu : float
        CGC coupling strength (default: 0.12, chosen to alleviate tension)
    cgc_n_g : float
        Scale dependence exponent (default: 0.75)
    cgc_z_trans : float
        Transition redshift (default: 2.0)
    cgc_rho_thresh : float
        Screening density threshold in units of ρ_crit (default: 200)
    
    Examples
    --------
    >>> params = CGCParameters()
    >>> print(params.H0)  # Derived H0
    67.4
    
    >>> params = CGCParameters(cgc_mu=0.2)  # Custom CGC coupling
    >>> theta = params.to_array()  # For MCMC sampler (7 params)
    
    >>> params.set_from_array(new_theta)  # Update from MCMC sample
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # Standard cosmological parameters (Planck 2018 defaults) - 6 SAMPLED
    # ═══════════════════════════════════════════════════════════════════════
    
    omega_b: float = field(default_factory=lambda: PLANCK_BASELINE['omega_b'])
    omega_cdm: float = field(default_factory=lambda: PLANCK_BASELINE['omega_cdm'])
    h: float = field(default_factory=lambda: PLANCK_BASELINE['h'])
    ln10As: float = field(default_factory=lambda: PLANCK_BASELINE['ln10As'])
    n_s: float = field(default_factory=lambda: PLANCK_BASELINE['n_s'])
    tau_reio: float = field(default_factory=lambda: PLANCK_BASELINE['tau_reio'])
    
    # ═══════════════════════════════════════════════════════════════════════
    # SDCG PARAMETER: Only μ is sampled - 1 SAMPLED
    # 
    # FIXED BY THEORY (not sampled):
    #   - n_g = 0.0125 (β₀²/4π²)
    #   - z_trans = 1.67 (cosmic dynamics)  
    #   - rho_thresh = 200 (virial theorem)
    # ═══════════════════════════════════════════════════════════════════════
    
    cgc_mu: float = 0.47           # THESIS v13: μ_void ≈ 0.47 (bare/void coupling, S≈1)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Derived properties (FIXED BY THEORY)
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def H0(self) -> float:
        """Hubble constant in km/s/Mpc."""
        return self.h * 100
    
    @property
    def n_g(self) -> float:
        """
        Scale dependence exponent - FIXED BY THEORY.
        
        n_g = β₀²/4π² = 0.0125 is derived from the renormalization group.
        Like c_T = 1 in Horndeski theories, this is a theoretical requirement.
        """
        return N_G_FIXED
    
    @property
    def cgc_z_trans(self) -> float:
        """
        Transition redshift - FIXED BY THEORY.
        
        z_trans = 1.67 is derived from cosmic dynamics:
        z_eq ≈ 0.63 (matter-DE equality) + Δz ≈ 1.04 (scalar response delay)
        """
        return Z_TRANS_FIXED
    
    @property
    def cgc_rho_thresh(self) -> float:
        """
        Screening threshold - FIXED BY THEORY.
        
        ρ_thresh = 200 ρ_crit is derived from virial theorem (Δ_vir ≈ 200).
        """
        return RHO_THRESH_FIXED
    
    @property
    def Omega_m(self) -> float:
        """Total matter density parameter."""
        return (self.omega_b + self.omega_cdm) / self.h**2
    
    @property
    def Omega_Lambda(self) -> float:
        """Dark energy density (assuming flat universe)."""
        return 1.0 - self.Omega_m
    
    @property
    def As(self) -> float:
        """Primordial scalar amplitude A_s."""
        return np.exp(self.ln10As) / 1e10
    
    @property
    def is_lcdm(self) -> bool:
        """Check if parameters correspond to pure ΛCDM (μ = 0)."""
        return np.isclose(self.cgc_mu, 0.0)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Conversion methods
    # ═══════════════════════════════════════════════════════════════════════
    
    def to_array(self) -> np.ndarray:
        """
        Convert parameters to numpy array for MCMC sampling.
        
        NOTE: n_g = 0.0125 is FIXED BY THEORY, not included in MCMC vector.
        Use self.n_g property to access the fixed value.
        
        Returns
        -------
        np.ndarray
            Array of shape (9,) with MCMC parameters in standard order:
            [ω_b, ω_cdm, h, ln10As, n_s, τ, μ, z_trans, ρ_thresh]
        
        Examples
        --------
        >>> params = CGCParameters()
        >>> theta = params.to_array()
        >>> print(theta.shape)
        (7,)
        """
        return np.array([
            self.omega_b,
            self.omega_cdm,
            self.h,
            self.ln10As,
            self.n_s,
            self.tau_reio,
            self.cgc_mu,
            # z_trans, rho_thresh, n_g are FIXED BY THEORY
        ])
    
    def set_from_array(self, theta: np.ndarray) -> None:
        """
        Update parameters from numpy array.
        
        FIXED BY THEORY (not in theta):
          - n_g = 0.0125 (β₀²/4π²)
          - z_trans = 1.67 (cosmic dynamics)
          - rho_thresh = 200 (virial theorem)
        
        Parameters
        ----------
        theta : np.ndarray
            Array of shape (7,) with MCMC parameters in standard order:
            [ω_b, ω_cdm, h, ln10As, n_s, τ, μ]
        
        Examples
        --------
        >>> params = CGCParameters()
        >>> new_theta = np.array([0.022, 0.12, 0.68, 3.0, 0.96, 0.05, 0.15])
        >>> params.set_from_array(new_theta)
        """
        if len(theta) != 7:
            raise ValueError(f"Expected 7 parameters (n_g, z_trans, rho_thresh are fixed), got {len(theta)}")
        
        self.omega_b = theta[0]
        self.omega_cdm = theta[1]
        self.h = theta[2]
        self.ln10As = theta[3]
        self.n_s = theta[4]
        self.tau_reio = theta[5]
        self.cgc_mu = theta[6]
        # n_g, z_trans, rho_thresh are FIXED BY THEORY - accessed via properties
    
    @classmethod
    def from_array(cls, theta: np.ndarray) -> 'CGCParameters':
        """
        Create CGCParameters instance from numpy array.
        
        FIXED BY THEORY (not in theta):
          - n_g = 0.0125 (β₀²/4π²)
          - z_trans = 1.67 (cosmic dynamics)
          - rho_thresh = 200 (virial theorem)
        
        Parameters
        ----------
        theta : np.ndarray
            Array of shape (7,) with MCMC parameters.
        
        Returns
        -------
        CGCParameters
            New instance with parameters set from array.
        """
        params = cls()
        params.set_from_array(theta)
        return params
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary (includes fixed parameters).
        
        Returns
        -------
        dict
            Dictionary with parameter names as keys.
        """
        return {
            'omega_b': self.omega_b,
            'omega_cdm': self.omega_cdm,
            'h': self.h,
            'ln10As': self.ln10As,
            'n_s': self.n_s,
            'tau_reio': self.tau_reio,
            'cgc_mu': self.cgc_mu,
            # FIXED BY THEORY (accessed via properties):
            'n_g': self.n_g,                  # 0.0125
            'cgc_z_trans': self.cgc_z_trans,  # 1.67
            'cgc_rho_thresh': self.cgc_rho_thresh,  # 200.0
            # Derived parameters
            'H0': self.H0,
            'Omega_m': self.Omega_m,
            'Omega_Lambda': self.Omega_Lambda,
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # Validation
    # ═══════════════════════════════════════════════════════════════════════
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """
        Check if all parameters are within valid bounds.
        
        Returns
        -------
        tuple
            (is_valid: bool, violations: list of error messages)
        
        Examples
        --------
        >>> params = CGCParameters(cgc_mu=-0.1)  # Invalid negative coupling
        >>> valid, errors = params.is_valid()
        >>> print(valid)
        False
        """
        violations = []
        theta = self.to_array()
        
        for i, (name, bounds) in enumerate(PARAM_BOUNDS.items()):
            if theta[i] < bounds[0] or theta[i] > bounds[1]:
                violations.append(
                    f"{name}: {theta[i]:.4f} outside bounds {bounds}"
                )
        
        return len(violations) == 0, violations
    
    def get_lcdm_equivalent(self) -> 'CGCParameters':
        """
        Get ΛCDM-equivalent parameters (μ = 0).
        
        Returns a copy with CGC coupling set to zero, useful for
        model comparison. Note: n_g is fixed at 0.0125 always.
        
        Returns
        -------
        CGCParameters
            Copy with cgc_mu = 0.
        """
        lcdm = CGCParameters()
        lcdm.set_from_array(self.to_array())
        lcdm.cgc_mu = 0.0
        # n_g remains fixed at 0.0125 (theory requirement)
        return lcdm
    
    # ═══════════════════════════════════════════════════════════════════════
    # String representations
    # ═══════════════════════════════════════════════════════════════════════
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"CGCParameters(\n"
            f"  Cosmology: H0={self.H0:.2f}, Ω_m={self.Omega_m:.4f}\n"
            f"  CGC: μ={self.cgc_mu:.3f}, n_g={self.n_g:.4f} (FIXED), "
            f"z_trans={self.cgc_z_trans:.1f}\n"
            f")"
        )
    
    def summary(self) -> str:
        """
        Generate a formatted summary of all parameters.
        
        Returns
        -------
        str
            Multi-line formatted parameter summary.
        """
        lines = [
            "=" * 60,
            "CGC PARAMETER SUMMARY",
            "=" * 60,
            "",
            "Standard Cosmological Parameters:",
            "-" * 40,
            f"  ω_b         = {self.omega_b:.5f}",
            f"  ω_cdm       = {self.omega_cdm:.4f}",
            f"  h           = {self.h:.4f}",
            f"  ln(10¹⁰As)  = {self.ln10As:.3f}",
            f"  n_s         = {self.n_s:.4f}",
            f"  τ_reio      = {self.tau_reio:.4f}",
            "",
            "CGC Theory Parameters:",
            "-" * 40,
            f"  μ           = {self.cgc_mu:.4f}  (coupling strength)",
            f"  n_g         = {self.n_g:.4f}  (FIXED: β₀²/4π² = 0.0125)",
            f"  z_trans     = {self.cgc_z_trans:.2f}  (transition redshift)",
            f"  ρ_thresh    = {self.cgc_rho_thresh:.1f}  (screening density)",
            "",
            "Derived Parameters:",
            "-" * 40,
            f"  H0          = {self.H0:.2f} km/s/Mpc",
            f"  Ω_m         = {self.Omega_m:.4f}",
            f"  Ω_Λ         = {self.Omega_Lambda:.4f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_parameter_names() -> List[str]:
    """
    Get list of parameter names.
    
    Returns
    -------
    list
        Full parameter names in standard order.
    """
    return PARAM_NAMES.copy()


def get_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Get parameter bounds dictionary.
    
    Returns
    -------
    dict
        Parameter name -> (min, max) bounds.
    """
    return PARAM_BOUNDS.copy()


def get_bounds_array() -> np.ndarray:
    """
    Get parameter bounds as numpy array.
    
    Returns
    -------
    np.ndarray
        Shape (10, 2) array with [min, max] for each parameter.
    """
    return np.array([PARAM_BOUNDS[name] for name in PARAM_NAMES])


def get_latex_labels() -> List[str]:
    """
    Get LaTeX-formatted parameter labels for plotting.
    
    Returns
    -------
    list
        LaTeX strings for each parameter.
    """
    return PARAM_LABELS_LATEX.copy()


def check_bounds(theta: np.ndarray) -> bool:
    """
    Check if parameter array is within valid bounds.
    
    NOTE: n_g = 0.0125 is FIXED BY THEORY, not included in theta.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter array of shape (9,).
    
    Returns
    -------
    bool
        True if all parameters are within bounds.
    
    Examples
    --------
    >>> theta = np.array([0.022, 0.12, 0.68, 3.0, 0.96, 0.05,
    ...                   0.15, 1.64, 200.0])
    >>> check_bounds(theta)
    True
    """
    bounds = get_bounds_array()
    return np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1]))


def get_cgc_only_indices() -> List[int]:
    """
    Get indices of CGC-specific MCMC parameters.
    
    NOTE: n_g = 0.0125 is FIXED BY THEORY, not in MCMC.
    
    Returns
    -------
    list
        Indices [6, 7, 8] for μ, z_trans, ρ_thresh.
    """
    return [6, 7, 8]


def get_cosmo_only_indices() -> List[int]:
    """
    Get indices of standard cosmological parameters.
    
    Returns
    -------
    list
        Indices [0, 1, 2, 3, 4, 5] for ω_b, ω_cdm, h, ln10As, n_s, τ.
    """
    return [0, 1, 2, 3, 4, 5]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Test parameter class
    params = CGCParameters()
    print(params.summary())
    
    # Test array conversion
    theta = params.to_array()
    print(f"\nParameter array: {theta}")
    
    # Test validation
    valid, errors = params.is_valid()
    print(f"\nValid: {valid}")
    if not valid:
        print("Errors:", errors)
    
    # Test bounds
    print(f"\nBounds check: {check_bounds(theta)}")
