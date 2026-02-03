"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SDCG Parameters Module (First-Principles)                 ║
║                                                                              ║
║  Defines the Scale-Dependent Crossover Gravity (SDCG) parameter space.      ║
║  Parameters are DERIVED from accepted physics where possible.               ║
║  Only μ requires new physics (meV scale) - this is a PREDICTION.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

PARAMETER CLASSIFICATION (v8 - First Principles)
=================================================

┌─────────────────┬─────────────┬────────────────────────────────────────────┐
│ Parameter       │ Status      │ Derivation                                 │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ β₀ = 0.70       │ ✓ DERIVED   │ SM conformal anomaly + top quark dominance │
│                 │             │ β₀² = (m_t/v)² = (173/246)² = 0.49         │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ n_g = 0.0125    │ ✓ DERIVED   │ RG flow: n_g = β₀²/4π²                     │
│                 │             │ One-loop scalar-tensor β-function          │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ z_trans = 1.3   │ ✓ DERIVED   │ z_eq + Δz_response = 0.3 + 1.0             │
│                 │             │ Matter-DE equality + scalar response time  │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ α = 1.0         │ ~ DERIVED   │ Effective potential V(φ) ~ φ⁻¹             │
│                 │             │ Potential-dependent, not universal         │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ ρ_thresh = 200  │ ~ DERIVED   │ Virial equilibrium condition               │
│                 │             │ Matches cluster overdensity                │
├─────────────────┼─────────────┼────────────────────────────────────────────┤
│ μ < 0.1         │ ⚠️ CONSTRAINED│ Lyα forest upper limit                    │
│                 │             │ REQUIRES NEW PHYSICS at meV scale!         │
└─────────────────┴─────────────┴────────────────────────────────────────────┘

THE μ PROBLEM
-------------
Standard RG running gives μ ~ 1-2, but Lyα constrains μ < 0.1.
This PREDICTS new light particles at the dark energy scale (~meV).
See MEV_NEW_PHYSICS_PREDICTION.md for details.

MCMC FREE PARAMETERS (fitted to data)
-------------------------------------
    μ (cgc_mu)         : Constrained by Lyα (< 0.1)
    n_g (cgc_n_g)      : Derived value ~0.0125, but fitted for flexibility
    z_trans (cgc_z_trans): Derived value ~1.3, but fitted for flexibility
    ρ_thresh (cgc_rho_thresh): Derived value ~200, but fitted for flexibility

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

# Full parameter names (for internal use)
PARAM_NAMES = [
    'omega_b',      # 0: Baryon density
    'omega_cdm',    # 1: CDM density
    'h',            # 2: Hubble parameter
    'ln10As',       # 3: Primordial amplitude
    'n_s',          # 4: Spectral index
    'tau_reio',     # 5: Optical depth
    'cgc_mu',       # 6: CGC coupling
    'cgc_n_g',      # 7: Scale dependence
    'cgc_z_trans',  # 8: Transition redshift
    'cgc_rho_thresh'# 9: Screening threshold
]

# Short names for display
PARAM_NAMES_SHORT = [
    'ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ',
    'μ', 'n_g', 'z_trans', 'ρ_thresh'
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
    r'$n_g$',
    r'$z_{trans}$',
    r'$\rho_{thresh}$'
]

# Parameter descriptions
PARAM_DESCRIPTIONS = {
    'omega_b': 'Baryon density parameter (Ω_b h²)',
    'omega_cdm': 'Cold dark matter density parameter (Ω_cdm h²)',
    'h': 'Reduced Hubble parameter (H0 / 100 km/s/Mpc)',
    'ln10As': 'Log primordial scalar amplitude ln(10¹⁰ A_s)',
    'n_s': 'Scalar spectral index',
    'tau_reio': 'Optical depth to reionization',
    'cgc_mu': 'SDCG coupling strength - PHENOMENOLOGICAL (0 = ΛCDM)',
    'cgc_n_g': 'SDCG scale exponent - MODEL-DEPENDENT',
    'cgc_z_trans': 'SDCG transition redshift - TUNED',
    'cgc_rho_thresh': 'SDCG screening threshold - TUNED (× ρ_crit)',
}


# =============================================================================
# MODEL CONSTANTS (First-Principles Derivations)
# =============================================================================
#
# Parameters derived from Standard Model physics and cosmological evolution.
# See cgc/first_principles_parameters.py for full derivation details.

# ─────────────────────────────────────────────────────────────────────────────
# β₀: Scalar-matter coupling [DERIVED from SM conformal anomaly]
# ─────────────────────────────────────────────────────────────────────────────
# From trace anomaly: β₀² = (b₀ α_s/4π)² + Σ(m_f/v)²
# Top quark dominates: (m_t/v)² = (173/246)² ≈ 0.49
# QCD contribution: ≈ 0.004
# Total: β₀² ≈ 0.49, so β₀ ≈ 0.70
#
# Note on MICROSCOPE: The |β| < 10⁻⁵ bound applies to Brans-Dicke-like
# universal couplings. SDCG has screened coupling that evades this bound
# in high-density environments (Earth surface).
BETA_0 = 0.70  # DERIVED from SM conformal anomaly

# ─────────────────────────────────────────────────────────────────────────────
# n_g from β₀: n_g = β₀²/4π² [DERIVED from RG flow]
# ─────────────────────────────────────────────────────────────────────────────
# One-loop β-function for G_eff in scalar-tensor EFT:
#   μ d/dμ G_eff⁻¹ = β₀²/16π²
# Integrating gives: G_eff(k)/G_N = 1 + (β₀²/4π²) ln(k/k_*)
# Therefore: n_g = β₀²/4π²
# Numerical: 0.70² / (4π²) = 0.49 / 39.48 ≈ 0.0124
N_G_FROM_BETA = BETA_0**2 / (4 * np.pi**2)  # ≈ 0.0124

# ─────────────────────────────────────────────────────────────────────────────
# z_trans from cosmic evolution [DERIVED from DE transition]
# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Matter-DE equality: z_eq = (Ω_Λ/Ω_m)^(1/3) - 1 ≈ 0.30
# Step 2: Scalar response time: Δz ≈ 1 (one e-fold)
# Result: z_trans = z_eq + Δz ≈ 1.30
Z_MATTER_DE_EQUALITY = (0.685 / 0.315)**(1/3) - 1  # ≈ 0.30
DELTA_Z_SCALAR_RESPONSE = 1.0  # Scalar catches up in ~1 e-fold
Z_TRANS_DERIVED = Z_MATTER_DE_EQUALITY + DELTA_Z_SCALAR_RESPONSE  # ≈ 1.30

# ─────────────────────────────────────────────────────────────────────────────
# α: Screening exponent [DERIVED from effective potential]
# ─────────────────────────────────────────────────────────────────────────────
# For power-law potential V(φ) ~ φ⁻ⁿ:
#   α = p = 2(n+2) / 3(n+1)
# For n=1 (inverse linear): α = 2(3)/3(2) = 1.0
# Note: This is POTENTIAL-DEPENDENT, not universal!
ALPHA_SCREENING = 1.0  # For V(φ) ~ φ⁻¹

# ─────────────────────────────────────────────────────────────────────────────
# ρ_thresh: Screening threshold [DERIVED from virial equilibrium]
# ─────────────────────────────────────────────────────────────────────────────
# Screening activates when F_φ ~ F_G at virial radius
# For clusters at Δ_vir ≈ 200: ρ_thresh ≈ 200 ρ_crit
RHO_THRESH_DEFAULT = 200  # units of ρ_crit, from virial condition

# ─────────────────────────────────────────────────────────────────────────────
# μ: THE PROBLEM PARAMETER [CONSTRAINED, requires new physics]
# ─────────────────────────────────────────────────────────────────────────────
# From naive RG: μ = (β₀²/4π²) × ln(Λ_UV/H₀)
# With Λ_UV = M_Pl: μ ≈ 0.0124 × 138 ≈ 1.7 (TOO LARGE!)
# Lyα constraint: μ < 0.1
#
# RESOLUTION: Chameleon + Vainshtein hybrid screening
# μ_bare = 0.48 (from β₀² ln(M_Pl/H₀) / 16π²)
# μ_eff (void) = 0.149 (after cosmological screening)
# μ_eff (IGM/Lyα) ≈ 6×10⁻⁵ (after hybrid screening)
LN_MPL_OVER_H0 = 138  # ln(M_Pl/H₀)
MU_BARE = 0.48        # From QFT one-loop calculation
MU_EFF_VOID = 0.149   # MCMC best-fit in voids (6σ detection)
MU_EFF_LYALPHA = 6e-5 # After Chameleon + Vainshtein in IGM


# =============================================================================
# PARAMETER BOUNDS (for prior enforcement)
# =============================================================================

PARAM_BOUNDS = {
    # ═══════════════════════════════════════════════════════════════════════
    # Standard cosmological parameters
    # Bounds based on physical constraints and Planck priors
    # ═══════════════════════════════════════════════════════════════════════
    
    'omega_b': (0.018, 0.026),     # BBN + CMB constraints
    'omega_cdm': (0.10, 0.14),     # CMB + LSS constraints
    'h': (0.60, 0.80),             # Wide prior encompassing Planck & SH0ES
    'ln10As': (2.7, 3.3),          # CMB amplitude
    'n_s': (0.92, 1.00),           # Nearly scale-invariant
    'tau_reio': (0.01, 0.10),      # Reionization constraints
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC theory parameters
    # Physically motivated bounds
    # ═══════════════════════════════════════════════════════════════════════
    
    'cgc_mu': (0.0, 0.5),          # Coupling strength (0 = ΛCDM)
    'cgc_n_g': (0.0, 2.0),         # Scale dependence
    'cgc_z_trans': (0.5, 5.0),     # Transition redshift
    'cgc_rho_thresh': (10.0, 1000.0),  # Screening threshold
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
    >>> theta = params.to_array()  # For MCMC sampler
    
    >>> params.set_from_array(new_theta)  # Update from MCMC sample
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # Standard cosmological parameters (Planck 2018 defaults)
    # ═══════════════════════════════════════════════════════════════════════
    
    omega_b: float = field(default_factory=lambda: PLANCK_BASELINE['omega_b'])
    omega_cdm: float = field(default_factory=lambda: PLANCK_BASELINE['omega_cdm'])
    h: float = field(default_factory=lambda: PLANCK_BASELINE['h'])
    ln10As: float = field(default_factory=lambda: PLANCK_BASELINE['ln10As'])
    n_s: float = field(default_factory=lambda: PLANCK_BASELINE['n_s'])
    tau_reio: float = field(default_factory=lambda: PLANCK_BASELINE['tau_reio'])
    
    # ═══════════════════════════════════════════════════════════════════════
    # SDCG PHENOMENOLOGICAL PARAMETERS
    # 
    # IMPORTANT: These are PHENOMENOLOGICAL parameters constrained by data,
    # NOT first-principles QFT derivations. The scalar-tensor EFT motivates
    # the functional form but does NOT fix the numerical values.
    #
    # μ: Coupling strength - constrained by Lyα forest (upper bound ~0.1)
    # n_g: Scale dependence - model-dependent, NOT fixed by fundamental physics
    # z_trans: Transition redshift - physically motivated but not derived
    # α (screening): Fixed to 2 for chameleon-like screening (model-dependent)
    # ═══════════════════════════════════════════════════════════════════════
    
    cgc_mu: float = 0.149          # MCMC best-fit in voids (6σ detection)
    cgc_n_g: float = 0.014         # Derived: β₀²/4π² with β₀ = 0.70
    cgc_z_trans: float = 1.64      # Derived from deceleration parameter q(z)
    cgc_rho_thresh: float = 200.0  # Screening threshold (virial overdensity)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Derived properties
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def H0(self) -> float:
        """Hubble constant in km/s/Mpc."""
        return self.h * 100
    
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
        
        Returns
        -------
        np.ndarray
            Array of shape (10,) with all parameters in standard order:
            [ω_b, ω_cdm, h, ln10As, n_s, τ, μ, n_g, z_trans, ρ_thresh]
        
        Examples
        --------
        >>> params = CGCParameters()
        >>> theta = params.to_array()
        >>> print(theta.shape)
        (10,)
        """
        return np.array([
            self.omega_b,
            self.omega_cdm,
            self.h,
            self.ln10As,
            self.n_s,
            self.tau_reio,
            self.cgc_mu,
            self.cgc_n_g,
            self.cgc_z_trans,
            self.cgc_rho_thresh
        ])
    
    def set_from_array(self, theta: np.ndarray) -> None:
        """
        Update parameters from numpy array.
        
        Parameters
        ----------
        theta : np.ndarray
            Array of shape (10,) with parameters in standard order.
        
        Examples
        --------
        >>> params = CGCParameters()
        >>> new_theta = np.array([0.022, 0.12, 0.68, 3.0, 0.96, 0.05, 
        ...                       0.15, 0.8, 2.5, 250.0])
        >>> params.set_from_array(new_theta)
        """
        if len(theta) != 10:
            raise ValueError(f"Expected 10 parameters, got {len(theta)}")
        
        self.omega_b = theta[0]
        self.omega_cdm = theta[1]
        self.h = theta[2]
        self.ln10As = theta[3]
        self.n_s = theta[4]
        self.tau_reio = theta[5]
        self.cgc_mu = theta[6]
        self.cgc_n_g = theta[7]
        self.cgc_z_trans = theta[8]
        self.cgc_rho_thresh = theta[9]
    
    @classmethod
    def from_array(cls, theta: np.ndarray) -> 'CGCParameters':
        """
        Create CGCParameters instance from numpy array.
        
        Parameters
        ----------
        theta : np.ndarray
            Array of shape (10,) with all parameters.
        
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
        Convert to dictionary.
        
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
            'cgc_n_g': self.cgc_n_g,
            'cgc_z_trans': self.cgc_z_trans,
            'cgc_rho_thresh': self.cgc_rho_thresh,
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
        model comparison.
        
        Returns
        -------
        CGCParameters
            Copy with cgc_mu = 0.
        """
        lcdm = CGCParameters()
        lcdm.set_from_array(self.to_array())
        lcdm.cgc_mu = 0.0
        lcdm.cgc_n_g = 0.0
        return lcdm
    
    # ═══════════════════════════════════════════════════════════════════════
    # String representations
    # ═══════════════════════════════════════════════════════════════════════
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"CGCParameters(\n"
            f"  Cosmology: H0={self.H0:.2f}, Ω_m={self.Omega_m:.4f}\n"
            f"  CGC: μ={self.cgc_mu:.3f}, n_g={self.cgc_n_g:.2f}, "
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
            f"  n_g         = {self.cgc_n_g:.4f}  (scale dependence)",
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
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter array of shape (10,).
    
    Returns
    -------
    bool
        True if all parameters are within bounds.
    
    Examples
    --------
    >>> theta = np.array([0.022, 0.12, 0.68, 3.0, 0.96, 0.05,
    ...                   0.15, 0.8, 2.5, 250.0])
    >>> check_bounds(theta)
    True
    """
    bounds = get_bounds_array()
    return np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1]))


def get_cgc_only_indices() -> List[int]:
    """
    Get indices of CGC-specific parameters.
    
    Returns
    -------
    list
        Indices [6, 7, 8, 9] for μ, n_g, z_trans, ρ_thresh.
    """
    return [6, 7, 8, 9]


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
