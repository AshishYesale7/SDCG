"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CGC Parameters Module                                 ║
║                                                                              ║
║  Defines the Casimir-Gravity Crossover (CGC) theory parameter space,        ║
║  including cosmological parameters, CGC-specific parameters, priors,         ║
║  and parameter bounds for MCMC sampling.                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

CGC Theory Parameters
---------------------
The CGC theory introduces four new parameters beyond standard ΛCDM:

    μ (cgc_mu)         : CGC coupling strength (0 = pure ΛCDM)
                         Controls the amplitude of gravity modification
                         
    n_g (cgc_n_g)      : Scale dependence exponent
                         Determines how CGC effects vary with scale/redshift
                         
    z_trans (cgc_z_trans): Transition redshift
                           Redshift where CGC effects become significant
                           
    ρ_thresh (cgc_rho_thresh): Screening density threshold
                               Density above which CGC is screened (×ρ_crit)

Standard Cosmological Parameters
--------------------------------
    ω_b       : Baryon density (Ω_b h²)
    ω_cdm     : Cold dark matter density (Ω_cdm h²)
    h         : Hubble parameter (H0/100)
    ln(10¹⁰As): Primordial amplitude
    n_s       : Scalar spectral index
    τ_reio    : Optical depth to reionization

Usage
-----
>>> from cgc.parameters import CGCParameters
>>> params = CGCParameters()
>>> theta = params.to_array()  # For MCMC
>>> params.set_from_array(theta_new)  # Update from MCMC sample
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
    'cgc_mu': 'CGC coupling strength (0 = ΛCDM limit)',
    'cgc_n_g': 'CGC scale dependence exponent',
    'cgc_z_trans': 'CGC transition redshift',
    'cgc_rho_thresh': 'CGC screening density threshold (× ρ_crit)',
}


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
    # CGC theory parameters (fiducial values chosen to reduce tensions)
    # ═══════════════════════════════════════════════════════════════════════
    
    cgc_mu: float = 0.12           # Coupling strength
    cgc_n_g: float = 0.75          # Scale dependence
    cgc_z_trans: float = 2.0       # Transition redshift
    cgc_rho_thresh: float = 200.0  # Screening threshold
    
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
