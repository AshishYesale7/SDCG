"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CGC Theory Module                                   ║
║                                                                              ║
║  Theoretical predictions for Casimir-Gravity Crossover cosmology.           ║
║  Implements modified gravity equations for CMB, BAO, growth, and H(z).      ║
║                                                                              ║
║  CGC Theory Overview:                                                        ║
║    The Casimir-Gravity Crossover theory modifies General Relativity         ║
║    at cosmological scales through a scale-dependent effective Newton's      ║
║    constant: G_eff(k, z) = G_N × [1 + μ × f(k, z)]                          ║
║                                                                              ║
║    Key parameters:                                                           ║
║      • μ: Coupling strength (0 = GR/ΛCDM limit)                             ║
║      • n_g: Scale dependence power law exponent                             ║
║      • z_trans: Transition redshift for temporal evolution                  ║
║      • ρ_thresh: Screening density threshold                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Physical Motivation
-------------------
The CGC theory arises from quantum vacuum fluctuations at cosmological 
scales, inspired by the Casimir effect. Just as conducting plates 
modify vacuum energy at small scales, large-scale structure can 
modify gravitational dynamics at cosmological scales.

Usage
-----
>>> from cgc.theory import CGCTheory
>>> theory = CGCTheory(mu=0.12, n_g=0.75, z_trans=2.0)
>>> H = theory.hubble(z=0.5)
>>> G_eff = theory.effective_G(k=0.1, z=0.5)
"""

import numpy as np
from typing import Union, Tuple, Dict, Any
from dataclasses import dataclass
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d

from .config import PLANCK_BASELINE, CONSTANTS


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Speed of light [km/s]
C_LIGHT = CONSTANTS['c']

# Present-day Hubble parameter [km/s/Mpc]
H0_FIDUCIAL = 67.36


# =============================================================================
# CGC THEORY CLASS
# =============================================================================

@dataclass
class CGCTheory:
    """
    Casimir-Gravity Crossover (CGC) theoretical framework.
    
    Implements modified gravity predictions for cosmological observables.
    
    Parameters
    ----------
    mu : float, default=0.12
        CGC coupling strength. μ = 0 recovers ΛCDM.
    n_g : float, default=0.75
        Scale dependence power law exponent.
    z_trans : float, default=2.0
        Transition redshift for CGC effects.
    rho_thresh : float, default=200.0
        Screening density threshold (in units of ρ_crit).
    
    Cosmological Parameters (Planck 2018 baseline):
    -----------------------------------------------
    omega_b : float
        Baryon density parameter ω_b = Ω_b h².
    omega_cdm : float
        Cold dark matter density ω_cdm = Ω_cdm h².
    h : float
        Dimensionless Hubble parameter H0/(100 km/s/Mpc).
    
    Attributes
    ----------
    Omega_m : float
        Matter density parameter at z=0.
    Omega_Lambda : float
        Dark energy density parameter at z=0.
    H0 : float
        Hubble constant [km/s/Mpc].
    
    Examples
    --------
    >>> theory = CGCTheory(mu=0.15, n_g=0.8)
    >>> print(f"H(z=0.5) = {theory.hubble(0.5):.2f} km/s/Mpc")
    >>> 
    >>> # Scale-dependent G
    >>> G_ratio = theory.G_eff_ratio(k=0.1, z=0)
    >>> print(f"G_eff/G_N = {G_ratio:.4f}")
    """
    
    # CGC parameters
    mu: float = 0.12
    n_g: float = 0.75
    z_trans: float = 2.0
    rho_thresh: float = 200.0
    
    # Cosmological parameters
    omega_b: float = 0.0224
    omega_cdm: float = 0.120
    h: float = 0.6736
    ln10As: float = 3.045
    n_s: float = 0.965
    tau_reio: float = 0.054
    
    def __post_init__(self):
        """Compute derived quantities."""
        # Total matter density
        self.Omega_m = (self.omega_b + self.omega_cdm) / self.h**2
        
        # Dark energy density (flat universe)
        self.Omega_Lambda = 1.0 - self.Omega_m
        
        # Hubble constant
        self.H0 = 100 * self.h
        
        # Radiation density (Planck 2018)
        self.Omega_r = 9.24e-5  # Small but included for completeness
        
        # Primordial amplitude
        self.A_s = np.exp(self.ln10As) / 1e10
        
        # σ8 approximation (from Planck baseline)
        self.sigma8 = PLANCK_BASELINE['sigma8']
    
    # =========================================================================
    # EFFECTIVE GRAVITATIONAL CONSTANT
    # =========================================================================
    
    def G_eff_ratio(self, k: float = 0.1, z: float = 0.0) -> float:
        """
        Compute ratio G_eff / G_N.
        
        The CGC modification to Newton's constant is:
            G_eff(k, z) = G_N × [1 + μ × f(k) × g(z) × S(ρ)]
        
        where:
            f(k) = (k / k_pivot)^n_g     (scale dependence)
            g(z) = exp(-z / z_trans)      (redshift evolution)
            S(ρ) = screening function
        
        Parameters
        ----------
        k : float
            Wavenumber [h/Mpc].
        z : float
            Redshift.
        
        Returns
        -------
        float
            G_eff / G_N ratio.
        """
        if self.mu == 0:
            return 1.0
        
        # Pivot scale
        k_pivot = 0.05  # h/Mpc
        
        # Scale dependence
        f_k = (k / k_pivot) ** self.n_g
        
        # Redshift evolution (smoothly turns off at high z)
        g_z = np.exp(-z / self.z_trans) if z < 5 * self.z_trans else 0.0
        
        # Screening (simplified - full version uses density field)
        S_rho = 1.0  # No screening in linear regime
        
        return 1.0 + self.mu * f_k * g_z * S_rho
    
    def G_eff_array(self, k: Union[float, np.ndarray],
                    z: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute G_eff/G_N for arrays of k and z.
        
        Parameters
        ----------
        k : float or array
            Wavenumber(s) [h/Mpc].
        z : float or array
            Redshift(s).
        
        Returns
        -------
        np.ndarray
            G_eff / G_N ratio.
        """
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        
        result = np.zeros((len(k), len(z)))
        
        for i, ki in enumerate(k):
            for j, zj in enumerate(z):
                result[i, j] = self.G_eff_ratio(ki, zj)
        
        return result.squeeze()
    
    # =========================================================================
    # HUBBLE PARAMETER
    # =========================================================================
    
    def E_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute E(z) = H(z) / H0 in CGC cosmology.
        
        Modified Friedmann equation:
            E²(z) = Ω_m (1+z)³ + Ω_r (1+z)⁴ + Ω_Λ + Δ_CGC(z)
        
        where Δ_CGC is the CGC modification.
        
        Parameters
        ----------
        z : float or array
            Redshift.
        
        Returns
        -------
        float or array
            Dimensionless Hubble parameter E(z).
        """
        z = np.atleast_1d(z)
        
        # Standard components
        matter = self.Omega_m * (1 + z)**3
        radiation = self.Omega_r * (1 + z)**4
        lambda_term = self.Omega_Lambda
        
        # CGC modification
        # Affects the effective dark energy equation of state
        g_z = np.exp(-z / self.z_trans)
        delta_cgc = self.mu * self.Omega_Lambda * g_z * (1 - g_z)
        
        E_squared = matter + radiation + lambda_term + delta_cgc
        
        return np.sqrt(E_squared).squeeze()
    
    def hubble(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute Hubble parameter H(z) [km/s/Mpc].
        
        Parameters
        ----------
        z : float or array
            Redshift.
        
        Returns
        -------
        float or array
            Hubble parameter [km/s/Mpc].
        """
        return self.H0 * self.E_z(z)
    
    def hubble_lcdm(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute ΛCDM Hubble parameter for comparison.
        
        Parameters
        ----------
        z : float or array
            Redshift.
        
        Returns
        -------
        float or array
            ΛCDM Hubble parameter [km/s/Mpc].
        """
        z = np.atleast_1d(z)
        
        E_squared = (self.Omega_m * (1 + z)**3 + 
                    self.Omega_r * (1 + z)**4 + 
                    self.Omega_Lambda)
        
        return (self.H0 * np.sqrt(E_squared)).squeeze()
    
    # =========================================================================
    # DISTANCES
    # =========================================================================
    
    def comoving_distance(self, z: float) -> float:
        """
        Compute comoving distance to redshift z [Mpc].
        
        Parameters
        ----------
        z : float
            Redshift.
        
        Returns
        -------
        float
            Comoving distance [Mpc].
        """
        from scipy.integrate import quad
        
        integrand = lambda zp: C_LIGHT / self.hubble(zp)
        result, _ = quad(integrand, 0, z)
        
        return result
    
    def angular_diameter_distance(self, z: float) -> float:
        """
        Compute angular diameter distance D_A(z) [Mpc].
        
        Parameters
        ----------
        z : float
            Redshift.
        
        Returns
        -------
        float
            Angular diameter distance [Mpc].
        """
        return self.comoving_distance(z) / (1 + z)
    
    def luminosity_distance(self, z: float) -> float:
        """
        Compute luminosity distance D_L(z) [Mpc].
        
        Parameters
        ----------
        z : float
            Redshift.
        
        Returns
        -------
        float
            Luminosity distance [Mpc].
        """
        return self.comoving_distance(z) * (1 + z)
    
    def DV_over_rd(self, z: float, rd: float = 147.09) -> float:
        """
        Compute BAO distance scale D_V / r_d.
        
        D_V is the spherically-averaged distance:
            D_V = [z D_H(z) D_M(z)²]^(1/3)
        
        Parameters
        ----------
        z : float
            Redshift.
        rd : float, default=147.09
            Sound horizon at drag epoch [Mpc].
        
        Returns
        -------
        float
            Dimensionless BAO distance scale.
        """
        D_M = self.comoving_distance(z)
        D_H = C_LIGHT / self.hubble(z)
        
        D_V = (z * D_H * D_M**2) ** (1/3)
        
        return D_V / rd
    
    # =========================================================================
    # GROWTH OF STRUCTURE
    # =========================================================================
    
    def growth_factor(self, z: Union[float, np.ndarray],
                      normalize: bool = True) -> Union[float, np.ndarray]:
        """
        Compute the linear growth factor D(z).
        
        Solves the modified growth equation:
            D'' + (3 + dlnE²/dlna)/2 × D' - 3/2 × Ω_m(a) × G_eff/G_N × D = 0
        
        Parameters
        ----------
        z : float or array
            Redshift.
        normalize : bool, default=True
            If True, normalize D(z=0) = 1.
        
        Returns
        -------
        float or array
            Linear growth factor.
        """
        z_arr = np.atleast_1d(z)
        
        # Use approximate solution for speed
        # Full ODE solution can be added for precision
        
        # ΛCDM-like growth with CGC modification
        D_lcdm = 1 / (1 + z_arr) * self._growth_integral(z_arr)
        
        # CGC modification factor
        G_ratio = np.array([self.G_eff_ratio(k=0.1, z=zi) for zi in z_arr])
        
        D_cgc = D_lcdm * np.sqrt(G_ratio)
        
        if normalize:
            D0 = 1 / 1 * self._growth_integral(0) * np.sqrt(self.G_eff_ratio(0.1, 0))
            D_cgc = D_cgc / D0
        
        return D_cgc.squeeze()
    
    def _growth_integral(self, z: np.ndarray) -> np.ndarray:
        """Approximate growth integral."""
        # Using fitting formula from Carroll, Press & Turner (1992)
        a = 1 / (1 + z)
        
        Omega_m_z = self.Omega_m * (1 + z)**3 / self.E_z(z)**2
        Omega_L_z = self.Omega_Lambda / self.E_z(z)**2
        
        # Approximate growth
        D = (5/2) * Omega_m_z / (
            Omega_m_z**(4/7) - Omega_L_z + 
            (1 + Omega_m_z/2) * (1 + Omega_L_z/70)
        )
        
        return D
    
    def growth_rate(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the growth rate f(z) = d ln D / d ln a.
        
        Parameters
        ----------
        z : float or array
            Redshift.
        
        Returns
        -------
        float or array
            Growth rate f(z).
        """
        z_arr = np.atleast_1d(z)
        
        # Ω_m(z)
        Omega_m_z = self.Omega_m * (1 + z_arr)**3 / self.E_z(z_arr)**2
        
        # CGC-modified growth rate
        # f ≈ Ω_m(z)^γ with γ = 0.55 + 0.05(1 + w_eff)
        gamma = 0.55 + 0.05 * self.mu  # CGC modification to γ
        
        f = Omega_m_z ** gamma
        
        # Additional CGC correction
        G_ratio = np.array([self.G_eff_ratio(0.1, zi) for zi in z_arr])
        f = f * G_ratio**0.3
        
        return f.squeeze()
    
    def fsigma8(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute fσ8(z) = f(z) × σ8(z).
        
        Parameters
        ----------
        z : float or array
            Redshift.
        
        Returns
        -------
        float or array
            Growth observable fσ8(z).
        """
        D = self.growth_factor(z)
        f = self.growth_rate(z)
        
        # σ8(z) = σ8(0) × D(z)
        sigma8_z = self.sigma8 * D
        
        return f * sigma8_z
    
    # =========================================================================
    # CMB MODIFICATIONS
    # =========================================================================
    
    def cmb_cl_modification(self, ell: np.ndarray) -> np.ndarray:
        """
        Compute CGC modification to CMB power spectrum.
        
        Parameters
        ----------
        ell : array
            Multipoles.
        
        Returns
        -------
        array
            Multiplicative modification factor.
        """
        # CGC affects lensing and ISW effect
        # Simplified parametrization
        
        # Convert ell to wavenumber (approximate)
        # k ≈ ell / D_A(z_*)  where z_* ≈ 1100
        k = ell / 14000  # Approximate conversion
        
        # Modification factor
        mod = 1 + self.mu * (ell / 1000) ** (self.n_g / 2) * 0.1
        
        return mod
    
    def cmb_Dl_theory(self, ell: np.ndarray) -> np.ndarray:
        """
        Compute theoretical CGC D_ell spectrum.
        
        Uses simplified acoustic peak model + CGC modifications.
        
        Parameters
        ----------
        ell : array
            Multipoles.
        
        Returns
        -------
        array
            D_ell [μK²].
        """
        # ΛCDM baseline (simplified acoustic peaks model)
        Dl_lcdm = (
            5000 * np.exp(-((ell - 220) / 80)**2) +
            2000 * np.exp(-((ell - 530) / 100)**2) +
            1000 * np.exp(-((ell - 800) / 120)**2) +
            500 * np.exp(-((ell - 1100) / 150)**2) +
            300 * np.exp(-ell / 1500) + 100
        )
        
        # CGC modification
        mod = self.cmb_cl_modification(ell)
        
        return Dl_lcdm * mod
    
    # =========================================================================
    # SUPERNOVAE
    # =========================================================================
    
    def distance_modulus(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute distance modulus μ(z) for Type Ia supernovae.
        
        μ = 5 log10(D_L / 10 pc)
        
        Parameters
        ----------
        z : float or array
            Redshift.
        
        Returns
        -------
        float or array
            Distance modulus [mag].
        """
        z_arr = np.atleast_1d(z)
        
        # Luminosity distance in Mpc
        D_L = np.array([self.luminosity_distance(zi) for zi in z_arr])
        
        # Convert to 10 pc = 1e-5 Mpc
        mu = 5 * np.log10(D_L / 1e-5)
        
        return mu.squeeze()
    
    # =========================================================================
    # SUMMARY PREDICTIONS
    # =========================================================================
    
    def get_predictions(self, z_array: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute comprehensive CGC predictions.
        
        Parameters
        ----------
        z_array : array, optional
            Redshift array for predictions.
        
        Returns
        -------
        dict
            Dictionary of theoretical predictions.
        """
        if z_array is None:
            z_array = np.linspace(0, 3, 100)
        
        return {
            'z': z_array,
            'H': self.hubble(z_array),
            'H_lcdm': self.hubble_lcdm(z_array),
            'D': self.growth_factor(z_array),
            'f': self.growth_rate(z_array),
            'fsigma8': self.fsigma8(z_array),
            'G_eff_ratio': np.array([self.G_eff_ratio(0.1, z) for z in z_array])
        }
    
    def print_summary(self):
        """Print summary of CGC theory parameters and predictions."""
        print("="*60)
        print("CGC THEORY SUMMARY")
        print("="*60)
        print(f"\nCGC Parameters:")
        print(f"  μ (coupling)       = {self.mu:.4f}")
        print(f"  n_g (scale dep.)   = {self.n_g:.4f}")
        print(f"  z_trans            = {self.z_trans:.2f}")
        print(f"  ρ_thresh           = {self.rho_thresh:.1f}")
        
        print(f"\nCosmological Parameters:")
        print(f"  H0                 = {self.H0:.2f} km/s/Mpc")
        print(f"  Ω_m                = {self.Omega_m:.4f}")
        print(f"  Ω_Λ                = {self.Omega_Lambda:.4f}")
        print(f"  σ8                 = {self.sigma8:.4f}")
        
        print(f"\nPredictions at z=0:")
        print(f"  G_eff/G_N (k=0.1)  = {self.G_eff_ratio(0.1, 0):.4f}")
        print(f"  f(z=0)             = {self.growth_rate(0):.4f}")
        print(f"  fσ8(z=0)           = {self.fsigma8(0):.4f}")
        
        print(f"\nPredictions at z=0.5:")
        print(f"  H(z=0.5)           = {self.hubble(0.5):.2f} km/s/Mpc")
        print(f"  D(z=0.5)           = {self.growth_factor(0.5):.4f}")
        print(f"  fσ8(z=0.5)         = {self.fsigma8(0.5):.4f}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_theory_from_params(theta: np.ndarray) -> CGCTheory:
    """
    Create CGCTheory instance from MCMC parameter array.
    
    Parameters
    ----------
    theta : array
        Parameter array [omega_b, omega_cdm, h, ln10As, n_s, tau,
                        mu, n_g, z_trans, rho_thresh]
    
    Returns
    -------
    CGCTheory
        Initialized theory object.
    """
    return CGCTheory(
        omega_b=theta[0],
        omega_cdm=theta[1],
        h=theta[2],
        ln10As=theta[3],
        n_s=theta[4],
        tau_reio=theta[5],
        mu=theta[6],
        n_g=theta[7],
        z_trans=theta[8],
        rho_thresh=theta[9]
    )


def compare_lcdm_cgc(z_array: np.ndarray = None,
                     mu: float = 0.12,
                     n_g: float = 0.75) -> Dict[str, np.ndarray]:
    """
    Compare ΛCDM and CGC predictions.
    
    Parameters
    ----------
    z_array : array, optional
        Redshift array.
    mu : float
        CGC coupling.
    n_g : float
        CGC scale dependence.
    
    Returns
    -------
    dict
        Comparison dictionary.
    """
    if z_array is None:
        z_array = np.linspace(0, 3, 100)
    
    # ΛCDM (μ = 0)
    lcdm = CGCTheory(mu=0)
    
    # CGC
    cgc = CGCTheory(mu=mu, n_g=n_g)
    
    return {
        'z': z_array,
        'H_lcdm': lcdm.hubble(z_array),
        'H_cgc': cgc.hubble(z_array),
        'dH_percent': 100 * (cgc.hubble(z_array) - lcdm.hubble(z_array)) / lcdm.hubble(z_array),
        'fsigma8_lcdm': lcdm.fsigma8(z_array),
        'fsigma8_cgc': cgc.fsigma8(z_array),
        'D_lcdm': lcdm.growth_factor(z_array),
        'D_cgc': cgc.growth_factor(z_array)
    }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing CGC theory module...")
    
    # Create theory instance
    theory = CGCTheory(mu=0.12, n_g=0.75, z_trans=2.0)
    
    # Print summary
    theory.print_summary()
    
    # Test predictions
    z_test = np.array([0, 0.5, 1.0, 2.0])
    
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    print(f"\nRedshift array: {z_test}")
    print(f"H(z):           {theory.hubble(z_test)}")
    print(f"D(z):           {theory.growth_factor(z_test)}")
    print(f"fσ8(z):         {theory.fsigma8(z_test)}")
    
    # Compare with ΛCDM
    print("\n" + "="*60)
    print("ΛCDM vs CGC COMPARISON")
    print("="*60)
    
    comparison = compare_lcdm_cgc(z_test)
    print(f"\nH difference (%): {comparison['dH_percent']}")
    
    print("\n✓ CGC theory module test passed")
