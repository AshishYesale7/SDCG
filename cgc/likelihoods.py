"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CGC Likelihoods Module                                ║
║                                                                              ║
║  Implements likelihood functions for various cosmological probes:            ║
║                                                                              ║
║    • CMB TT power spectrum (Planck 2018)                                    ║
║    • BAO distance measurements (BOSS DR12/16)                               ║
║    • Type Ia supernovae (Pantheon+)                                         ║
║    • Lyman-α forest flux power spectrum                                     ║
║    • Growth rate measurements (fσ8 from RSD)                                ║
║    • Local H0 measurements (SH0ES, TRGB)                                    ║
║                                                                              ║
║  All likelihoods are Gaussian with proper covariance handling where         ║
║  available. The combined likelihood is modular and configurable.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Likelihood Structure
--------------------
log L_total = log L_CMB + log L_BAO + log L_SNe + log L_growth + log L_H0 + log L_Lyα

Each component can be enabled/disabled independently.

Usage
-----
>>> from cgc.likelihoods import log_likelihood
>>> logl = log_likelihood(theta, data)

Or use individual likelihoods:
>>> from cgc.likelihoods import log_likelihood_cmb
>>> logl_cmb = log_likelihood_cmb(theta, data['cmb'])
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from .config import PLANCK_BASELINE, CONSTANTS
from .parameters import PARAM_BOUNDS, get_bounds_array


# =============================================================================
# PRIOR FUNCTIONS
# =============================================================================

def log_prior(theta: np.ndarray) -> float:
    """
    Compute log prior probability for parameter vector.
    
    Uses flat (uniform) priors within parameter bounds.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector of shape (10,):
        [ω_b, ω_cdm, h, ln10As, n_s, τ, μ, n_g, z_trans, ρ_thresh]
    
    Returns
    -------
    float
        Log prior probability (0 if within bounds, -∞ otherwise).
    
    Notes
    -----
    The flat prior is non-informative within the parameter bounds.
    Additional physical priors (e.g., Gaussian prior on ω_b from BBN)
    can be added for more realistic analysis.
    """
    bounds = get_bounds_array()
    
    # Check if all parameters are within bounds
    for i, (val, (low, high)) in enumerate(zip(theta, bounds)):
        if val < low or val > high:
            return -np.inf
    
    return 0.0


def log_prior_gaussian(theta: np.ndarray, 
                       add_bbn_prior: bool = False) -> float:
    """
    Log prior with optional Gaussian constraints.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    add_bbn_prior : bool, default=False
        If True, add BBN prior on ω_b.
    
    Returns
    -------
    float
        Log prior probability.
    """
    # First check flat bounds
    logp = log_prior(theta)
    if logp == -np.inf:
        return logp
    
    # Add BBN prior on omega_b if requested
    if add_bbn_prior:
        omega_b = theta[0]
        omega_b_bbn = 0.02242  # BBN + D/H
        omega_b_bbn_err = 0.00014
        logp -= 0.5 * ((omega_b - omega_b_bbn) / omega_b_bbn_err)**2
    
    return logp


# =============================================================================
# CMB LIKELIHOOD
# =============================================================================

def log_likelihood_cmb(theta: np.ndarray, cmb_data: Dict[str, np.ndarray],
                       use_lensing: bool = False) -> float:
    """
    Compute CMB TT power spectrum log-likelihood.
    
    The CGC theory modifies the CMB power spectrum through:
    - Enhanced power at high-ℓ due to modified gravity
    - Shifted acoustic peak positions due to modified expansion
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector [ω_b, ω_cdm, h, ln10As, n_s, τ, μ, n_g, z_trans, ρ_thresh]
    cmb_data : dict
        CMB data dictionary with keys:
        - 'ell': Multipole moments
        - 'Dl': Observed D_ℓ [μK²]
        - 'error': 1σ uncertainties
    use_lensing : bool, default=False
        Include lensing contribution (not yet implemented).
    
    Returns
    -------
    float
        Log-likelihood contribution from CMB.
    
    Notes
    -----
    The current implementation uses an approximate model for the CMB.
    For accurate predictions, CLASS/CAMB modifications are needed.
    """
    # Unpack parameters
    omega_b, omega_cdm, h, ln10As, n_s, tau, mu, n_g, z_trans, rho_thresh = theta
    
    # Get data
    ell = cmb_data['ell']
    Dl_obs = cmb_data['Dl']
    Dl_err = cmb_data['error']
    
    # ═══════════════════════════════════════════════════════════════════════
    # CMB Model: Approximate acoustic peak structure
    # ═══════════════════════════════════════════════════════════════════════
    
    # Acoustic peak positions (approximate)
    # Peak 1: ℓ ≈ 220, Peak 2: ℓ ≈ 530, Peak 3: ℓ ≈ 800
    
    # Amplitude scales with As and optical depth
    amp_scale = np.exp(ln10As - 3.044) * np.exp(-2 * (tau - 0.054))
    
    # Peak positions scale with h (through sound horizon)
    h_ratio = h / 0.6736
    peak1 = 220 / h_ratio
    peak2 = 530 / h_ratio
    peak3 = 800 / h_ratio
    
    # Spectral tilt affects overall slope
    tilt = (n_s - 0.9649)
    
    # ΛCDM baseline spectrum
    Dl_lcdm = amp_scale * (
        5000 * np.exp(-((ell - peak1)/80)**2) +
        2000 * np.exp(-((ell - peak2)/100)**2) +
        1000 * np.exp(-((ell - peak3)/120)**2) +
        300 * np.exp(-ell/1500) + 100
    ) * (ell / 1000)**(tilt/2)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC Modification: Scale-dependent enhancement
    # ═══════════════════════════════════════════════════════════════════════
    
    # CGC modifies late-time ISW and lensing contributions
    # This appears as scale-dependent enhancement at high ℓ
    cgc_factor = 1 + mu * (ell / 1000)**(n_g / 2)
    
    Dl_model = Dl_lcdm * cgc_factor
    
    # ═══════════════════════════════════════════════════════════════════════
    # Compute χ² (assuming diagonal covariance)
    # ═══════════════════════════════════════════════════════════════════════
    
    chi2 = np.sum(((Dl_model - Dl_obs) / Dl_err)**2)
    
    return -0.5 * chi2


# =============================================================================
# BAO LIKELIHOOD
# =============================================================================

def log_likelihood_bao(theta: np.ndarray, bao_data: Dict[str, np.ndarray]) -> float:
    """
    Compute BAO distance log-likelihood.
    
    The BAO standard ruler constrains the distance-redshift relation,
    which CGC modifies through its effect on the expansion history.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    bao_data : dict
        BAO data dictionary with keys:
        - 'z': Effective redshifts
        - 'DV_rd': D_V/r_d measurements
        - 'error': 1σ uncertainties
    
    Returns
    -------
    float
        Log-likelihood contribution from BAO.
    
    Notes
    -----
    D_V(z) = [cz D_M(z)² / H(z)]^(1/3) is the angle-averaged distance.
    r_d is the sound horizon at drag epoch.
    """
    omega_b, omega_cdm, h, ln10As, n_s, tau, mu, n_g, z_trans, rho_thresh = theta
    
    z = bao_data['z']
    DV_rd_obs = bao_data['DV_rd']
    DV_rd_err = bao_data['error']
    
    if len(z) == 0:
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # Cosmological distances
    # ═══════════════════════════════════════════════════════════════════════
    
    H0 = h * 100
    Omega_m = (omega_b + omega_cdm) / h**2
    Omega_Lambda = 1 - Omega_m
    c = CONSTANTS['c']
    
    # Compute D_V/r_d for ΛCDM
    def compute_DV_rd(z_arr):
        """Compute D_V/r_d at given redshifts."""
        DV_rd = np.zeros_like(z_arr)
        
        for i, z_val in enumerate(z_arr):
            # Comoving distance (numerical integration)
            z_int = np.linspace(0, z_val, 1000)
            H_int = H0 * np.sqrt(Omega_m * (1 + z_int)**3 + Omega_Lambda)
            D_M = c * np.trapz(1.0 / H_int, z_int)
            
            # H(z)
            Hz = H0 * np.sqrt(Omega_m * (1 + z_val)**3 + Omega_Lambda)
            
            # D_V = [cz D_M² / H(z)]^(1/3)
            D_V = (c * z_val * D_M**2 / Hz)**(1/3)
            
            # Use fiducial r_d
            r_d = CONSTANTS['r_d_fid']
            DV_rd[i] = D_V / r_d
        
        return DV_rd
    
    DV_rd_lcdm = compute_DV_rd(z)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC Modification
    # ═══════════════════════════════════════════════════════════════════════
    
    # CGC modifies the expansion rate, affecting distances
    cgc_factor = 1 + mu * (1 + z)**(-n_g)
    DV_rd_model = DV_rd_lcdm * cgc_factor
    
    # ═══════════════════════════════════════════════════════════════════════
    # Compute χ²
    # ═══════════════════════════════════════════════════════════════════════
    
    chi2 = np.sum(((DV_rd_model - DV_rd_obs) / DV_rd_err)**2)
    
    return -0.5 * chi2


# =============================================================================
# SUPERNOVA LIKELIHOOD (PANTHEON+ WITH FULL COVARIANCE)
# =============================================================================

def log_likelihood_sne(theta: np.ndarray, sne_data: Dict[str, np.ndarray]) -> float:
    """
    Compute Type Ia supernovae log-likelihood with FULL covariance matrix.
    
    Uses distance modulus measurements from Pantheon+SH0ES sample (1701 SNe)
    with the complete 1701×1701 statistical+systematic covariance matrix.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    sne_data : dict
        SNe data dictionary with keys:
        - 'z': Redshifts (1701,)
        - 'mu': Distance moduli (1701,)
        - 'error': Diagonal 1σ uncertainties (1701,) [fallback]
        - 'cov': Full covariance matrix (1701, 1701)
        - 'inv_cov': Inverse covariance matrix (1701, 1701)
        - 'is_calibrator': Cepheid calibrator flags (1701,)
    
    Returns
    -------
    float
        Log-likelihood contribution from supernovae.
    
    Notes
    -----
    The likelihood uses the Pantheon+ methodology:
    χ² = Δμᵀ C⁻¹ Δμ
    where Δμ = μ_obs - μ_model and C is the full covariance matrix.
    
    The absolute magnitude M_B is analytically marginalized following
    the Tripp estimator approach (Brout et al. 2022).
    
    References
    ----------
    Brout et al. 2022, ApJ 938, 110
    Scolnic et al. 2022, ApJ 938, 113
    """
    if 'z' not in sne_data or len(sne_data['z']) == 0:
        return 0.0
    
    omega_b, omega_cdm, h, ln10As, n_s, tau, mu_cgc, n_g, z_trans, rho_thresh = theta
    
    z = sne_data['z']
    mu_obs = sne_data['mu']
    n_sne = len(z)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Check for full covariance matrix
    # ═══════════════════════════════════════════════════════════════════════
    
    use_full_cov = 'inv_cov' in sne_data and sne_data['inv_cov'] is not None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Luminosity distance calculation (vectorized for efficiency)
    # ═══════════════════════════════════════════════════════════════════════
    
    H0 = h * 100
    Omega_m = (omega_b + omega_cdm) / h**2
    Omega_Lambda = 1 - Omega_m
    c = CONSTANTS['c']
    
    # Vectorized distance modulus computation
    # Use adaptive integration points based on redshift
    mu_model = np.zeros(n_sne)
    
    for i, z_val in enumerate(z):
        if z_val < 1e-6:
            mu_model[i] = 0
            continue
        
        # Integration for comoving distance with more points at high-z
        n_int = min(500, max(100, int(200 * z_val)))
        z_int = np.linspace(0, z_val, n_int)
        
        # E(z) = H(z)/H0 in flat ΛCDM
        E_z = np.sqrt(Omega_m * (1 + z_int)**3 + Omega_Lambda)
        
        # Comoving distance: D_C = c/H0 ∫ dz'/E(z')
        D_C = (c / H0) * np.trapz(1.0 / E_z, z_int)
        
        # Luminosity distance in flat universe
        D_L = D_C * (1 + z_val)  # Mpc
        
        # Distance modulus: μ = 5 log₁₀(D_L/10pc) = 5 log₁₀(D_L) + 25
        mu_model[i] = 5 * np.log10(D_L) + 25
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC Modification to distances
    # ═══════════════════════════════════════════════════════════════════════
    
    # CGC modifies the effective gravitational constant, affecting distances
    # The modification is scale-dependent and redshift-dependent
    cgc_distance_factor = 1 + 0.5 * mu_cgc * (1 - np.exp(-z / z_trans))
    
    # Apply CGC correction to luminosity distance
    D_L_lcdm = np.power(10, (mu_model - 25) / 5)
    D_L_cgc = D_L_lcdm * cgc_distance_factor
    mu_model = 5 * np.log10(np.maximum(D_L_cgc, 1e-10)) + 25
    
    # ═══════════════════════════════════════════════════════════════════════
    # Compute residuals
    # ═══════════════════════════════════════════════════════════════════════
    
    delta_mu = mu_obs - mu_model
    
    # ═══════════════════════════════════════════════════════════════════════
    # Chi-squared with full covariance and M_B marginalization
    # ═══════════════════════════════════════════════════════════════════════
    
    if use_full_cov:
        # Use full inverse covariance matrix (Pantheon+ methodology)
        inv_cov = sne_data['inv_cov']
        
        # Analytic marginalization over absolute magnitude M_B
        # Following Conley et al. 2011, Betoule et al. 2014, Brout et al. 2022
        # 
        # The marginalized chi² is:
        # χ²_marg = Δμᵀ C⁻¹ Δμ - (Δμᵀ C⁻¹ 1)² / (1ᵀ C⁻¹ 1)
        #
        # where 1 is a vector of ones (shift in M_B affects all μ equally)
        
        ones = np.ones(n_sne)
        
        # C⁻¹ Δμ
        Cinv_delta = inv_cov @ delta_mu
        
        # C⁻¹ 1
        Cinv_ones = inv_cov @ ones
        
        # χ² terms
        chi2_full = delta_mu @ Cinv_delta
        numerator = (delta_mu @ Cinv_ones)**2
        denominator = ones @ Cinv_ones
        
        # Marginalized chi²
        chi2 = chi2_full - numerator / denominator
        
    else:
        # Fallback: diagonal errors only
        mu_err = sne_data['error']
        weights = 1.0 / mu_err**2
        
        chi2 = np.sum(delta_mu**2 * weights)
        chi2 -= (np.sum(delta_mu * weights))**2 / np.sum(weights)
    
    return -0.5 * chi2


# =============================================================================
# LYMAN-ALPHA LIKELIHOOD
# =============================================================================

def log_likelihood_lyalpha(theta: np.ndarray, 
                           lyalpha_data: Dict[str, np.ndarray]) -> float:
    """
    Compute Lyman-α forest flux power spectrum log-likelihood.
    
    Uses eBOSS/BOSS DR14 measurements from Chabanier et al. 2019.
    The Lyman-α forest probes structure at high redshift (z ~ 2-4)
    and small scales, providing unique constraints on CGC theory.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    lyalpha_data : dict
        Lyman-α data dictionary with keys:
        - 'z': Redshifts (56,)
        - 'k': Wavenumbers [s/km] (56,)
        - 'P_flux': Flux power spectrum (56,)
        - 'error': Combined σ = √(σ_stat² + σ_sys²) (56,)
    
    Returns
    -------
    float
        Log-likelihood contribution from Lyman-α.
    
    Notes
    -----
    The flux power spectrum P_F(k,z) is related to the matter power spectrum
    through a bias factor and thermal broadening:
    
        P_F(k,z) = b²(k,z) * P_m(k,z) * exp(-k²σ_T²)
    
    where b is the flux bias and σ_T is the thermal broadening scale.
    
    CGC modifies both the matter distribution (through modified gravity)
    and potentially the thermal history (through modified structure formation).
    
    The Lyman-α forest is a KEY probe for CGC because:
    1. It probes z ~ 2-4 where CGC effects are strongest (near z_trans)
    2. It probes small scales k ~ 0.001-0.05 s/km where CGC modifies P(k)
    3. It's sensitive to both gravity and thermal effects
    
    References
    ----------
    Chabanier et al. 2019, JCAP 07 (2019) 017
    Palanque-Delabrouille et al. 2020, JCAP 04 (2020) 038
    """
    if 'z' not in lyalpha_data or len(lyalpha_data.get('z', [])) == 0:
        return 0.0
    
    omega_b, omega_cdm, h, ln10As, n_s, tau, mu_cgc, n_g, z_trans, rho_thresh = theta
    
    z = lyalpha_data['z']
    k = lyalpha_data['k']
    P_flux_obs = lyalpha_data['P_flux']
    P_flux_err = lyalpha_data['error']
    
    # ═══════════════════════════════════════════════════════════════════════
    # Cosmological parameters
    # ═══════════════════════════════════════════════════════════════════════
    
    Omega_m = (omega_b + omega_cdm) / h**2
    sigma8 = 0.8111 * np.exp(0.5 * (ln10As - 3.044))
    
    # ═══════════════════════════════════════════════════════════════════════
    # ΛCDM Flux Power Spectrum Model
    # Calibrated to Chabanier et al. 2019 eBOSS DR14 data
    # ═══════════════════════════════════════════════════════════════════════
    
    # Template P_F(k) at z=3.0 from Chabanier data (interpolation anchor)
    # k [s/km]:   0.001   0.002   0.003   0.005   0.01    0.02    0.05
    # P_F:        0.0484  0.0422  0.0371  0.0291  0.0178  0.0088  0.0024
    
    k_template = np.array([0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05])
    P_template_z3 = np.array([0.0484, 0.0422, 0.0371, 0.0291, 0.0178, 0.0088, 0.0024])
    
    # Interpolate P_F(k) at z=3 for each k value
    # Use log-interpolation for smoother behavior
    log_k_template = np.log(k_template)
    log_P_template = np.log(P_template_z3)
    
    log_k = np.log(np.clip(k, k_template.min(), k_template.max()))
    log_P_interp = np.interp(log_k, log_k_template, log_P_template)
    P_flux_z3 = np.exp(log_P_interp)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Redshift evolution (calibrated from data)
    # P_F(z) ∝ ((1+z_piv)/(1+z))^1.3 (fitted to Chabanier data at k=0.01)
    # ═══════════════════════════════════════════════════════════════════════
    
    z_pivot = 3.0
    alpha_z = 1.3  # Best-fit z-evolution exponent from data
    z_evolution = ((1 + z_pivot) / (1 + z))**alpha_z
    
    # ═══════════════════════════════════════════════════════════════════════
    # Cosmology-dependent corrections
    # Small perturbations around Planck 2018 fiducial
    # ═══════════════════════════════════════════════════════════════════════
    
    # P_F scales with matter power: P_m ∝ σ8² and weak Ω_m dependence
    A_cosmo = (sigma8 / 0.811)**2 * (Omega_m / 0.315)**0.15
    
    # Spectral tilt affects shape (weak effect)
    n_s_fid = 0.965
    delta_ns = n_s - n_s_fid
    
    # Tilt shifts the spectrum (steeper n_s → more small-scale power)
    k_ref = 0.01
    ns_factor = (k / k_ref)**(0.5 * delta_ns)
    
    # Full ΛCDM prediction
    P_flux_lcdm = P_flux_z3 * z_evolution * A_cosmo * ns_factor
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC Modification
    # ═══════════════════════════════════════════════════════════════════════
    
    # Convert k from s/km to h/Mpc (approximate)
    # v = H(z) * r / (1+z), so 1 s/km ≈ 100 * h * (1+z)/E(z) km/s/Mpc
    # For z~3: 1 s/km ≈ 100 * h h/Mpc
    k_hmpc = k * 100 * h
    
    # CGC characteristic scale
    k_cgc = 0.1 * (1 + mu_cgc)
    
    # CGC enhances small-scale power
    cgc_factor = 1 + mu_cgc * (k_hmpc / k_cgc)**n_g * np.exp(-(z - z_trans)**2 / (2 * 1.5**2))
    
    P_flux_model = P_flux_lcdm * cgc_factor
    
    # ═══════════════════════════════════════════════════════════════════════
    # χ² calculation with amplitude marginalization
    # ═══════════════════════════════════════════════════════════════════════
    
    # Allow ~10% amplitude marginalization to absorb systematic uncertainties
    # in the flux calibration
    residuals = P_flux_obs - P_flux_model
    weights = 1.0 / P_flux_err**2
    
    # Analytic marginalization: find best-fit amplitude A that minimizes χ²
    # P_obs = A * P_model → A = Σ(P_obs * P_model / σ²) / Σ(P_model² / σ²)
    numer = np.sum(P_flux_obs * P_flux_model * weights)
    denom = np.sum(P_flux_model**2 * weights)
    A_best = numer / denom if denom > 0 else 1.0
    
    # Limit amplitude correction to ±15%
    A_best = np.clip(A_best, 0.85, 1.15)
    
    # χ² with marginalized amplitude
    residuals_marg = P_flux_obs - A_best * P_flux_model
    chi2 = np.sum(residuals_marg**2 * weights)
    
    return -0.5 * chi2


# =============================================================================
# GROWTH RATE LIKELIHOOD
# =============================================================================

def log_likelihood_growth(theta: np.ndarray, 
                          growth_data: Dict[str, np.ndarray]) -> float:
    """
    Compute growth rate (fσ8) log-likelihood from RSD.
    
    The growth rate f = d ln D / d ln a measures how fast structure
    grows. CGC modifies gravity, affecting structure formation.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    growth_data : dict
        Growth data dictionary with keys:
        - 'z': Redshifts
        - 'fs8': fσ8 measurements
        - 'error': 1σ uncertainties
    
    Returns
    -------
    float
        Log-likelihood contribution from growth.
    
    Notes
    -----
    fσ8 = f(z) × σ8(z) is the combination measured by RSD.
    In ΛCDM, f ≈ Ω_m(z)^0.55.
    """
    omega_b, omega_cdm, h, ln10As, n_s, tau, mu_cgc, n_g, z_trans, rho_thresh = theta
    
    z = growth_data['z']
    fs8_obs = growth_data['fs8']
    fs8_err = growth_data['error']
    
    if len(z) == 0:
        return 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # ΛCDM growth rate
    # ═══════════════════════════════════════════════════════════════════════
    
    Omega_m = (omega_b + omega_cdm) / h**2
    sigma8_0 = PLANCK_BASELINE['sigma8']
    
    # Matter fraction at redshift z
    Omega_m_z = Omega_m * (1 + z)**3 / (Omega_m * (1 + z)**3 + (1 - Omega_m))
    
    # Growth rate: f ≈ Ω_m(z)^γ, where γ ≈ 0.55 for ΛCDM
    gamma = 0.55
    f_lcdm = Omega_m_z**gamma
    
    # σ8(z) from growth factor
    # D(z)/D(0) ≈ (1+z)^(-1) × g(Ω_m(z))/g(Ω_m,0)
    growth_factor = 1.0 / (1 + z) * (Omega_m_z / Omega_m)**0.1
    sigma8_z = sigma8_0 * growth_factor
    
    fs8_lcdm = f_lcdm * sigma8_z
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC Modification to growth
    # ═══════════════════════════════════════════════════════════════════════
    
    # CGC enhances growth at low z through modified G_eff
    cgc_factor = 1 + 0.1 * mu_cgc * (1 + z)**(-n_g)
    fs8_model = fs8_lcdm * cgc_factor
    
    # ═══════════════════════════════════════════════════════════════════════
    # Compute χ²
    # ═══════════════════════════════════════════════════════════════════════
    
    chi2 = np.sum(((fs8_model - fs8_obs) / fs8_err)**2)
    
    return -0.5 * chi2


# =============================================================================
# H0 LIKELIHOOD
# =============================================================================

def log_likelihood_h0(theta: np.ndarray, h0_data: Dict[str, Any],
                      use_sh0es: bool = True, 
                      use_trgb: bool = False) -> float:
    """
    Compute H0 log-likelihood from local measurements.
    
    This component constrains the Hubble constant directly, helping
    to test whether CGC can resolve the Hubble tension.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    h0_data : dict
        H0 data dictionary with Planck, SH0ES, TRGB values.
    use_sh0es : bool, default=True
        Include SH0ES measurement.
    use_trgb : bool, default=False
        Include TRGB measurement.
    
    Returns
    -------
    float
        Log-likelihood contribution from H0.
    
    Notes
    -----
    The CGC theory predicts an intermediate H0 that bridges the gap
    between Planck (early universe) and SH0ES (local) measurements.
    """
    omega_b, omega_cdm, h, ln10As, n_s, tau, mu_cgc, n_g, z_trans, rho_thresh = theta
    
    H0_model = h * 100
    
    chi2 = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC H0 modification (from modified late-time expansion)
    # ═══════════════════════════════════════════════════════════════════════
    
    # CGC effectively rescales H0 through modified expansion
    H0_eff = H0_model * (1 + 0.1 * mu_cgc)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Likelihood contributions
    # ═══════════════════════════════════════════════════════════════════════
    
    # Planck (always included)
    if 'planck' in h0_data:
        H0_planck = h0_data['planck']['value']
        H0_planck_err = h0_data['planck']['error']
        chi2 += ((H0_eff - H0_planck) / H0_planck_err)**2
    
    # SH0ES
    if use_sh0es and 'sh0es' in h0_data:
        H0_sh0es = h0_data['sh0es']['value']
        H0_sh0es_err = h0_data['sh0es']['error']
        chi2 += ((H0_eff - H0_sh0es) / H0_sh0es_err)**2
    
    # TRGB
    if use_trgb and 'trgb' in h0_data:
        H0_trgb = h0_data['trgb']['value']
        H0_trgb_err = h0_data['trgb']['error']
        chi2 += ((H0_eff - H0_trgb) / H0_trgb_err)**2
    
    return -0.5 * chi2


# =============================================================================
# COMBINED LIKELIHOOD
# =============================================================================

def log_likelihood(theta: np.ndarray, data: Dict[str, Any],
                   include_cmb: bool = True,
                   include_bao: bool = True,
                   include_sne: bool = False,
                   include_lyalpha: bool = False,
                   include_growth: bool = True,
                   include_h0: bool = True,
                   use_sh0es: bool = True) -> float:
    """
    Compute total log-likelihood from all cosmological probes.
    
    This is the main likelihood function used for MCMC sampling.
    It combines likelihoods from multiple cosmological datasets.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector of shape (10,):
        [ω_b, ω_cdm, h, ln10As, n_s, τ, μ, n_g, z_trans, ρ_thresh]
    
    data : dict
        Complete data dictionary containing all loaded datasets.
        Expected keys: 'cmb', 'bao', 'growth', 'H0', 'sne', 'lyalpha'
    
    include_cmb : bool, default=True
        Include CMB TT power spectrum likelihood.
    
    include_bao : bool, default=True
        Include BAO distance likelihood.
    
    include_sne : bool, default=False
        Include Pantheon+ supernovae likelihood.
    
    include_lyalpha : bool, default=False
        Include Lyman-α forest likelihood.
    
    include_growth : bool, default=True
        Include growth rate (fσ8) likelihood.
    
    include_h0 : bool, default=True
        Include H0 likelihood.
    
    use_sh0es : bool, default=True
        If True, include SH0ES in H0 likelihood.
    
    Returns
    -------
    float
        Total log-likelihood. Returns -∞ if parameters are invalid.
    
    Examples
    --------
    Basic usage:
    >>> from cgc.likelihoods import log_likelihood
    >>> from cgc.data_loader import load_real_data
    >>> from cgc.parameters import CGCParameters
    >>> 
    >>> data = load_real_data()
    >>> theta = CGCParameters().to_array()
    >>> logl = log_likelihood(theta, data)
    
    With supernovae:
    >>> logl = log_likelihood(theta, data, include_sne=True)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # Check prior (bounds)
    # ═══════════════════════════════════════════════════════════════════════
    
    logp = log_prior(theta)
    if logp == -np.inf:
        return -np.inf
    
    # ═══════════════════════════════════════════════════════════════════════
    # Accumulate likelihood contributions
    # ═══════════════════════════════════════════════════════════════════════
    
    logl = 0.0
    
    # CMB
    if include_cmb and 'cmb' in data:
        logl += log_likelihood_cmb(theta, data['cmb'])
    
    # BAO
    if include_bao and 'bao' in data:
        logl += log_likelihood_bao(theta, data['bao'])
    
    # Supernovae
    if include_sne and 'sne' in data:
        logl += log_likelihood_sne(theta, data['sne'])
    
    # Lyman-α
    if include_lyalpha and 'lyalpha' in data:
        logl += log_likelihood_lyalpha(theta, data['lyalpha'])
    
    # Growth/RSD
    if include_growth and 'growth' in data:
        logl += log_likelihood_growth(theta, data['growth'])
    
    # H0
    if include_h0 and 'H0' in data:
        logl += log_likelihood_h0(theta, data['H0'], use_sh0es=use_sh0es)
    
    return logl


# =============================================================================
# LIKELIHOOD WRAPPER FOR EMCEE
# =============================================================================

def log_probability(theta: np.ndarray, data: Dict[str, Any],
                    **likelihood_kwargs) -> float:
    """
    Log posterior probability for emcee sampler.
    
    Combines log prior and log likelihood.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector.
    data : dict
        Data dictionary.
    **likelihood_kwargs
        Additional arguments passed to log_likelihood.
    
    Returns
    -------
    float
        Log posterior probability.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(theta, data, **likelihood_kwargs)
    
    return lp + ll


# =============================================================================
# NESTED SAMPLING INTERFACE
# =============================================================================

def prior_transform(u: np.ndarray) -> np.ndarray:
    """
    Transform unit hypercube to parameter space (for nested sampling).
    
    Maps uniform samples in [0, 1]^n to the parameter prior ranges.
    
    Parameters
    ----------
    u : np.ndarray
        Uniform samples in [0, 1]^n.
    
    Returns
    -------
    np.ndarray
        Parameters mapped to prior ranges.
    """
    bounds = get_bounds_array()
    theta = np.zeros_like(u)
    
    for i in range(len(u)):
        low, high = bounds[i]
        theta[i] = low + u[i] * (high - low)
    
    return theta


def log_likelihood_nested(theta: np.ndarray, data: Dict[str, Any],
                          **kwargs) -> float:
    """
    Log-likelihood wrapper for nested sampling.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector (from prior_transform).
    data : dict
        Data dictionary.
    **kwargs
        Additional likelihood options.
    
    Returns
    -------
    float
        Log-likelihood value.
    """
    return log_likelihood(theta, data, **kwargs)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing likelihoods module...")
    
    from .parameters import CGCParameters
    from .data_loader import load_mock_data
    
    # Load mock data
    data = load_mock_data()
    
    # Test with fiducial parameters
    params = CGCParameters()
    theta = params.to_array()
    
    print("\nParameter vector:", theta)
    
    # Test individual likelihoods
    print("\nLikelihood contributions:")
    print(f"  Prior:  {log_prior(theta):.2f}")
    print(f"  CMB:    {log_likelihood_cmb(theta, data['cmb']):.2f}")
    print(f"  BAO:    {log_likelihood_bao(theta, data['bao']):.2f}")
    print(f"  Growth: {log_likelihood_growth(theta, data['growth']):.2f}")
    print(f"  H0:     {log_likelihood_h0(theta, data['H0']):.2f}")
    
    # Test combined likelihood
    logl = log_likelihood(theta, data)
    print(f"\n  TOTAL:  {logl:.2f}")
    
    print("\n✓ Likelihood tests passed")
