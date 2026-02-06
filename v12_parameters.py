#!/usr/bin/env python3
"""
v12 Parameters - Single Source of Truth
========================================

This file contains ALL standardized parameters for the MCMC_cgc project.
Import from this file to ensure consistency across the project.

Based on:
- Planck 2018 cosmology
- Latest SDCG/CGC model constraints
- Published simulation results

Usage:
    from v12_parameters import COSMO, CGC, PHYSICS
    
    H0 = COSMO['H0']  # 67.4
    mu = CGC['mu_upper_limit']  # 0.012
"""

# =============================================================================
# COSMOLOGICAL PARAMETERS (Planck 2018 - arXiv:1807.06209)
# =============================================================================

COSMO = {
    # Hubble constant
    'H0': 67.4,  # km/s/Mpc
    'H0_err': 0.5,  # km/s/Mpc (Planck uncertainty)
    'h': 0.674,  # H0/100
    
    # Matter densities
    'Omega_m': 0.315,
    'Omega_m_err': 0.007,
    'Omega_b': 0.0493,  # Baryonic
    'Omega_b_h2': 0.02237,
    'Omega_cdm': 0.266,  # Cold dark matter
    'Omega_cdm_h2': 0.1200,
    
    # Dark energy / Cosmological constant
    'Omega_Lambda': 0.685,
    'w0': -1.0,  # Dark energy EOS
    'wa': 0.0,  # Dark energy evolution
    
    # Primordial perturbations
    'sigma8': 0.811,
    'sigma8_err': 0.006,
    'n_s': 0.965,  # Spectral index
    'n_s_err': 0.004,
    'A_s': 2.1e-9,  # Amplitude at k=0.05 Mpc^-1
    
    # Optical depth
    'tau': 0.054,
    'tau_err': 0.007,
    
    # Derived parameters
    'S8': 0.832,  # sigma8 * sqrt(Omega_m/0.3)
    'S8_err': 0.013,
    
    # Ages and distances
    'age_Gyr': 13.797,  # Age of universe in Gyr
    'z_reion': 7.67,  # Reionization redshift
}


# =============================================================================
# CGC/SDCG MODEL PARAMETERS
# =============================================================================

CGC = {
    # Core CGC parameters
    'alpha_cgc': 0.48,  # Coupling shape parameter
    'alpha_cgc_err': 0.05,
    'beta_cgc': 1.2,  # Redshift evolution
    'beta_cgc_err': 0.1,
    'gamma_cgc': 0.85,  # Scale dependence
    'gamma_cgc_err': 0.08,
    
    # Mu parameter (gravitational coupling modification)
    'mu_mcmc': 0.41,  # From MCMC without Lyman-alpha constraint
    'mu_mcmc_err': 0.04,
    'mu_upper_limit': 0.012,  # 95% CL upper limit with Lyman-alpha
    'mu_upper_limit_90': 0.024,  # 90% CL
    'mu_upper_limit_99': 0.005,  # 99% CL
    
    # EFT priors (from EFT of dark energy)
    'mu_eft_prior': 0.149,
    'mu_eft_prior_err': 0.02,
    
    # Transition parameters
    'n_g': 0.65,  # Growth index modification
    'n_g_err': 0.05,
    'z_trans': 2.43,  # Transition redshift
    'z_trans_err': 0.15,
    
    # Screening
    'rho_thresh': 100.0,  # Screening threshold (rho/rho_mean)
    'screening_efficiency': 0.85,
}


# =============================================================================
# DWARF GALAXY / TIDAL STRIPPING PARAMETERS
# =============================================================================

DWARF = {
    # Tidal stripping velocity shift (from simulations)
    'delta_v_strip': 7.9,  # km/s
    'delta_v_strip_err': 0.9,
    
    # Individual simulation results
    'delta_v_EAGLE': 6.5,  # km/s
    'delta_v_FIRE': 9.2,
    'delta_v_IllustrisTNG': 7.8,
    'delta_v_SIMBA': 8.1,
    
    # Observed values (from LITTLE THINGS and SPARC)
    'delta_v_observed': -2.49,  # km/s (field - cluster)
    'delta_v_observed_err': 5.0,
    
    # Environment classification
    'r_virial_cluster': 1.5,  # Mpc
    'r_isolation_field': 3.0,  # Mpc
    
    # Velocity dispersion
    'sigma_v_cluster': 12.5,  # km/s
    'sigma_v_field': 8.2,
}


# =============================================================================
# LYMAN-ALPHA CONSTRAINTS
# =============================================================================

LYMAN_ALPHA = {
    # Redshift range
    'z_min': 2.2,
    'z_max': 4.6,
    'z_mean': 2.8,
    
    # P1D constraints
    'mu_eff': 5.76e-5,  # Effective optical depth parameter
    'mu_eff_err': 0.1e-5,
    
    # Maximum allowed enhancement
    'max_enhancement_5pct': 1.05,  # 5% deviation tolerance
    'max_enhancement_10pct': 1.10,
    
    # Key result: tension
    'tension_no_lya_sigma': 3.92,  # Tension in sigma without Lya
    'tension_with_lya_sigma': 0.59,  # Tension reduced with Lya
}


# =============================================================================
# PHYSICAL CONSTANTS (SI units unless noted)
# =============================================================================

PHYSICS = {
    # Fundamental constants
    'c_km_s': 299792.458,  # Speed of light km/s
    'c_m_s': 299792458.0,  # Speed of light m/s
    'G': 6.67430e-11,  # Gravitational constant N⋅m²/kg²
    'G_cgs': 6.67430e-8,  # cm³/(g⋅s²)
    'h_planck': 6.62607e-34,  # Planck constant J⋅s
    'hbar': 1.054572e-34,  # Reduced Planck constant
    'k_B': 1.380649e-23,  # Boltzmann constant J/K
    
    # Astronomical constants
    'M_sun': 1.989e30,  # kg
    'M_sun_cgs': 1.989e33,  # g
    'pc_to_m': 3.0857e16,  # parsec in meters
    'Mpc_to_m': 3.0857e22,  # Megaparsec in meters
    'Mpc_to_km': 3.0857e19,  # Megaparsec in km
    'Gyr_to_s': 3.1557e16,  # Gigayear in seconds
    
    # CMB temperature
    'T_CMB': 2.7255,  # K
    
    # Critical density (h=1)
    'rho_crit_h2': 2.775e11,  # M_sun/Mpc³ × h²
}


# =============================================================================
# DEPRECATED VALUES - DO NOT USE
# =============================================================================

DEPRECATED = {
    'H0_WMAP': 70.0,  # Old WMAP value
    'H0_SH0ES': 73.04,  # Local measurement (tension with Planck)
    'Omega_m_old': 0.3,  # Rounded value
    'sigma8_old': 0.8,  # Old value
}


# =============================================================================
# PRIORS FOR MCMC
# =============================================================================

PRIORS = {
    'H0': {'min': 60.0, 'max': 80.0, 'type': 'flat'},
    'Omega_m': {'min': 0.2, 'max': 0.5, 'type': 'flat'},
    'sigma8': {'min': 0.6, 'max': 1.0, 'type': 'flat'},
    'mu': {'min': 0.0, 'max': 1.0, 'type': 'flat'},
    'n_g': {'min': 0.0, 'max': 1.0, 'type': 'flat'},
    'z_trans': {'min': 1.0, 'max': 5.0, 'type': 'flat'},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_parameters():
    """Return all parameters as a single dict"""
    return {
        'cosmology': COSMO,
        'cgc': CGC,
        'dwarf': DWARF,
        'lyman_alpha': LYMAN_ALPHA,
        'physics': PHYSICS,
        'priors': PRIORS,
    }


def validate_parameter(name, value):
    """Check if a parameter value matches v12 specs"""
    all_params = get_all_parameters()
    
    for category, params in all_params.items():
        if name in params:
            expected = params[name]
            if isinstance(expected, (int, float)):
                tolerance = abs(expected * 0.01)  # 1% tolerance
                if abs(value - expected) > tolerance:
                    return False, f"Expected {expected}, got {value}"
            return True, "OK"
    
    return None, f"Parameter {name} not found in v12 specs"


def print_summary():
    """Print summary of v12 parameters"""
    print("=" * 60)
    print("v12 PARAMETER SUMMARY")
    print("=" * 60)
    print(f"\nCosmology (Planck 2018):")
    print(f"  H0 = {COSMO['H0']} ± {COSMO['H0_err']} km/s/Mpc")
    print(f"  Ωm = {COSMO['Omega_m']} ± {COSMO['Omega_m_err']}")
    print(f"  σ8 = {COSMO['sigma8']} ± {COSMO['sigma8_err']}")
    print(f"  S8 = {COSMO['S8']} ± {COSMO['S8_err']}")
    print(f"\nCGC/SDCG Model:")
    print(f"  μ = {CGC['mu_mcmc']} ± {CGC['mu_mcmc_err']} (MCMC)")
    print(f"  μ < {CGC['mu_upper_limit']} (95% CL with Lyα)")
    print(f"  n_g = {CGC['n_g']} ± {CGC['n_g_err']}")
    print(f"  z_trans = {CGC['z_trans']} ± {CGC['z_trans_err']}")
    print(f"\nDwarf Galaxy Predictions:")
    print(f"  Δv_strip = {DWARF['delta_v_strip']} ± {DWARF['delta_v_strip_err']} km/s")
    print(f"  Δv_observed = {DWARF['delta_v_observed']} ± {DWARF['delta_v_observed_err']} km/s")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
