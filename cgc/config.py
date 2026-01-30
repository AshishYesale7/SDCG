"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CGC Configuration Module                             ║
║                                                                              ║
║  This module contains all configuration settings, paths, and constants       ║
║  used throughout the CGC analysis framework.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Configuration Categories
------------------------
1. PATHS : Directory structure for data, results, and plots
2. CONSTANTS : Physical constants and unit conversions
3. PLANCK_BASELINE : Planck 2018 best-fit ΛCDM parameters
4. MCMC_DEFAULTS : Default MCMC sampler settings
5. PLOT_SETTINGS : Matplotlib styling for publication figures

Usage
-----
>>> from cgc.config import PATHS, CONSTANTS
>>> print(PATHS['data'])  # Get data directory path
>>> print(CONSTANTS['c'])  # Speed of light in km/s
"""

import os
import numpy as np

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

# Base directory - automatically detected from this file's location
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Project directory paths
PATHS = {
    # Root project directory
    'base': _BASE_DIR,
    
    # Data directories
    'data': os.path.join(_BASE_DIR, "data"),
    'data_planck': os.path.join(_BASE_DIR, "data", "planck"),
    'data_bao': os.path.join(_BASE_DIR, "data", "bao"),
    'data_sne': os.path.join(_BASE_DIR, "data", "sne"),
    'data_lyalpha': os.path.join(_BASE_DIR, "data", "lyalpha"),
    'data_growth': os.path.join(_BASE_DIR, "data", "growth"),
    'data_misc': os.path.join(_BASE_DIR, "data", "misc"),
    'data_cgc_sim': os.path.join(_BASE_DIR, "data", "cgc_simulations"),
    
    # Output directories
    'results': os.path.join(_BASE_DIR, "results"),
    'plots': os.path.join(_BASE_DIR, "plots"),
    'chains': os.path.join(_BASE_DIR, "results", "chains"),
    
    # Code directories
    'scripts': os.path.join(_BASE_DIR, "scripts"),
    'class_cgc': os.path.join(_BASE_DIR, "class_cgc"),
    'cgc_theory': os.path.join(_BASE_DIR, "cgc_theory"),
}


def setup_directories():
    """
    Create all necessary directories for the analysis.
    
    This function is idempotent - it can be called multiple times safely.
    It creates directories with proper permissions for writing results.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> from cgc.config import setup_directories
    >>> setup_directories()  # Creates all directories
    """
    for name, path in PATHS.items():
        if not name.startswith('data'):  # Don't create data subdirs automatically
            os.makedirs(path, exist_ok=True)
    
    # Create output subdirectories
    os.makedirs(PATHS['chains'], exist_ok=True)
    
    print(f"✓ Working directory: {PATHS['base']}")


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

CONSTANTS = {
    # Speed of light [km/s]
    'c': 299792.458,
    
    # Gravitational constant [m³ kg⁻¹ s⁻²]
    'G': 6.67430e-11,
    
    # Reduced Planck constant [J·s]
    'hbar': 1.054571817e-34,
    
    # Boltzmann constant [J/K]
    'k_B': 1.380649e-23,
    
    # CMB temperature today [K]
    'T_cmb': 2.7255,
    
    # Critical density [h² Msun/Mpc³]
    'rho_crit_h2': 2.775e11,
    
    # Megaparsec in meters
    'Mpc_in_m': 3.0857e22,
    
    # Solar mass in kg
    'Msun_kg': 1.989e30,
    
    # Seconds in a year
    'year_s': 3.1557e7,
    
    # Sound horizon at drag epoch (approximate) [Mpc]
    'r_d_fid': 147.09,
}


# =============================================================================
# PLANCK 2018 BASELINE COSMOLOGY
# =============================================================================

PLANCK_BASELINE = {
    # ═══════════════════════════════════════════════════════════════════════
    # Primary parameters (TT,TE,EE+lowE+lensing)
    # Reference: Planck Collaboration 2018, Table 2, Column 6
    # ═══════════════════════════════════════════════════════════════════════
    
    # Baryon density parameter
    'omega_b': 0.02237,          # Ω_b h²
    'omega_b_err': 0.00015,
    
    # Cold dark matter density parameter
    'omega_cdm': 0.1200,         # Ω_c h²
    'omega_cdm_err': 0.0012,
    
    # Hubble parameter
    'h': 0.6736,                 # H0 / (100 km/s/Mpc)
    'h_err': 0.0054,
    'H0': 67.36,                 # [km/s/Mpc]
    'H0_err': 0.54,
    
    # Primordial power spectrum amplitude
    'ln10As': 3.044,             # ln(10¹⁰ A_s)
    'ln10As_err': 0.014,
    'As': 2.1e-9,                # A_s
    
    # Scalar spectral index
    'n_s': 0.9649,
    'n_s_err': 0.0042,
    
    # Optical depth to reionization
    'tau_reio': 0.0544,
    'tau_reio_err': 0.0073,
    
    # ═══════════════════════════════════════════════════════════════════════
    # Derived parameters
    # ═══════════════════════════════════════════════════════════════════════
    
    # Matter density
    'Omega_m': 0.3153,
    'Omega_m_err': 0.0073,
    
    # Dark energy density (assuming flat universe)
    'Omega_Lambda': 0.6847,
    'Omega_Lambda_err': 0.0073,
    
    # Amplitude of matter fluctuations
    'sigma8': 0.8111,
    'sigma8_err': 0.0060,
    
    # S8 parameter (weak lensing combination)
    'S8': 0.832,                 # σ8 (Ω_m/0.3)^0.5
    'S8_err': 0.013,
    
    # Age of the universe [Gyr]
    'age': 13.797,
    'age_err': 0.023,
    
    # Sound horizon at recombination [Mpc]
    'r_star': 144.43,
    'r_star_err': 0.26,
    
    # Sound horizon at drag epoch [Mpc]
    'r_d': 147.09,
    'r_d_err': 0.26,
}


# =============================================================================
# TENSION REFERENCE VALUES
# =============================================================================

TENSIONS = {
    # ═══════════════════════════════════════════════════════════════════════
    # Hubble Tension (H0) [km/s/Mpc]
    # ═══════════════════════════════════════════════════════════════════════
    
    # Planck 2018 (CMB-inferred)
    'H0_planck': {'value': 67.36, 'error': 0.54},
    
    # SH0ES 2022 (Riess et al.) - Cepheid-calibrated
    'H0_sh0es': {'value': 73.04, 'error': 1.04},
    
    # TRGB (Freedman et al. 2021)
    'H0_trgb': {'value': 69.8, 'error': 1.7},
    
    # H0LiCOW (strong lensing time delays)
    'H0_lensing': {'value': 73.3, 'error': 1.8},
    
    # ═══════════════════════════════════════════════════════════════════════
    # S8 Tension (σ8 × √(Ω_m/0.3))
    # ═══════════════════════════════════════════════════════════════════════
    
    # Planck 2018 (CMB-inferred)
    'S8_planck': {'value': 0.832, 'error': 0.013},
    
    # Weak lensing combined (KiDS + DES + HSC)
    'S8_wl': {'value': 0.770, 'error': 0.015},
    
    # KiDS-1000 (weak lensing)
    'S8_kids': {'value': 0.759, 'error': 0.024},
    
    # DES Y3 (weak lensing)
    'S8_des': {'value': 0.776, 'error': 0.017},
}


# =============================================================================
# MCMC DEFAULT SETTINGS
# =============================================================================

MCMC_DEFAULTS = {
    # ═══════════════════════════════════════════════════════════════════════
    # Sampler Configuration
    # ═══════════════════════════════════════════════════════════════════════
    
    # Number of walkers (should be ≥ 2 × n_params)
    'n_walkers': 32,
    
    # Number of steps (quick test)
    'n_steps_test': 500,
    
    # Number of steps (standard run)
    'n_steps_standard': 1000,
    
    # Number of steps (publication quality)
    'n_steps_publication': 10000,
    
    # Burn-in fraction (fraction of chain to discard)
    'burn_in_fraction': 0.2,
    
    # Thinning factor (take every nth sample)
    'thin': 10,
    
    # Random seed for reproducibility
    'seed': 42,
    
    # ═══════════════════════════════════════════════════════════════════════
    # Convergence Criteria
    # ═══════════════════════════════════════════════════════════════════════
    
    # Gelman-Rubin R-hat threshold (< 1.1 is good)
    'rhat_threshold': 1.1,
    
    # Minimum effective sample size per parameter
    'min_ess': 100,
    
    # Autocorrelation length multiplier for burn-in
    'autocorr_multiplier': 50,
}


# =============================================================================
# PLOTTING SETTINGS
# =============================================================================

PLOT_SETTINGS = {
    # ═══════════════════════════════════════════════════════════════════════
    # Figure sizes [inches]
    # ═══════════════════════════════════════════════════════════════════════
    'fig_single': (8, 6),
    'fig_double': (14, 6),
    'fig_quad': (14, 10),
    'fig_corner': (12, 12),
    'fig_dashboard': (16, 12),
    
    # ═══════════════════════════════════════════════════════════════════════
    # Resolution
    # ═══════════════════════════════════════════════════════════════════════
    'dpi_screen': 100,
    'dpi_publication': 300,
    
    # ═══════════════════════════════════════════════════════════════════════
    # Colors
    # ═══════════════════════════════════════════════════════════════════════
    'color_lcdm': '#1f77b4',      # Blue
    'color_cgc': '#d62728',       # Red
    'color_planck': '#1f77b4',    # Blue
    'color_shoes': '#ff7f0e',     # Orange
    'color_wl': '#2ca02c',        # Green
    'color_data': '#7f7f7f',      # Gray
    
    # ═══════════════════════════════════════════════════════════════════════
    # Fonts
    # ═══════════════════════════════════════════════════════════════════════
    'font_size_title': 14,
    'font_size_label': 12,
    'font_size_tick': 10,
    'font_size_legend': 10,
    
    # ═══════════════════════════════════════════════════════════════════════
    # Style
    # ═══════════════════════════════════════════════════════════════════════
    'grid_alpha': 0.3,
    'scatter_alpha': 0.3,
    'fill_alpha': 0.3,
    'line_width': 2,
}


def configure_matplotlib():
    """
    Configure matplotlib for publication-quality figures.
    
    Sets up fonts, sizes, and styles for consistent plotting across
    all analysis outputs.
    
    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': PLOT_SETTINGS['font_size_label'],
        'axes.titlesize': PLOT_SETTINGS['font_size_title'],
        'axes.labelsize': PLOT_SETTINGS['font_size_label'],
        'xtick.labelsize': PLOT_SETTINGS['font_size_tick'],
        'ytick.labelsize': PLOT_SETTINGS['font_size_tick'],
        'legend.fontsize': PLOT_SETTINGS['font_size_legend'],
        'figure.dpi': PLOT_SETTINGS['dpi_screen'],
        'savefig.dpi': PLOT_SETTINGS['dpi_publication'],
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': PLOT_SETTINGS['grid_alpha'],
    })


# =============================================================================
# DATA FILE PATHS
# =============================================================================

DATA_FILES = {
    # ═══════════════════════════════════════════════════════════════════════
    # Planck CMB data
    # ═══════════════════════════════════════════════════════════════════════
    'planck_raw_tt': os.path.join(PATHS['data_planck'], 'planck_raw_TT.txt'),
    'planck_binned_tt': os.path.join(PATHS['data_planck'], 'planck_TT_binned.txt'),
    'planck_plik': os.path.join(PATHS['data_planck'], 'plik_lite_v22_TT.dat'),
    
    # ═══════════════════════════════════════════════════════════════════════
    # BAO data
    # ═══════════════════════════════════════════════════════════════════════
    'boss_dr12': os.path.join(PATHS['data_bao'], 'boss_dr12_consensus.txt'),
    'eboss_dr16': os.path.join(PATHS['data_bao'], 'eboss_dr16_summary.txt'),
    
    # ═══════════════════════════════════════════════════════════════════════
    # Supernova data (REAL Pantheon+ with full covariance)
    # ═══════════════════════════════════════════════════════════════════════
    'pantheon_plus': os.path.join(PATHS['data_sne'], 'pantheon_plus', 'Pantheon+SH0ES.dat'),
    'pantheon_plus_cov': os.path.join(PATHS['data_sne'], 'pantheon_plus', 'Pantheon+SH0ES_STAT+SYS.cov'),
    'pantheon_ceph': os.path.join(PATHS['data_sne'], 'pantheon_plus', 'ceph_distances.dat'),
    'sh0es': os.path.join(PATHS['data_sne'], 'sh0es_2022.txt'),
    
    # ═══════════════════════════════════════════════════════════════════════
    # Growth/RSD data
    # ═══════════════════════════════════════════════════════════════════════
    'rsd': os.path.join(PATHS['data_growth'], 'rsd_measurements.txt'),
    'rsd_compilation': os.path.join(PATHS['data_growth'], 'rsd_compilation.txt'),
    
    # ═══════════════════════════════════════════════════════════════════════
    # Lyman-α data (REAL eBOSS DR14 flux power spectrum)
    # ═══════════════════════════════════════════════════════════════════════
    'lyalpha_flux': os.path.join(PATHS['data_lyalpha'], 'eboss_lyalpha_REAL.dat'),
    
    # ═══════════════════════════════════════════════════════════════════════
    # Reference parameters
    # ═══════════════════════════════════════════════════════════════════════
    'planck_params': os.path.join(PATHS['data_misc'], 'planck_params_simple.txt'),
    'weak_lensing_s8': os.path.join(PATHS['data_misc'], 'weak_lensing_s8.txt'),
    
    # ═══════════════════════════════════════════════════════════════════════
    # CGC simulations
    # ═══════════════════════════════════════════════════════════════════════
    'hubble_evolution': os.path.join(PATHS['data_cgc_sim'], 'hubble_evolution.txt'),
    'growth_evolution': os.path.join(PATHS['data_cgc_sim'], 'growth_evolution.txt'),
}


# =============================================================================
# PRINT CONFIGURATION (for debugging)
# =============================================================================

def print_config():
    """
    Print current configuration for debugging purposes.
    
    Displays all paths, constants, and settings currently in use.
    """
    print("\n" + "="*70)
    print("CGC CONFIGURATION")
    print("="*70)
    
    print("\n📁 PATHS:")
    for name, path in PATHS.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"   {exists} {name:20s} : {path}")
    
    print("\n🔬 PLANCK BASELINE:")
    for key in ['H0', 'Omega_m', 'sigma8', 'S8']:
        val = PLANCK_BASELINE[key]
        err = PLANCK_BASELINE.get(f'{key}_err', 0)
        print(f"   {key:10s} = {val:.4f} ± {err:.4f}")
    
    print("\n⚙️  MCMC DEFAULTS:")
    for key, val in MCMC_DEFAULTS.items():
        print(f"   {key:25s} : {val}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print_config()
