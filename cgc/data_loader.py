"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CGC Data Loader Module                              ║
║                                                                              ║
║  Loads real cosmological observations and generates mock data for the       ║
║  CGC theory analysis. Supports multiple datasets:                            ║
║                                                                              ║
║    • Planck 2018 CMB TT power spectrum (2507 multipoles)                    ║
║    • BOSS DR12/16 BAO measurements                                           ║
║    • Pantheon+ Type Ia supernovae (1701 SNe)                                ║
║    • Lyman-α forest flux power spectrum                                     ║
║    • RSD growth rate measurements (fσ8)                                     ║
║    • SH0ES 2022 Cepheid-calibrated H0                                       ║
║    • Weak lensing S8 measurements                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
>>> from cgc.data_loader import DataLoader
>>> loader = DataLoader(use_real_data=True)
>>> data = loader.load_all(include_sne=True, include_lyalpha=True)

Or use convenience functions:
>>> from cgc.data_loader import load_real_data, load_pantheon_sne
>>> data = load_real_data()
>>> sne_data = load_pantheon_sne()
"""

import os
import numpy as np
from typing import Dict, Optional, Any
import warnings

from .config import PATHS, DATA_FILES, PLANCK_BASELINE, TENSIONS
from .parameters import CGCParameters


# =============================================================================
# FILE LOADING UTILITIES
# =============================================================================

def load_data_file(filepath: str, skip_comments: bool = True, 
                   max_columns: int = None) -> np.ndarray:
    """
    Load a whitespace-delimited data file.
    
    Handles comment lines (starting with #), empty lines, and various
    delimiters. Robust to minor formatting issues.
    
    Parameters
    ----------
    filepath : str
        Path to the data file.
    skip_comments : bool, default=True
        Skip lines starting with '#'.
    max_columns : int, optional
        Maximum number of columns to read (ignores extra columns).
    
    Returns
    -------
    np.ndarray
        2D array of data values.
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the file is empty or has inconsistent column counts.
    
    Examples
    --------
    >>> data = load_data_file('data/planck/planck_raw_TT.txt')
    >>> print(data.shape)
    (2507, 4)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments
            if skip_comments and line.startswith('#'):
                continue
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse values
            try:
                # Handle potential text columns at the end (survey names)
                parts = line.split()
                numeric_parts = []
                for part in parts:
                    try:
                        numeric_parts.append(float(part))
                    except ValueError:
                        break  # Stop at first non-numeric
                
                if numeric_parts:
                    if max_columns:
                        numeric_parts = numeric_parts[:max_columns]
                    data.append(numeric_parts)
            except Exception:
                continue
    
    if not data:
        raise ValueError(f"No valid data found in {filepath}")
    
    # Ensure consistent column count
    min_cols = min(len(row) for row in data)
    data = [row[:min_cols] for row in data]
    
    return np.array(data)


# =============================================================================
# PLANCK CMB DATA
# =============================================================================

def load_planck_cmb() -> Dict[str, np.ndarray]:
    """
    Load Planck 2018 CMB TT power spectrum.
    
    Attempts to load in order of preference:
    1. Raw unbinned spectrum (2507 multipoles)
    2. Plik-lite compressed data
    3. Binned approximate spectrum
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'ell': Multipole moments ℓ
        - 'Dl': D_ℓ power spectrum [μK²]
        - 'error': 1σ error bars
        - 'n_points': Number of data points
        - 'source': Data source description
    
    Notes
    -----
    The D_ℓ spectrum is defined as D_ℓ = ℓ(ℓ+1)C_ℓ/(2π).
    """
    result = {}
    
    # Try raw data first (best)
    raw_file = DATA_FILES.get('planck_raw_tt', 
                              os.path.join(PATHS['data_planck'], 'planck_raw_TT.txt'))
    
    if os.path.exists(raw_file):
        print("    📊 Loading RAW Planck 2018 TT spectrum")
        data = load_data_file(raw_file)
        
        result['ell'] = data[:, 0]
        result['Dl'] = data[:, 1]
        # Raw data often has asymmetric errors: use average
        if data.shape[1] >= 4:
            result['error'] = (np.abs(data[:, 2]) + np.abs(data[:, 3])) / 2
        else:
            result['error'] = data[:, 2]
        result['source'] = 'Planck 2018 Raw TT'
        result['n_points'] = len(data)
        
        print(f"       ✓ Loaded {len(data)} multipoles (ℓ={int(data[0,0])}-{int(data[-1,0])})")
        return result
    
    # Try plik-lite
    plik_file = DATA_FILES.get('planck_plik',
                               os.path.join(PATHS['data_planck'], 'plik_lite_v22_TT.dat'))
    
    if os.path.exists(plik_file):
        print("    📊 Loading Plik-lite TT spectrum")
        data = load_data_file(plik_file)
        
        result['ell'] = data[:, 0]
        result['Dl'] = data[:, 1]
        result['error'] = data[:, 2] if data.shape[1] > 2 else 0.05 * data[:, 1]
        result['source'] = 'Planck 2018 Plik-lite'
        result['n_points'] = len(data)
        
        print(f"       ✓ Loaded {len(data)} data points")
        return result
    
    # Fall back to binned data
    binned_file = DATA_FILES.get('planck_binned_tt',
                                 os.path.join(PATHS['data_planck'], 'planck_TT_binned.txt'))
    
    if os.path.exists(binned_file):
        print("    📊 Loading binned TT spectrum (approximate)")
        data = load_data_file(binned_file)
        
        result['ell'] = data[:, 0]
        result['Dl'] = data[:, 1]
        result['error'] = data[:, 2]
        result['source'] = 'Planck 2018 Binned (approximate)'
        result['n_points'] = len(data)
        
        print(f"       ✓ Loaded {len(data)} bins")
        return result
    
    # Generate approximate CMB if no data files found
    warnings.warn("No CMB data files found - generating approximate spectrum")
    ell = np.arange(2, 2501)
    
    # Approximate acoustic peak structure
    Dl = (5000 * np.exp(-((ell - 220)/80)**2) +
          2000 * np.exp(-((ell - 530)/100)**2) +
          1000 * np.exp(-((ell - 800)/120)**2) +
          300 * np.exp(-ell/1500) + 100)
    
    result['ell'] = ell.astype(float)
    result['Dl'] = Dl
    result['error'] = 0.02 * Dl + 10  # Approximate errors
    result['source'] = 'Approximate (no data file)'
    result['n_points'] = len(ell)
    
    return result


# =============================================================================
# BAO DATA
# =============================================================================

def load_bao_data() -> Dict[str, np.ndarray]:
    """
    Load BAO measurements from BOSS DR12 and eBOSS DR16.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'z': Effective redshifts
        - 'DV_rd': D_V/r_d measurements
        - 'error': 1σ uncertainties
        - 'DM_rd': D_M/r_d (if available)
        - 'H_rd': H(z)·r_d (if available)
        - 'surveys': Survey names
    """
    result = {'z': [], 'DV_rd': [], 'error': [], 'surveys': []}
    
    # Load BOSS DR12
    boss_file = DATA_FILES.get('boss_dr12',
                               os.path.join(PATHS['data_bao'], 'boss_dr12_consensus.txt'))
    
    if os.path.exists(boss_file):
        data = load_data_file(boss_file)
        n_boss = len(data)
        
        result['z'] = list(data[:, 0])
        result['DV_rd'] = list(data[:, 1])
        result['error'] = list(data[:, 2])
        result['surveys'] = ['BOSS DR12'] * n_boss
        
        # Additional columns if available (D_M/r_d, H*r_d)
        if data.shape[1] >= 5:
            result['DM_rd'] = list(data[:, 3])
            result['error_DM'] = list(data[:, 4])
        if data.shape[1] >= 7:
            result['H_rd'] = list(data[:, 5])
            result['error_H'] = list(data[:, 6])
        
        print(f"    📊 BOSS DR12 BAO: {n_boss} redshift bins")
    
    # Load eBOSS DR16
    eboss_file = DATA_FILES.get('eboss_dr16',
                                os.path.join(PATHS['data_bao'], 'eboss_dr16_summary.txt'))
    
    if os.path.exists(eboss_file):
        data = load_data_file(eboss_file, max_columns=4)
        n_eboss = len(data)
        
        # eBOSS format: z, measurement_type, value, error
        # For simplicity, treat as D_V/r_d approximations
        result['z'].extend(list(data[:, 0]))
        result['DV_rd'].extend(list(data[:, 2]))
        result['error'].extend(list(data[:, 3]))
        result['surveys'].extend(['eBOSS DR16'] * n_eboss)
        
        print(f"    📊 eBOSS DR16 BAO: {n_eboss} measurements")
    
    # Convert to numpy arrays
    for key in ['z', 'DV_rd', 'error']:
        result[key] = np.array(result[key])
    
    result['n_points'] = len(result['z'])
    
    if result['n_points'] == 0:
        warnings.warn("No BAO data loaded")
    
    return result


# =============================================================================
# PANTHEON+ SUPERNOVAE (FULL 1701 SNe with Covariance)
# =============================================================================

def load_pantheon_sne(use_full_cov: bool = True) -> Dict[str, np.ndarray]:
    """
    Load REAL Pantheon+ Type Ia supernova data (Brout et al. 2022, Scolnic et al. 2022).
    
    Loads the full 1701 SNe dataset with statistical+systematic covariance matrix.
    
    Parameters
    ----------
    use_full_cov : bool, default=True
        If True, load the full 1701×1701 covariance matrix.
        If False, use only diagonal errors (faster but less accurate).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'z': Redshifts in CMB frame (zHD)
        - 'z_hel': Heliocentric redshifts
        - 'mu': Distance moduli (m_b_corr)
        - 'mu_err': Diagonal uncertainties
        - 'cov': Full covariance matrix (if use_full_cov=True)
        - 'inv_cov': Inverse covariance matrix
        - 'n_sne': Number of supernovae (1701)
        - 'is_calibrator': Array indicating Cepheid host galaxies
        - 'H0_sh0es': SH0ES H0 measurement
        - 'H0_sh0es_err': SH0ES uncertainty
    
    Notes
    -----
    Reference: Brout et al. 2022, ApJ 938, 110
    The Pantheon+ sample contains 1701 light curves from 1550 unique SNe Ia.
    """
    result = {}
    
    # Load Pantheon+ data file
    pantheon_file = DATA_FILES.get('pantheon_plus')
    cov_file = DATA_FILES.get('pantheon_plus_cov')
    
    if os.path.exists(pantheon_file):
        print("    📊 Loading REAL Pantheon+ (1701 SNe)...")
        
        # Parse the Pantheon+ data file (space-delimited with header)
        # Columns: CID IDSURVEY zHD zHDERR zCMB zCMBERR zHEL zHELERR m_b_corr m_b_corr_err_DIAG 
        #          MU_SH0ES MU_SH0ES_ERR_DIAG CEPH_DIST IS_CALIBRATOR ...
        z_hd = []
        z_hel = []
        mu_obs = []
        mu_err_diag = []
        is_calibrator = []
        
        with open(pantheon_file, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 14:
                    continue
                try:
                    z_hd.append(float(parts[2]))       # zHD (CMB frame) - col 2
                    z_hel.append(float(parts[6]))      # zHEL (heliocentric) - col 6
                    mu_obs.append(float(parts[8]))     # m_b_corr (corrected apparent mag) - col 8
                    mu_err_diag.append(float(parts[9]))  # diagonal error - col 9
                    # IS_CALIBRATOR is column 13 (0-indexed)
                    is_cal = int(parts[13]) if parts[13] in ['0', '1'] else 0
                    is_calibrator.append(is_cal)
                except (ValueError, IndexError):
                    continue
        
        result['z'] = np.array(z_hd)
        result['z_hel'] = np.array(z_hel)
        result['mu'] = np.array(mu_obs)
        result['mu_err'] = np.array(mu_err_diag)
        result['is_calibrator'] = np.array(is_calibrator)
        result['n_sne'] = len(result['z'])
        result['source'] = 'Pantheon+ SH0ES (Brout et al. 2022)'
        
        print(f"       ✓ Loaded {result['n_sne']} supernovae")
        print(f"       ✓ Redshift range: z = {result['z'].min():.4f} - {result['z'].max():.4f}")
        print(f"       ✓ Cepheid calibrators: {np.sum(result['is_calibrator'])} SNe")
        
        # Load covariance matrix
        if use_full_cov and os.path.exists(cov_file):
            print(f"    📊 Loading covariance matrix (32 MB)...")
            try:
                # Covariance file format: first line is N, then N×N matrix
                with open(cov_file, 'r') as f:
                    n_cov = int(f.readline().strip())
                    cov_data = []
                    for line in f:
                        cov_data.extend([float(x) for x in line.strip().split()])
                
                cov_matrix = np.array(cov_data).reshape(n_cov, n_cov)
                
                # Truncate to match our loaded data
                n = min(n_cov, result['n_sne'])
                result['cov'] = cov_matrix[:n, :n]
                
                # Compute inverse covariance
                result['inv_cov'] = np.linalg.inv(result['cov'])
                result['has_full_cov'] = True
                
                print(f"       ✓ Covariance matrix: {n}×{n}")
                print(f"       ✓ Condition number: {np.linalg.cond(result['cov']):.2e}")
            except Exception as e:
                warnings.warn(f"Could not load covariance: {e}")
                result['cov'] = np.diag(result['mu_err']**2)
                result['inv_cov'] = np.diag(1.0 / result['mu_err']**2)
                result['has_full_cov'] = False
        else:
            # Use diagonal covariance
            result['cov'] = np.diag(result['mu_err']**2)
            result['inv_cov'] = np.diag(1.0 / result['mu_err']**2)
            result['has_full_cov'] = False
            
    else:
        warnings.warn(f"Pantheon+ data not found at {pantheon_file}")
        result['z'] = np.array([])
        result['mu'] = np.array([])
        result['mu_err'] = np.array([])
        result['n_sne'] = 0
        result['has_full_cov'] = False
    
    # Add error alias for backwards compatibility
    if 'mu_err' in result:
        result['error'] = result['mu_err']
    
    # Load SH0ES calibration (H0 from Cepheid distance ladder)
    result['H0_sh0es'] = TENSIONS['H0_sh0es']['value']
    result['H0_sh0es_err'] = TENSIONS['H0_sh0es']['error']
    print(f"    📊 SH0ES H0: {result['H0_sh0es']:.2f} ± {result['H0_sh0es_err']:.2f} km/s/Mpc")
    
    return result


# =============================================================================
# LYMAN-ALPHA FOREST (REAL eBOSS DR14 data)
# =============================================================================

def load_lyalpha_data() -> Dict[str, np.ndarray]:
    """
    Load REAL Lyman-α forest flux power spectrum from eBOSS DR14.
    
    Reference: Chabanier et al. 2019, JCAP 07 (2019) 017
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'z': Redshifts (z = 2.2 - 4.2)
        - 'k': Wavenumbers [s/km]
        - 'P_flux': Flux power spectrum
        - 'error_stat': Statistical uncertainties
        - 'error_sys': Systematic uncertainties
        - 'error': Combined error (stat + sys in quadrature)
        - 'n_points': Number of data points
    
    Notes
    -----
    The Lyman-α forest probes structure at high redshift (z ~ 2-4)
    and small scales (k ~ 0.001-0.05 s/km), providing unique 
    constraints on CGC at the transition epoch.
    """
    result = {}
    
    lyalpha_file = DATA_FILES.get('lyalpha_flux',
                                  os.path.join(PATHS['data_lyalpha'], 'eboss_lyalpha_REAL.dat'))
    
    if os.path.exists(lyalpha_file):
        data = load_data_file(lyalpha_file)
        
        result['z'] = data[:, 0]
        result['k'] = data[:, 1]
        result['P_flux'] = data[:, 2]
        
        # Handle both old format (4 cols) and new format (5 cols with stat+sys)
        if data.shape[1] >= 5:
            result['error_stat'] = data[:, 3]
            result['error_sys'] = data[:, 4]
            # Combine stat + sys in quadrature
            result['error'] = np.sqrt(data[:, 3]**2 + data[:, 4]**2)
        else:
            result['error'] = data[:, 3]
            result['error_stat'] = data[:, 3]
            result['error_sys'] = np.zeros_like(data[:, 3])
        
        result['n_points'] = len(data)
        result['source'] = 'eBOSS DR14 Lyman-α (Chabanier et al. 2019)'
        
        # Get unique redshift bins
        unique_z = np.unique(result['z'])
        print(f"    📊 Lyman-α forest: {len(data)} points")
        print(f"       ✓ Redshift bins: z = {unique_z.min():.1f} - {unique_z.max():.1f}")
        print(f"       ✓ Scale range: k = {result['k'].min():.4f} - {result['k'].max():.3f} s/km")
    else:
        warnings.warn(f"Lyman-α data not found at {lyalpha_file}")
        result['z'] = np.array([])
        result['k'] = np.array([])
        result['P_flux'] = np.array([])
        result['error'] = np.array([])
        result['n_points'] = 0
    
    return result


# =============================================================================
# GROWTH (RSD) DATA
# =============================================================================

def load_growth_data() -> Dict[str, np.ndarray]:
    """
    Load growth rate measurements from redshift-space distortions.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'z': Redshifts
        - 'fs8': fσ8 measurements
        - 'error': 1σ uncertainties
        - 'surveys': Survey names
        - 'n_points': Number of data points
    """
    result = {'z': [], 'fs8': [], 'error': [], 'surveys': []}
    
    # Try primary RSD file
    rsd_file = DATA_FILES.get('rsd',
                              os.path.join(PATHS['data_growth'], 'rsd_measurements.txt'))
    
    if os.path.exists(rsd_file):
        data = load_data_file(rsd_file, max_columns=3)
        
        result['z'] = data[:, 0]
        result['fs8'] = data[:, 1]
        result['error'] = data[:, 2]
        result['n_points'] = len(data)
        result['source'] = 'RSD compilation'
        
        print(f"    📊 RSD growth: {len(data)} measurements")
        return result
    
    # Try compilation file
    compilation_file = DATA_FILES.get('rsd_compilation',
                                      os.path.join(PATHS['data_growth'], 'rsd_compilation.txt'))
    
    if os.path.exists(compilation_file):
        data = load_data_file(compilation_file, max_columns=3)
        
        result['z'] = data[:, 0]
        result['fs8'] = data[:, 1]
        result['error'] = data[:, 2]
        result['n_points'] = len(data)
        result['source'] = 'RSD compilation'
        
        print(f"    📊 RSD compilation: {len(data)} measurements")
        return result
    
    warnings.warn("No RSD/growth data found")
    result['z'] = np.array([])
    result['fs8'] = np.array([])
    result['error'] = np.array([])
    result['n_points'] = 0
    
    return result


# =============================================================================
# H0 MEASUREMENTS
# =============================================================================

def load_h0_data() -> Dict[str, Dict[str, float]]:
    """
    Load H0 measurements from various sources.
    
    Returns
    -------
    dict
        Nested dictionary with H0 measurements:
        - 'planck': {'value': float, 'error': float}
        - 'sh0es': {'value': float, 'error': float}
        - 'trgb': {'value': float, 'error': float}
        - 'lensing': {'value': float, 'error': float}
    """
    result = {}
    
    # Planck
    result['planck'] = {
        'value': PLANCK_BASELINE['H0'],
        'error': PLANCK_BASELINE['H0_err'],
        'source': 'Planck 2018 TT,TE,EE+lowE+lensing'
    }
    
    # SH0ES (try file first, then use reference)
    sh0es_file = os.path.join(PATHS['data_sne'], 'sh0es_2022.txt')
    if os.path.exists(sh0es_file):
        data = load_data_file(sh0es_file)
        result['sh0es'] = {
            'value': data[0, 0],
            'error': data[0, 1],
            'source': 'SH0ES 2022 (Riess et al.)'
        }
    else:
        result['sh0es'] = {
            'value': TENSIONS['H0_sh0es']['value'],
            'error': TENSIONS['H0_sh0es']['error'],
            'source': 'SH0ES 2022 (reference)'
        }
    
    # TRGB
    result['trgb'] = {
        'value': TENSIONS['H0_trgb']['value'],
        'error': TENSIONS['H0_trgb']['error'],
        'source': 'TRGB (Freedman et al.)'
    }
    
    # Strong lensing
    result['lensing'] = {
        'value': TENSIONS['H0_lensing']['value'],
        'error': TENSIONS['H0_lensing']['error'],
        'source': 'H0LiCOW (time delays)'
    }
    
    print(f"    📊 H0 data: Planck={result['planck']['value']:.1f}, "
          f"SH0ES={result['sh0es']['value']:.1f}")
    
    return result


# =============================================================================
# S8 MEASUREMENTS
# =============================================================================

def load_s8_data() -> Dict[str, Dict[str, float]]:
    """
    Load S8 measurements from weak lensing and CMB.
    
    Returns
    -------
    dict
        S8 measurements from various sources.
    """
    result = {}
    
    # Planck
    result['planck'] = {
        'value': PLANCK_BASELINE['S8'],
        'error': PLANCK_BASELINE['S8_err'],
        'source': 'Planck 2018'
    }
    
    # Weak lensing surveys
    wl_file = DATA_FILES.get('weak_lensing_s8',
                             os.path.join(PATHS['data_misc'], 'weak_lensing_s8.txt'))
    
    if os.path.exists(wl_file):
        data = load_data_file(wl_file)
        result['weak_lensing'] = {
            'values': data[:, 0],
            'errors': data[:, 1],
            'mean': np.mean(data[:, 0]),
            'std': np.std(data[:, 0]),
            'source': 'Weak lensing compilation'
        }
    else:
        # Use reference values from nested TENSIONS dict
        result['weak_lensing'] = {
            'values': np.array([TENSIONS['S8_kids']['value'], TENSIONS['S8_des']['value']]),
            'errors': np.array([TENSIONS['S8_kids']['error'], TENSIONS['S8_des']['error']]),
            'mean': TENSIONS['S8_wl']['value'],
            'std': TENSIONS['S8_wl']['error'],
            'source': 'Reference values'
        }
    
    print(f"    📊 S8 data: Planck={result['planck']['value']:.3f}, "
          f"WL={result['weak_lensing']['mean']:.3f}")
    
    return result


# =============================================================================
# THEORY PREDICTIONS FOR COMPARISON
# =============================================================================

def compute_lcdm_predictions(data: Dict[str, Any], params: CGCParameters = None) -> Dict[str, np.ndarray]:
    """
    Compute ΛCDM theoretical predictions for each dataset.
    
    Parameters
    ----------
    data : dict
        Loaded data dictionary from DataLoader.
    params : CGCParameters, optional
        Parameters to use. If None, uses Planck baseline.
    
    Returns
    -------
    dict
        Dictionary with 'true_lcdm' predictions for each dataset.
    """
    if params is None:
        params = CGCParameters()
    
    predictions = {}
    
    # CMB: Approximate ΛCDM spectrum
    if 'cmb' in data and data['cmb']['n_points'] > 0:
        ell = data['cmb']['ell']
        Dl_lcdm = (5000 * np.exp(-((ell - 220)/80)**2) +
                   2000 * np.exp(-((ell - 530)/100)**2) +
                   1000 * np.exp(-((ell - 800)/120)**2) +
                   300 * np.exp(-ell/1500) + 100)
        predictions['cmb'] = Dl_lcdm
    
    # BAO: Use observed as baseline (in ΛCDM, model = observed + noise)
    if 'bao' in data and data['bao']['n_points'] > 0:
        predictions['bao'] = data['bao']['DV_rd'].copy()
    
    # Growth: fσ8(z) in ΛCDM
    if 'growth' in data and data['growth']['n_points'] > 0:
        z = data['growth']['z']
        sigma8 = PLANCK_BASELINE['sigma8']
        Omega_m = PLANCK_BASELINE['Omega_m']
        # Approximate: fσ8 ≈ σ8 × Ω_m(z)^0.55
        f_growth = Omega_m**0.55 * (1 + z)**0.05
        predictions['growth'] = sigma8 * f_growth * (1 + z)**(-0.5)
    
    return predictions


def compute_cgc_predictions(data: Dict[str, Any], params: CGCParameters) -> Dict[str, np.ndarray]:
    """
    Compute CGC theoretical predictions for each dataset.
    
    Parameters
    ----------
    data : dict
        Loaded data dictionary.
    params : CGCParameters
        CGC parameters to use.
    
    Returns
    -------
    dict
        Dictionary with 'true_cgc' predictions for each dataset.
    """
    # Get ΛCDM baseline
    lcdm = compute_lcdm_predictions(data, params)
    predictions = {}
    
    mu = params.cgc_mu
    n_g = params.cgc_n_g
    z_trans = params.cgc_z_trans
    
    # CMB: CGC enhances power at high-ℓ
    if 'cmb' in lcdm:
        ell = data['cmb']['ell']
        predictions['cmb'] = lcdm['cmb'] * (1 + mu * (ell/1000)**(n_g/2))
    
    # BAO: CGC modifies distance-redshift relation
    if 'bao' in lcdm:
        z = data['bao']['z']
        predictions['bao'] = lcdm['bao'] * (1 + mu * (1 + z)**(-n_g))
    
    # Growth: CGC modifies growth rate
    if 'growth' in lcdm:
        z = data['growth']['z']
        predictions['growth'] = lcdm['growth'] * (1 + 0.1 * mu * (1 + z)**(-n_g))
    
    return predictions


# =============================================================================
# DATA LOADER CLASS
# =============================================================================

class DataLoader:
    """
    Unified data loader for CGC cosmological analysis.
    
    This class provides a convenient interface for loading all cosmological
    datasets needed for CGC theory testing.
    
    Parameters
    ----------
    use_real_data : bool, default=True
        If True, load real observational data.
        If False, generate synthetic mock data.
    seed : int, default=42
        Random seed for mock data generation.
    verbose : bool, default=True
        Print loading progress messages.
    
    Attributes
    ----------
    data : dict
        Loaded data dictionary (after calling load_all).
    
    Examples
    --------
    Load all real data:
    >>> loader = DataLoader(use_real_data=True)
    >>> data = loader.load_all()
    
    Load with specific options:
    >>> loader = DataLoader(use_real_data=True)
    >>> data = loader.load_all(include_sne=True, include_lyalpha=True)
    
    Generate mock data:
    >>> loader = DataLoader(use_real_data=False, seed=123)
    >>> mock_data = loader.load_all()
    """
    
    def __init__(self, use_real_data: bool = True, seed: int = 42, 
                 verbose: bool = True):
        """Initialize the data loader."""
        self.use_real_data = use_real_data
        self.seed = seed
        self.verbose = verbose
        self.data = {}
        self._params = CGCParameters()
    
    def load_all(self, include_sne: bool = False, 
                 include_lyalpha: bool = False) -> Dict[str, Any]:
        """
        Load all cosmological datasets.
        
        Parameters
        ----------
        include_sne : bool, default=False
            Include Pantheon+ supernovae data.
        include_lyalpha : bool, default=False
            Include Lyman-α forest data.
        
        Returns
        -------
        dict
            Complete data dictionary with all loaded datasets.
        """
        if self.use_real_data:
            return self._load_real_data(include_sne, include_lyalpha)
        else:
            return self._generate_mock_data(include_sne, include_lyalpha)
    
    def _load_real_data(self, include_sne: bool, 
                        include_lyalpha: bool) -> Dict[str, Any]:
        """Load real cosmological data."""
        if self.verbose:
            print("\n" + "="*60)
            print("LOADING REAL COSMOLOGICAL DATA")
            print("="*60)
        
        data = {}
        
        # CMB
        if self.verbose:
            print("\n  📡 Planck CMB:")
        data['cmb'] = load_planck_cmb()
        
        # BAO
        if self.verbose:
            print("\n  📏 BAO measurements:")
        data['bao'] = load_bao_data()
        
        # Growth/RSD
        if self.verbose:
            print("\n  📈 Growth rate (RSD):")
        data['growth'] = load_growth_data()
        
        # H0
        if self.verbose:
            print("\n  🌍 H0 measurements:")
        data['H0'] = load_h0_data()
        
        # S8
        if self.verbose:
            print("\n  🌊 S8 measurements:")
        data['S8'] = load_s8_data()
        
        # Supernovae (optional)
        if include_sne:
            if self.verbose:
                print("\n  ⭐ Supernovae:")
            data['sne'] = load_pantheon_sne()
        
        # Lyman-α (optional)
        if include_lyalpha:
            if self.verbose:
                print("\n  🌌 Lyman-α forest:")
            data['lyalpha'] = load_lyalpha_data()
        
        # Add theory predictions
        if self.verbose:
            print("\n  🔬 Computing theory predictions...")
        data['lcdm_predictions'] = compute_lcdm_predictions(data, self._params)
        data['cgc_predictions'] = compute_cgc_predictions(data, self._params)
        
        # Add Planck baseline for reference
        data['planck_params'] = PLANCK_BASELINE.copy()
        data['cgc_params'] = self._params.to_dict()
        
        if self.verbose:
            print("\n" + "="*60)
            print("DATA LOADING COMPLETE")
            print("="*60)
            self._print_summary(data)
        
        self.data = data
        return data
    
    def _generate_mock_data(self, include_sne: bool, 
                           include_lyalpha: bool) -> Dict[str, Any]:
        """Generate synthetic mock data for testing."""
        np.random.seed(self.seed)
        
        if self.verbose:
            print("\n" + "="*60)
            print("GENERATING MOCK DATA")
            print("="*60)
        
        data = {}
        params = self._params
        
        # Mock CMB
        ell = np.linspace(2, 2500, 100)
        Dl_lcdm = (5000 * np.exp(-((ell - 220)/80)**2) +
                   2000 * np.exp(-((ell - 530)/100)**2) +
                   1000 * np.exp(-((ell - 800)/120)**2) +
                   300 * np.exp(-ell/1500) + 100)
        Dl_cgc = Dl_lcdm * (1 + params.cgc_mu * (ell/1000)**(params.cgc_n_g/2))
        noise = 0.02 * Dl_lcdm * np.random.randn(len(ell))
        
        data['cmb'] = {
            'ell': ell,
            'Dl': Dl_cgc + noise,
            'error': 0.02 * Dl_lcdm + 10,
            'n_points': len(ell),
            'source': 'Mock data'
        }
        
        # Mock BAO
        bao_z = np.array([0.38, 0.51, 0.61, 0.70, 1.48, 2.33])
        bao_true = np.array([8.47, 8.86, 9.44, 10.0, 12.5, 14.2])
        bao_cgc = bao_true * (1 + params.cgc_mu * (1 + bao_z)**(-params.cgc_n_g))
        
        data['bao'] = {
            'z': bao_z,
            'DV_rd': bao_cgc + 0.05 * np.random.randn(len(bao_z)),
            'error': 0.1 * np.ones_like(bao_z),
            'n_points': len(bao_z)
        }
        
        # Mock growth
        z_growth = np.array([0.02, 0.10, 0.32, 0.38, 0.51, 0.57, 0.61, 0.73, 0.86])
        fs8_lcdm = 0.43 * (1 + z_growth)**(-0.5)
        fs8_cgc = fs8_lcdm * (1 + 0.1 * params.cgc_mu * (1 + z_growth)**(-params.cgc_n_g))
        
        data['growth'] = {
            'z': z_growth,
            'fs8': fs8_cgc + 0.03 * np.random.randn(len(z_growth)),
            'error': 0.03 * np.ones_like(z_growth),
            'n_points': len(z_growth)
        }
        
        # H0
        data['H0'] = {
            'planck': {'value': 67.36 + 0.5 * np.random.randn(), 'error': 0.54},
            'sh0es': {'value': 73.04 + 1.0 * np.random.randn(), 'error': 1.04},
            'trgb': {'value': 69.8 + 1.7 * np.random.randn(), 'error': 1.7},
        }
        
        # S8
        data['S8'] = {
            'planck': {'value': 0.832, 'error': 0.013},
            'weak_lensing': {'mean': 0.77, 'std': 0.02}
        }
        
        # Mock SNe (optional)
        if include_sne:
            z_sne = np.logspace(-2, 0.3, 50)
            mu_true = 5 * np.log10((1+z_sne) * 4000 * z_sne) + 25  # Approximate
            
            data['sne'] = {
                'z': z_sne,
                'mu': mu_true + 0.1 * np.random.randn(len(z_sne)),
                'error': 0.1 * np.ones_like(z_sne),
                'n_sne': len(z_sne),
                'H0_sh0es': 73.04,
                'H0_sh0es_err': 1.04
            }
        
        # Mock Lyman-α (optional)
        if include_lyalpha:
            z_lya = np.repeat([2.2, 2.4, 3.0], 7)
            k_lya = np.tile([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1], 3)
            P_flux = 0.1 * np.exp(-k_lya / 0.02)
            
            data['lyalpha'] = {
                'z': z_lya,
                'k': k_lya,
                'P_flux': P_flux + 0.01 * np.random.randn(len(z_lya)),
                'error': 0.02 * np.ones_like(z_lya),
                'n_points': len(z_lya)
            }
        
        # Add predictions
        data['lcdm_predictions'] = compute_lcdm_predictions(data, params)
        data['cgc_predictions'] = compute_cgc_predictions(data, params)
        data['planck_params'] = PLANCK_BASELINE.copy()
        data['cgc_params'] = params.to_dict()
        
        if self.verbose:
            print("\n" + "="*60)
            print("MOCK DATA GENERATED")
            print("="*60)
            self._print_summary(data)
        
        self.data = data
        return data
    
    def _print_summary(self, data: Dict[str, Any]) -> None:
        """Print data loading summary."""
        print(f"\n  📊 Data Summary:")
        print(f"     CMB:    {data['cmb']['n_points']} multipoles")
        print(f"     BAO:    {data['bao']['n_points']} redshift bins")
        print(f"     Growth: {data['growth']['n_points']} fσ8 measurements")
        
        if 'sne' in data:
            print(f"     SNe:    {data['sne']['n_sne']} supernovae")
        if 'lyalpha' in data:
            print(f"     Lyα:    {data['lyalpha']['n_points']} points")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_real_data(params: CGCParameters = None, verbose: bool = True,
                   **kwargs) -> Dict[str, Any]:
    """
    Convenience function to load real cosmological data.
    
    Parameters
    ----------
    params : CGCParameters, optional
        Parameters for computing theory predictions.
    verbose : bool, default=True
        Print loading progress.
    **kwargs
        Additional arguments passed to DataLoader.load_all().
    
    Returns
    -------
    dict
        Complete data dictionary.
    """
    loader = DataLoader(use_real_data=True, verbose=verbose)
    if params is not None:
        loader._params = params
    return loader.load_all(**kwargs)


def load_mock_data(params: CGCParameters = None, seed: int = 42,
                   verbose: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to generate mock data.
    
    Parameters
    ----------
    params : CGCParameters, optional
        Parameters for generating mock data.
    seed : int, default=42
        Random seed.
    verbose : bool, default=True
        Print progress messages.
    **kwargs
        Additional arguments passed to DataLoader.load_all().
    
    Returns
    -------
    dict
        Mock data dictionary.
    """
    loader = DataLoader(use_real_data=False, seed=seed, verbose=verbose)
    if params is not None:
        loader._params = params
    return loader.load_all(**kwargs)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing data loader...")
    
    # Test real data loading
    print("\n" + "="*60)
    print("TEST: Loading real data")
    print("="*60)
    
    try:
        data = load_real_data(include_sne=True)
        print("\n✓ Real data loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading real data: {e}")
    
    # Test mock data generation
    print("\n" + "="*60)
    print("TEST: Generating mock data")
    print("="*60)
    
    mock_data = load_mock_data(include_sne=True, include_lyalpha=True)
    print("\n✓ Mock data generated successfully")
