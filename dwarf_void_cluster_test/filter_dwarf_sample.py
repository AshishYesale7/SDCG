#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SDCG FALSIFICATION TEST: Void vs Cluster Dwarf Galaxies            ║
║                                                                              ║
║  Phase 2: Filter Dwarf Galaxy Sample & Classify Environments                ║
║                                                                              ║
║  Method:                                                                     ║
║    1. Select dwarf galaxies: 10^7 < M_HI < 10^9.5 M_sun                     ║
║    2. Compute rotation velocity: v_rot = W50 / (2 × sin(i))                 ║
║    3. Classify environment using LOCAL DENSITY from ALFALFA itself          ║
║       + proximity to Abell clusters                                         ║
║                                                                              ║
║  Environment Classification:                                                 ║
║    VOID:    Local density < 0.3 × mean AND far from clusters (>5 Mpc)       ║
║    CLUSTER: Within 3 Mpc of Abell cluster OR high local density             ║
║    FIELD:   Everything else                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = Path(__file__).parent / "filtered"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================

# Mass cuts (in log10(M_HI/M_sun))
LOG_MHI_MIN = 7.0    # 10^7 M_sun - minimum for reliable rotation
LOG_MHI_MAX = 9.5    # 10^9.5 M_sun - above this, self-screening kicks in

# Environment classification thresholds
VOID_DENSITY_PERCENTILE = 20     # Bottom 20% in local density = void
CLUSTER_DISTANCE_MPC = 3.0       # Within 3 Mpc of Abell cluster = cluster
CLUSTER_DENSITY_PERCENTILE = 80  # Top 20% in local density = cluster

# Data quality cuts
W50_MIN = 30       # km/s - minimum reliable line width
W50_MAX = 400      # km/s - maximum for dwarfs
W50_ERROR_MAX = 50 # km/s - maximum error

# Cosmology
H0 = 70.0  # km/s/Mpc
C_LIGHT = 299792.458  # km/s


def sexagesimal_to_deg(coord_str):
    """Convert sexagesimal RA or Dec string to degrees."""
    parts = coord_str.strip().split()
    if len(parts) == 3:
        h_or_d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        
        # Check if negative (for Dec)
        sign = -1 if h_or_d < 0 or coord_str.strip().startswith('-') else 1
        h_or_d = abs(h_or_d)
        
        return sign * (h_or_d + m/60 + s/3600)
    return float(coord_str)


def load_alfalfa() -> pd.DataFrame:
    """Load ALFALFA α.100 catalog."""
    path = RAW_DIR / "alfalfa_a100.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"ALFALFA catalog not found: {path}")
    
    print(f"Loading ALFALFA: {path}")
    df = pd.read_csv(path)
    print(f"  Total sources: {len(df)}")
    
    # Parse RA/Dec from sexagesimal format (HH MM SS.s, DD MM SS.s)
    print("  Parsing coordinates...")
    
    # Convert RA (hours to degrees)
    def parse_ra(ra_str):
        try:
            parts = str(ra_str).strip().split()
            if len(parts) >= 3:
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                return 15 * (h + m/60 + s/3600)  # 15 deg/hour
            return np.nan
        except:
            return np.nan
    
    # Convert Dec (degrees)
    def parse_dec(dec_str):
        try:
            parts = str(dec_str).strip().split()
            if len(parts) >= 3:
                d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                sign = -1 if str(dec_str).strip().startswith('-') else 1
                d = abs(d)
                return sign * (d + m/60 + s/3600)
            return np.nan
        except:
            return np.nan
    
    df['RA_deg'] = df['RAJ2000'].apply(parse_ra)
    df['Dec_deg'] = df['DEJ2000'].apply(parse_dec)
    
    # Calculate redshift from Vhel
    df['z'] = df['Vhel'] / C_LIGHT
    
    # Calculate distance in Mpc (Hubble flow)
    df['Dist_Mpc'] = df['Vhel'] / H0
    
    # Use provided distance if available
    if 'Dist' in df.columns:
        mask = df['Dist'].notna() & (df['Dist'] > 0)
        df.loc[mask, 'Dist_Mpc'] = df.loc[mask, 'Dist']
    
    # Clean up
    valid = df['RA_deg'].notna() & df['Dec_deg'].notna() & (df['Vhel'] > 0)
    df = df[valid].copy()
    print(f"  Valid sources with coordinates: {len(df)}")
    
    return df


def load_clusters() -> pd.DataFrame:
    """Load Abell cluster catalog."""
    path = RAW_DIR / "sdss_clusters.fits"
    
    if not path.exists():
        print("  No cluster catalog found - will use density only")
        return None
    
    print(f"Loading Abell clusters: {path}")
    
    try:
        with fits.open(path) as hdul:
            data = hdul[1].data
            
            # Parse RA (HH MM SS.s to degrees)
            def parse_ra_hms(ra_str):
                try:
                    parts = str(ra_str).strip().split()
                    if len(parts) >= 3:
                        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                        return 15 * (h + m/60 + s/3600)  # 15 deg/hour
                    return np.nan
                except:
                    return np.nan
            
            # Parse Dec (DD MM SS to degrees)
            def parse_dec_dms(dec_str):
                try:
                    s = str(dec_str).strip()
                    sign = -1 if s.startswith('-') else 1
                    parts = s.replace('-', ' ').replace('+', ' ').split()
                    if len(parts) >= 3:
                        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
                        return sign * (d + m/60 + sec/3600)
                    return np.nan
                except:
                    return np.nan
            
            ra_deg = np.array([parse_ra_hms(r) for r in data['_RA.icrs']])
            dec_deg = np.array([parse_dec_dms(d) for d in data['_DE.icrs']])
            
            # Handle redshift - fix byte order
            z_arr = np.array(data['z'])
            if z_arr.dtype.byteorder == '>':
                z_arr = z_arr.byteswap().newbyteorder()
            z_arr = z_arr.astype(float)
            
            df = pd.DataFrame({
                'ACO': np.array(data['ACO']),
                'RA_deg': ra_deg,
                'Dec_deg': dec_deg,
                'z': z_arr,
                'Richness': np.array(data['Rich']) if 'Rich' in data.names else 0
            })
        
        # Filter clusters with valid redshifts and coordinates
        valid = (
            df['z'].notna() & (df['z'] > 0) & (df['z'] < 0.2) &
            df['RA_deg'].notna() & df['Dec_deg'].notna()
        )
        df = df[valid].copy()
        
        # Calculate distance
        df['Dist_Mpc'] = C_LIGHT * df['z'] / H0
        
        print(f"  Clusters with valid z < 0.2: {len(df)}")
        return df
        
    except Exception as e:
        print(f"  Error loading clusters: {e}")
        import traceback
        traceback.print_exc()
        print("  Will use density-only classification")
        return None


def compute_local_density(galaxies: pd.DataFrame, k_neighbors: int = 10) -> np.ndarray:
    """
    Compute local galaxy density using k-th nearest neighbor distance.
    
    ρ ∝ 1 / d_k^3  where d_k is distance to k-th nearest neighbor
    """
    print(f"\nComputing local density (k={k_neighbors} neighbors)...")
    
    # Convert to 3D comoving coordinates
    dist = galaxies['Dist_Mpc'].values
    ra = np.radians(galaxies['RA_deg'].values)
    dec = np.radians(galaxies['Dec_deg'].values)
    
    x = dist * np.cos(dec) * np.cos(ra)
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    
    coords = np.column_stack([x, y, z])
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Find k-th nearest neighbor distance
    distances, _ = tree.query(coords, k=k_neighbors+1)
    d_k = distances[:, -1]  # k-th neighbor distance (excluding self)
    
    # Local density proxy (avoid division by zero)
    d_k = np.maximum(d_k, 0.1)  # Minimum 0.1 Mpc
    local_density = 1.0 / d_k**3
    
    print(f"  Density range: {local_density.min():.2e} to {local_density.max():.2e}")
    
    return local_density


def classify_environment(galaxies: pd.DataFrame, clusters: pd.DataFrame = None) -> pd.DataFrame:
    """
    Classify galaxies into void, cluster, or field environments.
    
    Method:
    1. Compute local galaxy density from ALFALFA
    2. Check proximity to Abell clusters
    3. Combine both criteria
    """
    print("\nClassifying environments...")
    
    galaxies = galaxies.copy()
    
    # Compute local density
    local_density = compute_local_density(galaxies)
    galaxies['local_density'] = local_density
    
    # Density percentiles
    void_threshold = np.percentile(local_density, VOID_DENSITY_PERCENTILE)
    cluster_threshold = np.percentile(local_density, CLUSTER_DENSITY_PERCENTILE)
    
    print(f"  Void threshold (p{VOID_DENSITY_PERCENTILE}): {void_threshold:.2e}")
    print(f"  Cluster threshold (p{CLUSTER_DENSITY_PERCENTILE}): {cluster_threshold:.2e}")
    
    # Initialize as field
    galaxies['environment'] = 'field'
    galaxies['cluster_dist_Mpc'] = np.nan
    
    # Check cluster proximity if available
    if clusters is not None and len(clusters) > 0:
        print(f"  Checking proximity to {len(clusters)} Abell clusters...")
        
        # Galaxy coordinates
        gal_coords = SkyCoord(
            ra=galaxies['RA_deg'].values * u.deg,
            dec=galaxies['Dec_deg'].values * u.deg
        )
        
        # Cluster coordinates
        cluster_coords = SkyCoord(
            ra=clusters['RA_deg'].values * u.deg,
            dec=clusters['Dec_deg'].values * u.deg
        )
        
        # For each galaxy, find distance to nearest cluster
        gal_dist = galaxies['Dist_Mpc'].values
        
        for i in range(len(galaxies)):
            # Angular separation to all clusters
            sep = gal_coords[i].separation(cluster_coords).deg
            
            # Physical distance (simplified - assumes similar redshift)
            cluster_dist_mpc = clusters['Dist_Mpc'].values
            gal_d = gal_dist[i]
            
            # 3D distance approximation
            angular_dist_mpc = sep * np.pi / 180 * gal_d
            radial_dist_mpc = np.abs(cluster_dist_mpc - gal_d)
            total_dist = np.sqrt(angular_dist_mpc**2 + radial_dist_mpc**2)
            
            min_dist = np.min(total_dist)
            galaxies.loc[galaxies.index[i], 'cluster_dist_Mpc'] = min_dist
            
            if min_dist < CLUSTER_DISTANCE_MPC:
                galaxies.loc[galaxies.index[i], 'environment'] = 'cluster'
    
    # Now apply density-based classification for non-cluster galaxies
    # Low density + far from clusters = VOID
    void_mask = (
        (galaxies['local_density'] < void_threshold) & 
        (galaxies['environment'] != 'cluster')
    )
    if clusters is not None:
        void_mask &= (galaxies['cluster_dist_Mpc'] > 5.0)  # At least 5 Mpc from cluster
    
    galaxies.loc[void_mask, 'environment'] = 'void'
    
    # High density = CLUSTER (even if not near Abell)
    high_density_mask = (
        (galaxies['local_density'] > cluster_threshold) &
        (galaxies['environment'] == 'field')
    )
    galaxies.loc[high_density_mask, 'environment'] = 'cluster'
    
    # Summary
    env_counts = galaxies['environment'].value_counts()
    print(f"\n  Environment classification:")
    for env, count in env_counts.items():
        pct = 100 * count / len(galaxies)
        print(f"    {env:10s}: {count:5d} ({pct:.1f}%)")
    
    return galaxies


def compute_rotation_velocity(galaxies: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rotation velocity from W50 line width.
    
    v_rot = W50 / (2 × sin(i))
    
    For random inclinations: <sin(i)> = π/4 ≈ 0.785
    """
    galaxies = galaxies.copy()
    
    # Average sin(i) for randomly oriented disks
    sin_i_avg = np.pi / 4
    
    # v_rot from W50
    galaxies['v_rot'] = galaxies['W50'] / (2 * sin_i_avg)
    
    # Error propagation
    if 'e_W50' in galaxies.columns:
        galaxies['e_v_rot'] = galaxies['e_W50'] / (2 * sin_i_avg)
    else:
        galaxies['e_v_rot'] = 0.1 * galaxies['v_rot']  # 10% default error
    
    return galaxies


def apply_quality_cuts(galaxies: pd.DataFrame) -> pd.DataFrame:
    """Apply data quality cuts."""
    print("\nApplying quality cuts...")
    
    initial = len(galaxies)
    
    # W50 range
    mask = (galaxies['W50'] >= W50_MIN) & (galaxies['W50'] <= W50_MAX)
    galaxies = galaxies[mask].copy()
    print(f"  W50 in [{W50_MIN}, {W50_MAX}] km/s: {len(galaxies)}/{initial}")
    
    # W50 error
    if 'e_W50' in galaxies.columns:
        mask = galaxies['e_W50'] < W50_ERROR_MAX
        galaxies = galaxies[mask].copy()
        print(f"  e_W50 < {W50_ERROR_MAX} km/s: {len(galaxies)}")
    
    # Positive distance
    mask = galaxies['Dist_Mpc'] > 0
    galaxies = galaxies[mask].copy()
    print(f"  Dist > 0: {len(galaxies)}")
    
    return galaxies


def apply_mass_cuts(galaxies: pd.DataFrame) -> pd.DataFrame:
    """Apply HI mass cuts for dwarf galaxies."""
    print("\nApplying mass cuts (dwarf selection)...")
    
    initial = len(galaxies)
    
    # HI mass range
    mask = (galaxies['logMHI'] >= LOG_MHI_MIN) & (galaxies['logMHI'] <= LOG_MHI_MAX)
    galaxies = galaxies[mask].copy()
    
    print(f"  {LOG_MHI_MIN} < log(M_HI/M_sun) < {LOG_MHI_MAX}: {len(galaxies)}/{initial}")
    
    return galaxies


def main():
    """Main filtering pipeline."""
    print("=" * 70)
    print("SDCG FALSIFICATION TEST: FILTER DWARF SAMPLE")
    print("=" * 70)
    print()
    
    # Load data
    print("PHASE 2A: Loading Data")
    print("-" * 70)
    
    galaxies = load_alfalfa()
    clusters = load_clusters()
    
    # Apply quality cuts
    print("\nPHASE 2B: Quality Cuts")
    print("-" * 70)
    
    galaxies = apply_quality_cuts(galaxies)
    
    # Apply mass cuts (dwarf selection)
    galaxies = apply_mass_cuts(galaxies)
    
    # Compute rotation velocity
    print("\nPHASE 2C: Compute Rotation Velocities")
    print("-" * 70)
    
    galaxies = compute_rotation_velocity(galaxies)
    print(f"  v_rot range: {galaxies['v_rot'].min():.1f} - {galaxies['v_rot'].max():.1f} km/s")
    print(f"  v_rot mean: {galaxies['v_rot'].mean():.1f} ± {galaxies['v_rot'].std():.1f} km/s")
    
    # Classify environments
    print("\nPHASE 2D: Environment Classification")
    print("-" * 70)
    
    galaxies = classify_environment(galaxies, clusters)
    
    # Split by environment
    void_dwarfs = galaxies[galaxies['environment'] == 'void'].copy()
    cluster_dwarfs = galaxies[galaxies['environment'] == 'cluster'].copy()
    field_dwarfs = galaxies[galaxies['environment'] == 'field'].copy()
    
    # Save filtered samples
    print("\nPHASE 2E: Saving Filtered Samples")
    print("-" * 70)
    
    void_dwarfs.to_csv(OUTPUT_DIR / "void_dwarfs.csv", index=False)
    cluster_dwarfs.to_csv(OUTPUT_DIR / "cluster_dwarfs.csv", index=False)
    field_dwarfs.to_csv(OUTPUT_DIR / "field_dwarfs.csv", index=False)
    
    print(f"  Saved: void_dwarfs.csv ({len(void_dwarfs)} galaxies)")
    print(f"  Saved: cluster_dwarfs.csv ({len(cluster_dwarfs)} galaxies)")
    print(f"  Saved: field_dwarfs.csv ({len(field_dwarfs)} galaxies)")
    
    # Summary statistics
    print()
    print("=" * 70)
    print("SAMPLE SUMMARY")
    print("=" * 70)
    print()
    
    print(f"  VOID DWARFS:")
    print(f"    N = {len(void_dwarfs)}")
    print(f"    <v_rot> = {void_dwarfs['v_rot'].mean():.1f} ± {void_dwarfs['v_rot'].std()/np.sqrt(len(void_dwarfs)):.1f} km/s")
    print(f"    <log M_HI> = {void_dwarfs['logMHI'].mean():.2f}")
    print()
    
    print(f"  CLUSTER DWARFS:")
    print(f"    N = {len(cluster_dwarfs)}")
    print(f"    <v_rot> = {cluster_dwarfs['v_rot'].mean():.1f} ± {cluster_dwarfs['v_rot'].std()/np.sqrt(len(cluster_dwarfs)):.1f} km/s")
    print(f"    <log M_HI> = {cluster_dwarfs['logMHI'].mean():.2f}")
    print()
    
    # Preliminary Δv
    if len(void_dwarfs) > 0 and len(cluster_dwarfs) > 0:
        delta_v = void_dwarfs['v_rot'].mean() - cluster_dwarfs['v_rot'].mean()
        se = np.sqrt(
            void_dwarfs['v_rot'].std()**2 / len(void_dwarfs) +
            cluster_dwarfs['v_rot'].std()**2 / len(cluster_dwarfs)
        )
        print(f"  PRELIMINARY Δv:")
        print(f"    Δv = {delta_v:+.2f} ± {se:.2f} km/s")
        print()
        print(f"  SDCG Predictions:")
        print(f"    Unconstrained (μ=0.41): +34 km/s")
        print(f"    MCMC best-fit (μ=0.149): +12 ± 3 km/s")
        print(f"    ΛCDM: 0 km/s")
    
    print()
    print("Next step: python run_void_cluster_analysis.py")


if __name__ == "__main__":
    main()
