#!/usr/bin/env python3
"""
CORRECTED Analysis: 3D Distance to Abell Clusters
==================================================
Uses proper coordinate parsing and 3D distances.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from pathlib import Path
import sys

# Setup paths - find project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data paths
DATA_DIR = PROJECT_ROOT / 'data'
LOCAL_DATA_DIR = SCRIPT_DIR / 'data'

def parse_sexagesimal_coords(alfalfa):
    """Parse RA/Dec from various formats to degrees."""
    # Check if we already have degree columns
    if 'RAdeg_HI' in alfalfa.columns and 'Decdeg_HI' in alfalfa.columns:
        alfalfa = alfalfa.copy()
        alfalfa['RA_deg'] = alfalfa['RAdeg_HI']
        alfalfa['Dec_deg'] = alfalfa['Decdeg_HI']
        valid_mask = ~(alfalfa['RA_deg'].isna() | alfalfa['Dec_deg'].isna())
        return alfalfa[valid_mask].reset_index(drop=True)
    
    # Otherwise try sexagesimal format
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    ra_deg = []
    dec_deg = []
    valid_mask = []
    
    for i, row in alfalfa.iterrows():
        try:
            ra_str = row['RAJ2000']
            dec_str = row['DEJ2000']
            coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
            ra_deg.append(coord.ra.deg)
            dec_deg.append(coord.dec.deg)
            valid_mask.append(True)
        except:
            ra_deg.append(np.nan)
            dec_deg.append(np.nan)
            valid_mask.append(False)
    
    alfalfa = alfalfa.copy()
    alfalfa['RA_deg'] = ra_deg
    alfalfa['Dec_deg'] = dec_deg
    return alfalfa[valid_mask].reset_index(drop=True)

def compute_3d_distance_to_clusters(gal_ra, gal_dec, gal_dist, 
                                     cluster_ra, cluster_dec, cluster_dist):
    """
    Compute 3D comoving distance from each galaxy to nearest cluster.
    Uses Cartesian approximation (valid for small areas).
    """
    # Galaxy Cartesian coords
    gal_x = gal_dist * np.cos(np.radians(gal_dec)) * np.cos(np.radians(gal_ra))
    gal_y = gal_dist * np.cos(np.radians(gal_dec)) * np.sin(np.radians(gal_ra))
    gal_z = gal_dist * np.sin(np.radians(gal_dec))
    
    # Cluster Cartesian coords
    cl_x = cluster_dist * np.cos(np.radians(cluster_dec)) * np.cos(np.radians(cluster_ra))
    cl_y = cluster_dist * np.cos(np.radians(cluster_dec)) * np.sin(np.radians(cluster_ra))
    cl_z = cluster_dist * np.sin(np.radians(cluster_dec))
    
    # Build KDTree for clusters
    cluster_coords = np.column_stack([cl_x, cl_y, cl_z])
    tree = cKDTree(cluster_coords)
    
    # Query nearest cluster for each galaxy
    gal_coords = np.column_stack([gal_x, gal_y, gal_z])
    distances, _ = tree.query(gal_coords, k=1)
    
    return distances  # Mpc

def main():
    print("="*70)
    print("CORRECTED ANALYSIS: 3D Distance to Abell Clusters")
    print("="*70)
    print()
    
    # Load ALFALFA - try multiple locations
    print("Loading ALFALFA catalog...")
    alfalfa_paths = [
        DATA_DIR / 'alfalfa' / 'alfalfa_a40.csv',
        DATA_DIR / 'misc' / 'a40.datafile1.csv',
        LOCAL_DATA_DIR / 'alfalfa_a100.csv',
    ]
    
    alfalfa = None
    for path in alfalfa_paths:
        if path.exists():
            alfalfa = pd.read_csv(path)
            print(f"  Loaded from: {path}")
            break
    
    if alfalfa is None:
        print("  ERROR: No ALFALFA data found. Tried:")
        for p in alfalfa_paths:
            print(f"    - {p}")
        return
    
    print(f"  Raw: {len(alfalfa)} sources")
    
    # Parse coordinates
    print("Parsing sexagesimal coordinates...")
    alfalfa = parse_sexagesimal_coords(alfalfa)
    print(f"  Valid coordinates: {len(alfalfa)}")
    
    # Quality cuts - use logMsun (HI mass column name may vary)
    alfalfa = alfalfa[alfalfa['W50'] > 0]
    alfalfa = alfalfa[(alfalfa['W50'] >= 30) & (alfalfa['W50'] <= 400)]
    
    # Handle different column names for HI mass
    mhi_col = 'logMass' if 'logMass' in alfalfa.columns else 'logMsun'
    alfalfa = alfalfa[(alfalfa[mhi_col] >= 7.0) & (alfalfa[mhi_col] <= 9.5)]
    alfalfa = alfalfa[alfalfa['Dist'] > 0]
    print(f"  After quality cuts: {len(alfalfa)} dwarf galaxies")
    print()
    
    # Load Abell clusters
    print("Loading Abell/ACO cluster catalog...")
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    cluster_paths = [
        DATA_DIR / 'alfalfa' / 'sdss_clusters.fits',
        DATA_DIR / 'sdss_voids' / 'sdss_clusters.fits',
        DATA_DIR / 'misc' / 'sdss_clusters.fits',
        LOCAL_DATA_DIR / 'sdss_clusters.fits',
    ]
    
    cluster_data = None
    for path in cluster_paths:
        if path.exists():
            with fits.open(path) as hdul:
                cluster_data = hdul[1].data
            print(f"  Loaded from: {path}")
            break
    
    if cluster_data is None:
        print("  WARNING: No cluster catalog found. Using mock cluster positions.")
        # Create mock cluster data for testing
        cluster_ra = np.random.uniform(100, 260, 50)
        cluster_dec = np.random.uniform(-10, 70, 50)
        cluster_dist = np.random.uniform(50, 200, 50)
    else:
        H0 = 70.0
        c_light = 299792.458
        
        cluster_ra = []
        cluster_dec = []
        cluster_dist = []
        
        for row in cluster_data:
            try:
                z = row['z']
                if z > 0 and z < 0.2:
                    coord = SkyCoord(row['_RA.icrs'], row['_DE.icrs'], unit=(u.hourangle, u.deg))
                    cluster_ra.append(coord.ra.deg)
                    cluster_dec.append(coord.dec.deg)
                    cluster_dist.append(c_light * z / H0)  # Mpc
            except:
                continue
        
        cluster_ra = np.array(cluster_ra)
        cluster_dec = np.array(cluster_dec)
        cluster_dist = np.array(cluster_dist)
    
    print(f"  Loaded {len(cluster_ra)} clusters with z < 0.2")
    print()
    
    # Compute 3D distance to nearest cluster
    print("Computing 3D distance to nearest cluster...")
    dist_to_nearest = compute_3d_distance_to_clusters(
        alfalfa['RA_deg'].values,
        alfalfa['Dec_deg'].values,
        alfalfa['Dist'].values,
        np.array(cluster_ra),
        np.array(cluster_dec),
        np.array(cluster_dist)
    )
    
    alfalfa = alfalfa.copy()
    alfalfa['dist_to_cluster'] = dist_to_nearest
    
    # Classify by 3D distance
    VOID_THRESHOLD = 20.0    # Mpc - far from clusters
    CLUSTER_THRESHOLD = 5.0  # Mpc - close to clusters
    
    void = alfalfa[alfalfa['dist_to_cluster'] > VOID_THRESHOLD].copy()
    cluster = alfalfa[alfalfa['dist_to_cluster'] < CLUSTER_THRESHOLD].copy()
    field = alfalfa[(alfalfa['dist_to_cluster'] >= CLUSTER_THRESHOLD) & 
                    (alfalfa['dist_to_cluster'] <= VOID_THRESHOLD)].copy()
    
    print(f"Environment classification (3D distance):")
    print(f"  Void (>20 Mpc from cluster): {len(void)}")
    print(f"  Cluster (<5 Mpc from cluster): {len(cluster)}")
    print(f"  Field (5-20 Mpc): {len(field)}")
    print()
    
    if len(void) < 50 or len(cluster) < 50:
        print("ERROR: Insufficient sample size!")
        print("This means the Abell clusters dont overlap well with ALFALFA footprint.")
        print()
        print("Trying alternative: use DISTANCE distributions directly")
        print("(Void = distant galaxies, Cluster = nearby galaxies)")
        
        # Alternative: use galaxy distance as proxy
        dist_thresh_low = alfalfa['Dist'].quantile(0.2)
        dist_thresh_high = alfalfa['Dist'].quantile(0.8)
        
        nearby = alfalfa[alfalfa['Dist'] < dist_thresh_low].copy()
        distant = alfalfa[alfalfa['Dist'] > dist_thresh_high].copy()
        
        print(f"\nUsing distance proxy:")
        print(f"  Nearby (<{dist_thresh_low:.0f} Mpc): {len(nearby)}")
        print(f"  Distant (>{dist_thresh_high:.0f} Mpc): {len(distant)}")
        
        void = distant
        cluster = nearby
    
    # Compute rotation velocities
    sin_i_mean = np.pi / 4  # 0.785
    void['v_rot'] = void['W50'] / (2 * sin_i_mean)
    cluster['v_rot'] = cluster['W50'] / (2 * sin_i_mean)
    
    # Check if distance bias is removed
    print()
    print("Distance distributions:")
    print(f"  Void:    <D>={void['Dist'].mean():.1f} Mpc, <logMass>={void['logMsun'].mean():.2f}")
    print(f"  Cluster: <D>={cluster['Dist'].mean():.1f} Mpc, <logMass>={cluster['logMsun'].mean():.2f}")
    print()
    
    # Mass-matched comparison
    print("="*70)
    print("MASS-MATCHED ANALYSIS")
    print("="*70)
    
    mass_min = max(void['logMsun'].quantile(0.1), cluster['logMsun'].quantile(0.1))
    mass_max = min(void['logMsun'].quantile(0.9), cluster['logMsun'].quantile(0.9))
    
    # Check if there's valid mass overlap
    if mass_min >= mass_max:
        print(f"WARNING: No mass overlap between samples!")
        print(f"  Void mass range: {void['logMsun'].quantile(0.1):.2f} - {void['logMsun'].quantile(0.9):.2f}")
        print(f"  Cluster mass range: {cluster['logMsun'].quantile(0.1):.2f} - {cluster['logMsun'].quantile(0.9):.2f}")
        print("  Using full samples without mass-matching.")
        void_m = void
        cluster_m = cluster
        mass_min = min(void['logMsun'].min(), cluster['logMsun'].min())
        mass_max = max(void['logMsun'].max(), cluster['logMsun'].max())
    else:
        void_m = void[(void['logMsun'] >= mass_min) & (void['logMsun'] <= mass_max)]
        cluster_m = cluster[(cluster['logMsun'] >= mass_min) & (cluster['logMsun'] <= mass_max)]
    
    print(f"Mass range: {mass_min:.2f} < logMass < {mass_max:.2f}")
    print(f"  Void:    N={len(void_m)}, <logMass>={void_m['logMsun'].mean():.2f}, <v_rot>={void_m['v_rot'].mean():.1f}")
    print(f"  Cluster: N={len(cluster_m)}, <logMass>={cluster_m['logMsun'].mean():.2f}, <v_rot>={cluster_m['v_rot'].mean():.1f}")
    print()
    
    delta_v = void_m['v_rot'].mean() - cluster_m['v_rot'].mean()
    se = np.sqrt(void_m['v_rot'].var()/len(void_m) + cluster_m['v_rot'].var()/len(cluster_m))
    
    t_stat, p_value = stats.ttest_ind(void_m['v_rot'], cluster_m['v_rot'], equal_var=False)
    
    print("="*70)
    print("RESULT")
    print("="*70)
    print(f"  Delta_v (void - cluster) = {delta_v:+.2f} +/- {se:.2f} km/s")
    print(f"  t-statistic = {t_stat:.3f}")
    print(f"  p-value = {p_value:.4f}")
    print()
    
    # Interpretation
    pred_sdcg = 4.49
    pred_lcdm = 0.0
    
    print("INTERPRETATION:")
    print(f"  SDCG predicts: +{pred_sdcg} km/s (void dwarfs rotate faster)")
    print(f"  LCDM predicts: 0 km/s (no difference)")
    print()
    print(f"  Tension with SDCG: {abs(delta_v - pred_sdcg)/se:.1f} sigma")
    print(f"  Tension with LCDM: {abs(delta_v - pred_lcdm)/se:.1f} sigma")
    print()
    
    if delta_v > 0:
        print("  RESULT: Void dwarfs rotate FASTER (consistent with SDCG direction)")
    else:
        print("  RESULT: Void dwarfs rotate SLOWER (opposite to SDCG prediction)")

if __name__ == "__main__":
    main()
