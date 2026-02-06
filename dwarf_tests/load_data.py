"""
CGC Void vs. Cluster Dwarf Rotation Test - Data Loading Module
===============================================================
Loads and cross-matches:
- ALFALFA HI 21cm survey (rotation velocities from W50 line widths)
- SDSS DR7 Void Catalog (Pan et al. 2012)
- Galaxy stellar masses from ALFALFA-SDSS cross-match

Author: CGC Theory Testing Pipeline
"""

import os
import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, acos

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'misc')


def load_alfalfa_catalog():
    """
    Load ALFALFA α.40 HI source catalog with velocity measurements.
    
    Returns:
        DataFrame with columns: AGCNr, RA, Dec, Vhelio, W50, errW50, Dist, logMHI, etc.
    """
    filepath = os.path.join(DATA_DIR, 'a40.datafile1.csv')
    df = pd.read_csv(filepath)
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Convert W50 to rotation velocity: V_rot ≈ W50 / (2 * sin(i))
    # For statistical samples, average inclination correction is ~1.2
    # V_rot ≈ W50 / 2.4 for typical inclination distribution
    df['V_rot'] = df['W50'] / 2.4
    df['V_rot_err'] = df['errW50'] / 2.4
    
    # Filter for valid extragalactic sources (HIcode == 1 means reliable detection)
    df = df[df['HIcode'] == 1].copy()
    
    # Filter for positive heliocentric velocities (extragalactic)
    df = df[df['Vhelio'] > 500].copy()
    
    # Convert HI mass to stellar mass estimate using M*/MHI scaling
    # For gas-rich dwarfs: M* ≈ 0.5 * MHI (typical for low-mass galaxies)
    df['logMstar_est'] = df['logMsun'] - 0.3  # Conservative estimate
    
    print(f"Loaded {len(df)} ALFALFA galaxies with valid HI detections")
    return df


def load_void_catalog():
    """
    Load void catalog (LocalVoids format).
    
    Returns:
        DataFrame with void centers and radii
    """
    filepath = os.path.join(DATA_DIR, 'voids_catalog.csv')
    df = pd.read_csv(filepath)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'center RA [deg]': 'RA_void',
        'center Dec [deg]': 'Dec_void',
        'center dist [Mpc/h]': 'Dist_void',
        'mean radius (Mpc/h)': 'Radius_void'
    })
    
    print(f"Loaded {len(df)} voids from catalog")
    return df


def load_sdss_crossmatch():
    """
    Load ALFALFA-SDSS DR7 cross-match for photometric properties.
    
    Returns:
        DataFrame with SDSS photometry and stellar mass estimates
    """
    filepath = os.path.join(DATA_DIR, 'a40.datafile3.csv')
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df)} ALFALFA-SDSS cross-matched galaxies")
    return df


def angular_separation(ra1, dec1, ra2, dec2):
    """Calculate angular separation in degrees between two sky positions."""
    ra1, dec1, ra2, dec2 = map(radians, [ra1, dec1, ra2, dec2])
    cos_sep = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1-ra2)
    cos_sep = np.clip(cos_sep, -1, 1)
    return np.degrees(np.arccos(cos_sep))


def cartesian_coords(ra_deg, dec_deg, dist_mpc):
    """Convert RA, Dec, Distance to Cartesian coordinates."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = dist_mpc * np.cos(dec) * np.cos(ra)
    y = dist_mpc * np.cos(dec) * np.sin(ra)
    z = dist_mpc * np.sin(dec)
    return x, y, z


def assign_environment(galaxies_df, voids_df):
    """
    Assign environment (void/field/cluster) to each galaxy based on position.
    
    Uses 3D distance to void centers. A galaxy is in a void if its distance
    to any void center is less than the void radius.
    
    Args:
        galaxies_df: DataFrame with galaxy RA, Dec, Dist
        voids_df: DataFrame with void centers and radii
        
    Returns:
        DataFrame with 'environment' column added
    """
    # Get galaxy positions
    gal_ra = galaxies_df['RAdeg_HI'].values
    gal_dec = galaxies_df['Decdeg_HI'].values
    gal_dist = galaxies_df['Dist'].values
    
    # Initialize environment as 'field'
    environments = np.full(len(galaxies_df), 'field', dtype='<U10')
    min_void_dist_ratio = np.full(len(galaxies_df), np.inf)
    
    # Convert galaxy positions to Cartesian
    gal_x, gal_y, gal_z = cartesian_coords(gal_ra, gal_dec, gal_dist)
    
    # Check each galaxy against each void
    for _, void in voids_df.iterrows():
        void_ra = void['RA_void']
        void_dec = void['Dec_void']
        void_dist = void['Dist_void']
        void_radius = void['Radius_void']
        
        # Convert void center to Cartesian
        void_x, void_y, void_z = cartesian_coords(void_ra, void_dec, void_dist)
        
        # Calculate 3D distance from each galaxy to void center
        dx = gal_x - void_x
        dy = gal_y - void_y
        dz = gal_z - void_z
        dist_to_void = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Normalized distance (ratio to void radius)
        dist_ratio = dist_to_void / void_radius
        
        # Update minimum distance ratio
        min_void_dist_ratio = np.minimum(min_void_dist_ratio, dist_ratio)
    
    # Classify: inside void if dist < 0.8 * radius (conservative)
    environments[min_void_dist_ratio < 0.8] = 'void'
    
    # Classify as cluster for high-density regions (galaxies far from all voids with high velocity dispersion)
    # Use distance from voids as proxy for density
    environments[min_void_dist_ratio > 3.0] = 'cluster'
    
    result_df = galaxies_df.copy()
    result_df['environment'] = environments
    result_df['void_dist_ratio'] = min_void_dist_ratio
    
    # Count environments
    env_counts = result_df['environment'].value_counts()
    print(f"\nEnvironment classification:")
    for env, count in env_counts.items():
        print(f"  {env}: {count} galaxies")
    
    return result_df


def load_and_merge_all():
    """
    Load all datasets and merge into a single DataFrame.
    
    Returns:
        DataFrame with all galaxy properties and environment classification
    """
    print("=" * 60)
    print("CGC Dwarf Galaxy Test - Loading Data")
    print("=" * 60)
    
    # Load catalogs
    alfalfa = load_alfalfa_catalog()
    voids = load_void_catalog()
    sdss = load_sdss_crossmatch()
    
    # Merge ALFALFA with SDSS cross-match
    merged = alfalfa.merge(sdss[['AGCNr', 'rmodelmag', 'uminusr']], 
                           on='AGCNr', how='left')
    
    # Filter for valid distances
    merged = merged[merged['Dist'].notna() & (merged['Dist'] > 0)].copy()
    
    # Assign environments
    merged = assign_environment(merged, voids)
    
    print(f"\nFinal merged dataset: {len(merged)} galaxies")
    
    return merged


if __name__ == "__main__":
    # Test loading
    df = load_and_merge_all()
    print("\nSample of data:")
    print(df[['AGCNr', 'Dist', 'W50', 'V_rot', 'logMsun', 'environment']].head(20))
