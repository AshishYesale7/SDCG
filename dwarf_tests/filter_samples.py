"""
CGC Void vs. Cluster Dwarf Rotation Test - Sample Filtering Module
===================================================================
Applies the physics-based filters specified in the CGC thesis:
- Stellar Mass: 10^7 to 10^9 M_sun (dwarf galaxies)
- Environment: Void (δ < -0.5) vs Cluster (δ > 100)
- Quality cuts for reliable rotation measurements

Author: CGC Theory Testing Pipeline
"""

import pandas as pd
import numpy as np
from load_data import load_and_merge_all


# CGC Theory Parameters
MASS_MIN_LOG = 7.0    # 10^7 M_sun - minimum stellar mass
MASS_MAX_LOG = 9.0    # 10^9 M_sun - maximum stellar mass (above this, self-screening)
MIN_VROT = 15.0       # km/s - minimum reliable rotation velocity
MAX_VROT = 150.0      # km/s - maximum for dwarfs
MIN_DISTANCE = 10.0   # Mpc - minimum distance to avoid Local Group peculiar velocities
MAX_DISTANCE = 200.0  # Mpc - maximum distance for reliable measurements
MIN_SNR = 6.5         # Minimum signal-to-noise for HI detection


def filter_dwarf_galaxies(df):
    """
    Apply CGC dwarf galaxy selection criteria.
    
    Args:
        df: DataFrame with galaxy properties
        
    Returns:
        Filtered DataFrame containing only dwarf galaxies
    """
    print("\n" + "=" * 60)
    print("Applying Dwarf Galaxy Selection Criteria")
    print("=" * 60)
    
    n_initial = len(df)
    print(f"Initial sample: {n_initial} galaxies")
    
    # Filter 1: Stellar Mass (using HI mass as proxy for gas-rich dwarfs)
    # logMHI in range [7, 10] typically corresponds to M* in [10^7, 10^9]
    mask_mass = (df['logMsun'] >= MASS_MIN_LOG) & (df['logMsun'] <= MASS_MAX_LOG + 1.0)
    df_filtered = df[mask_mass].copy()
    print(f"After mass cut ({MASS_MIN_LOG} < log(MHI/M_sun) < {MASS_MAX_LOG+1}): {len(df_filtered)}")
    
    # Filter 2: Rotation velocity range (plausible dwarf range)
    mask_vrot = (df_filtered['V_rot'] >= MIN_VROT) & (df_filtered['V_rot'] <= MAX_VROT)
    df_filtered = df_filtered[mask_vrot].copy()
    print(f"After V_rot cut ({MIN_VROT} < V_rot < {MAX_VROT} km/s): {len(df_filtered)}")
    
    # Filter 3: Distance range
    mask_dist = (df_filtered['Dist'] >= MIN_DISTANCE) & (df_filtered['Dist'] <= MAX_DISTANCE)
    df_filtered = df_filtered[mask_dist].copy()
    print(f"After distance cut ({MIN_DISTANCE} < D < {MAX_DISTANCE} Mpc): {len(df_filtered)}")
    
    # Filter 4: Signal-to-noise
    mask_snr = df_filtered['SNR'] >= MIN_SNR
    df_filtered = df_filtered[mask_snr].copy()
    print(f"After SNR cut (SNR >= {MIN_SNR}): {len(df_filtered)}")
    
    # Filter 5: Valid velocity errors
    mask_err = (df_filtered['V_rot_err'] > 0) & (df_filtered['V_rot_err'] < 50)
    df_filtered = df_filtered[mask_err].copy()
    print(f"After error cut (0 < err < 50 km/s): {len(df_filtered)}")
    
    print(f"\nFinal dwarf sample: {len(df_filtered)} galaxies ({100*len(df_filtered)/n_initial:.1f}% of original)")
    
    return df_filtered


def split_by_environment(df):
    """
    Split sample into void and cluster subsamples.
    
    Returns:
        tuple: (df_void, df_field, df_cluster)
    """
    df_void = df[df['environment'] == 'void'].copy()
    df_field = df[df['environment'] == 'field'].copy()
    df_cluster = df[df['environment'] == 'cluster'].copy()
    
    print("\n" + "=" * 60)
    print("Environment Split")
    print("=" * 60)
    print(f"Void galaxies: {len(df_void)}")
    print(f"Field galaxies: {len(df_field)}")
    print(f"Cluster/High-density galaxies: {len(df_cluster)}")
    
    return df_void, df_field, df_cluster


def match_morphology(df_void, df_cluster, df_field):
    """
    Attempt to match samples by observational properties.
    
    Since we don't have direct morphology, we match by:
    - HI mass distribution
    - Distance distribution
    - Velocity error distribution
    
    Returns:
        tuple: matched (df_void, df_cluster)
    """
    print("\n" + "=" * 60)
    print("Sample Matching")
    print("=" * 60)
    
    if len(df_void) == 0 or len(df_cluster) == 0:
        print("Warning: One or more samples is empty. Using field as comparison.")
        if len(df_void) == 0:
            # Use field galaxies split by void distance as proxy
            median_ratio = df_field['void_dist_ratio'].median()
            df_void = df_field[df_field['void_dist_ratio'] < median_ratio].copy()
            df_cluster = df_field[df_field['void_dist_ratio'] >= median_ratio].copy()
            print(f"Using void_dist_ratio split: void-like {len(df_void)}, cluster-like {len(df_cluster)}")
    
    # Report statistics before matching
    print("\nBefore matching:")
    for name, sample in [("Void", df_void), ("Cluster", df_cluster)]:
        if len(sample) > 0:
            print(f"  {name}: N={len(sample)}, <logMHI>={sample['logMsun'].mean():.2f}, "
                  f"<D>={sample['Dist'].mean():.1f} Mpc, <V_rot>={sample['V_rot'].mean():.1f} km/s")
    
    # For a more rigorous analysis, we could implement propensity score matching
    # For now, we report the raw samples with appropriate error analysis
    
    return df_void, df_cluster


def prepare_samples():
    """
    Main function to prepare void and cluster dwarf samples.
    
    Returns:
        dict with 'void', 'field', 'cluster' DataFrames
    """
    # Load data
    df = load_and_merge_all()
    
    # Apply dwarf selection
    df_dwarfs = filter_dwarf_galaxies(df)
    
    # Split by environment
    df_void, df_field, df_cluster = split_by_environment(df_dwarfs)
    
    # Match samples
    df_void_matched, df_cluster_matched = match_morphology(df_void, df_cluster, df_field)
    
    return {
        'void': df_void_matched,
        'field': df_field,
        'cluster': df_cluster_matched,
        'all_dwarfs': df_dwarfs
    }


if __name__ == "__main__":
    samples = prepare_samples()
    
    print("\n" + "=" * 60)
    print("SAMPLE SUMMARY")
    print("=" * 60)
    
    for name, df in samples.items():
        if len(df) > 0:
            print(f"\n{name.upper()} ({len(df)} galaxies):")
            print(df[['AGCNr', 'Dist', 'V_rot', 'V_rot_err', 'logMsun', 'environment']].describe())
