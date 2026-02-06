"""
CGC Void vs. Cluster Dwarf Rotation Test - Velocity Analysis Module
====================================================================
Performs the core statistical analysis:
- Calculate mean rotation velocities for void vs cluster dwarfs
- Error propagation using standard error of the mean
- Statistical significance testing
- CGC prediction: Δv = +12 ± 3 km/s excess in voids

Author: CGC Theory Testing Pipeline
"""

import numpy as np
import pandas as pd
from scipy import stats
from filter_samples import prepare_samples


# CGC Theory Predictions
CGC_PREDICTED_DELTA_V = 12.0  # km/s - predicted void excess
CGC_PREDICTED_ERROR = 3.0    # km/s - theoretical uncertainty
LCDM_PREDICTED_DELTA_V = 0.0  # Standard physics: no difference


def weighted_mean(values, errors):
    """
    Calculate inverse-variance weighted mean.
    
    Args:
        values: array of measurements
        errors: array of measurement uncertainties
        
    Returns:
        tuple: (weighted_mean, error_on_mean)
    """
    if len(values) == 0:
        return np.nan, np.nan
    
    # Handle zero or very small errors
    errors = np.maximum(errors, 1e-6)
    
    weights = 1.0 / errors**2
    weighted_mean = np.sum(weights * values) / np.sum(weights)
    error_on_mean = 1.0 / np.sqrt(np.sum(weights))
    
    return weighted_mean, error_on_mean


def bootstrap_mean_diff(v1, v2, n_bootstrap=10000):
    """
    Bootstrap the difference in means to get robust error estimate.
    
    Args:
        v1, v2: arrays of velocities for two samples
        n_bootstrap: number of bootstrap iterations
        
    Returns:
        tuple: (mean_diff, std_diff, p_value)
    """
    n1, n2 = len(v1), len(v2)
    
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    
    diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(v1, size=n1, replace=True)
        sample2 = np.random.choice(v2, size=n2, replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))
    
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # P-value for null hypothesis (difference = 0)
    p_value = np.mean(np.abs(diffs) >= np.abs(mean_diff))
    
    return mean_diff, std_diff, p_value


def analyze_velocity_difference(samples):
    """
    Main analysis: compare rotation velocities between void and cluster dwarfs.
    
    Args:
        samples: dict with 'void' and 'cluster' DataFrames
        
    Returns:
        dict with analysis results
    """
    print("\n" + "=" * 70)
    print("CGC VOID vs. CLUSTER DWARF ROTATION VELOCITY ANALYSIS")
    print("=" * 70)
    
    df_void = samples['void']
    df_cluster = samples['cluster']
    df_field = samples['field']
    
    results = {}
    
    # --- Void Sample Statistics ---
    print("\n### VOID DWARF SAMPLE ###")
    if len(df_void) > 0:
        v_void = df_void['V_rot'].values
        err_void = df_void['V_rot_err'].values
        
        mean_void, sem_void = weighted_mean(v_void, err_void)
        median_void = np.median(v_void)
        std_void = np.std(v_void)
        n_void = len(v_void)
        
        print(f"N = {n_void}")
        print(f"Mean V_rot = {mean_void:.2f} ± {sem_void:.2f} km/s (weighted)")
        print(f"Median V_rot = {median_void:.2f} km/s")
        print(f"Std Dev = {std_void:.2f} km/s")
        
        results['void'] = {
            'n': n_void, 'mean': mean_void, 'sem': sem_void,
            'median': median_void, 'std': std_void
        }
    else:
        print("WARNING: No void galaxies in sample!")
        results['void'] = None
    
    # --- Cluster Sample Statistics ---
    print("\n### CLUSTER/HIGH-DENSITY DWARF SAMPLE ###")
    if len(df_cluster) > 0:
        v_cluster = df_cluster['V_rot'].values
        err_cluster = df_cluster['V_rot_err'].values
        
        mean_cluster, sem_cluster = weighted_mean(v_cluster, err_cluster)
        median_cluster = np.median(v_cluster)
        std_cluster = np.std(v_cluster)
        n_cluster = len(v_cluster)
        
        print(f"N = {n_cluster}")
        print(f"Mean V_rot = {mean_cluster:.2f} ± {sem_cluster:.2f} km/s (weighted)")
        print(f"Median V_rot = {median_cluster:.2f} km/s")
        print(f"Std Dev = {std_cluster:.2f} km/s")
        
        results['cluster'] = {
            'n': n_cluster, 'mean': mean_cluster, 'sem': sem_cluster,
            'median': median_cluster, 'std': std_cluster
        }
    else:
        print("WARNING: No cluster galaxies in sample!")
        results['cluster'] = None
    
    # --- Field Sample Statistics (Control) ---
    print("\n### FIELD DWARF SAMPLE (CONTROL) ###")
    if len(df_field) > 0:
        v_field = df_field['V_rot'].values
        err_field = df_field['V_rot_err'].values
        
        mean_field, sem_field = weighted_mean(v_field, err_field)
        n_field = len(v_field)
        
        print(f"N = {n_field}")
        print(f"Mean V_rot = {mean_field:.2f} ± {sem_field:.2f} km/s (weighted)")
        
        results['field'] = {
            'n': n_field, 'mean': mean_field, 'sem': sem_field
        }
    
    # --- Main Result: Velocity Difference ---
    print("\n" + "=" * 70)
    print("MAIN RESULT: VOID - CLUSTER VELOCITY DIFFERENCE")
    print("=" * 70)
    
    if results['void'] is not None and results['cluster'] is not None:
        # Observed difference
        delta_v = results['void']['mean'] - results['cluster']['mean']
        delta_v_err = np.sqrt(results['void']['sem']**2 + results['cluster']['sem']**2)
        
        # Bootstrap for robust error
        if len(df_void) >= 5 and len(df_cluster) >= 5:
            delta_boot, delta_boot_err, p_null = bootstrap_mean_diff(
                df_void['V_rot'].values, df_cluster['V_rot'].values
            )
            print(f"\nBootstrap estimate: Δv = {delta_boot:.2f} ± {delta_boot_err:.2f} km/s")
        else:
            delta_boot = delta_v
            delta_boot_err = delta_v_err
            p_null = np.nan
        
        # Welch's t-test for unequal variances
        t_stat, p_value = stats.ttest_ind(
            df_void['V_rot'].values, 
            df_cluster['V_rot'].values,
            equal_var=False
        )
        
        print(f"\n>>> OBSERVED: Δv (void - cluster) = {delta_v:.2f} ± {delta_v_err:.2f} km/s <<<")
        print(f"    (Welch's t-test: t = {t_stat:.2f}, p = {p_value:.4f})")
        
        # Compare with predictions
        print("\n" + "-" * 50)
        print("COMPARISON WITH THEORETICAL PREDICTIONS:")
        print("-" * 50)
        
        # CGC prediction
        cgc_diff = delta_v - CGC_PREDICTED_DELTA_V
        cgc_sigma = abs(cgc_diff) / delta_v_err if delta_v_err > 0 else np.inf
        
        print(f"\n  CGC Prediction: Δv = +{CGC_PREDICTED_DELTA_V} ± {CGC_PREDICTED_ERROR} km/s")
        print(f"  Observed - CGC: {cgc_diff:+.2f} km/s ({cgc_sigma:.1f}σ from CGC)")
        
        # ΛCDM prediction (null)
        lcdm_diff = delta_v - LCDM_PREDICTED_DELTA_V
        lcdm_sigma = abs(lcdm_diff) / delta_v_err if delta_v_err > 0 else np.inf
        
        print(f"\n  ΛCDM Prediction: Δv = 0 km/s (no environment dependence)")
        print(f"  Observed - ΛCDM: {lcdm_diff:+.2f} km/s ({lcdm_sigma:.1f}σ from ΛCDM)")
        
        # Verdict
        print("\n" + "=" * 70)
        print("STATISTICAL VERDICT")
        print("=" * 70)
        
        if lcdm_sigma >= 3.0:
            print(f">>> SIGNIFICANT DEVIATION FROM ΛCDM at {lcdm_sigma:.1f}σ <<<")
            if cgc_sigma < 2.0:
                print(f">>> RESULT CONSISTENT WITH CGC PREDICTION <<<")
            else:
                print(f">>> But also deviates from CGC by {cgc_sigma:.1f}σ <<<")
        elif lcdm_sigma >= 2.0:
            print(f">>> MARGINAL DEVIATION FROM ΛCDM at {lcdm_sigma:.1f}σ (needs more data) <<<")
        else:
            print(f">>> NO SIGNIFICANT DEVIATION from either prediction ({lcdm_sigma:.1f}σ) <<<")
            print(f"    Current data cannot distinguish CGC from ΛCDM")
        
        results['delta_v'] = {
            'value': delta_v,
            'error': delta_v_err,
            'cgc_sigma': cgc_sigma,
            'lcdm_sigma': lcdm_sigma,
            't_stat': t_stat,
            'p_value': p_value
        }
    else:
        print("\nCANNOT COMPUTE: Insufficient galaxies in one or both samples")
        results['delta_v'] = None
    
    return results


def run_null_test(samples):
    """
    Run null test: compare field galaxies split by some property.
    This should show Δv ≈ 0 if the method is unbiased.
    """
    print("\n" + "=" * 70)
    print("NULL TEST: Random split of field sample")
    print("=" * 70)
    
    df_field = samples['field']
    
    if len(df_field) < 20:
        print("Insufficient field galaxies for null test")
        return
    
    # Random 50-50 split
    np.random.seed(42)
    mask = np.random.choice([True, False], size=len(df_field))
    
    v1 = df_field.loc[mask, 'V_rot'].values
    v2 = df_field.loc[~mask, 'V_rot'].values
    
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    sem1 = np.std(v1) / np.sqrt(len(v1))
    sem2 = np.std(v2) / np.sqrt(len(v2))
    
    delta_null = mean1 - mean2
    delta_null_err = np.sqrt(sem1**2 + sem2**2)
    
    print(f"Random split A: N={len(v1)}, <V>={mean1:.2f} ± {sem1:.2f} km/s")
    print(f"Random split B: N={len(v2)}, <V>={mean2:.2f} ± {sem2:.2f} km/s")
    print(f"Δv (A-B) = {delta_null:.2f} ± {delta_null_err:.2f} km/s")
    print(f"Expected: Δv ≈ 0 km/s")
    
    if abs(delta_null) < 2 * delta_null_err:
        print(">>> NULL TEST PASSED: No systematic bias detected <<<")
    else:
        print(">>> WARNING: Possible systematic bias in methodology <<<")


if __name__ == "__main__":
    # Prepare samples
    samples = prepare_samples()
    
    # Run main analysis
    results = analyze_velocity_difference(samples)
    
    # Run null test
    run_null_test(samples)
    
    # Save results
    print("\n" + "=" * 70)
    print("Analysis complete. Results ready for plotting.")
    print("=" * 70)
