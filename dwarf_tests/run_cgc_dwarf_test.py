#!/usr/bin/env python3
"""
run_cgc_dwarf_test.py - Complete Pipeline for Void vs Cluster Dwarf Test

This script executes the full CGC validation pipeline:
1. Load ALFALFA HI + void catalog data
2. Filter for dwarf galaxies and classify by environment
3. Calculate velocity differences with statistical analysis
4. Generate publication-quality plots

CGC Prediction: Δv = +12 ± 3 km/s (void dwarfs rotate faster)
ΛCDM Prediction: Δv = 0 km/s (no systematic difference)
"""

import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_samples import prepare_samples
from analyze_velocity import analyze_velocity_difference, run_null_test


def run_full_analysis():
    """
    Execute the complete CGC Dwarf Rotation Test pipeline.
    """
    print("=" * 70)
    print("     VOID vs CLUSTER DWARF ROTATION TEST - CGC VALIDATION")
    print("=" * 70)
    print()
    print("CGC Prediction: Δv = +12 ± 3 km/s (void dwarfs rotate faster)")
    print("ΛCDM Prediction: Δv = 0 km/s (no environment dependence)")
    print()
    print("-" * 70)
    
    # ===== PHASE 1 & 2: Data Loading + Filtering =====
    print("\n[PHASE 1 & 2] Loading Data and Applying Filters...")
    print("-" * 40)
    
    try:
        # prepare_samples() handles both loading and filtering internally
        samples = prepare_samples()
        df_void = samples['void']
        df_field = samples['field']
        df_cluster = samples['cluster']
        print(f"✓ Void dwarfs:    N = {len(df_void)}")
        print(f"✓ Field dwarfs:   N = {len(df_field)}")
        print(f"✓ Cluster dwarfs: N = {len(df_cluster)}")
    except Exception as e:
        print(f"✗ Filtering failed: {e}")
        return None
    
    # Check minimum sample sizes
    if len(df_void) < 10 or len(df_cluster) < 10:
        print("\n⚠ WARNING: Sample sizes too small for reliable statistics!")
        print("  Consider relaxing filter criteria.")
    
    # ===== PHASE 3: Statistical Analysis =====
    print("\n[PHASE 3] Statistical Analysis...")
    print("-" * 40)
    
    try:
        # Pass the samples dict to the analysis function
        results = analyze_velocity_difference(samples)
        
        # Extract velocities for later use
        v_void = df_void['V_rot'].values
        v_cluster = df_cluster['V_rot'].values
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ===== PHASE 4: Null Test =====
    print("\n[PHASE 4] Running Null Test (Bias Detection)...")
    print("-" * 40)
    
    try:
        null_results = run_null_test(samples)
        if null_results:
            results['null_test'] = null_results
    except Exception as e:
        print(f"⚠ Null test failed: {e}")
    
    # ===== SAVE RESULTS =====
    print("\n[SAVING] Storing Results...")
    print("-" * 40)
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract key values from nested structure
    delta_v_data = results.get('delta_v', {})
    if isinstance(delta_v_data, dict):
        delta_v = delta_v_data.get('value', np.nan)
        delta_v_err = delta_v_data.get('error', np.nan)
        t_stat = delta_v_data.get('t_stat', np.nan)
        p_value = delta_v_data.get('p_value', np.nan)
    else:
        delta_v = delta_v_err = t_stat = p_value = np.nan
    
    void_data = results.get('void', {})
    cluster_data = results.get('cluster', {})
    mean_void = void_data.get('mean', np.nan)
    err_void = void_data.get('sem', np.nan)
    mean_cluster = cluster_data.get('mean', np.nan)
    err_cluster = cluster_data.get('sem', np.nan)
    
    # Prepare flat results for saving and plotting
    results_flat = {
        'delta_v': delta_v,
        'delta_v_err': delta_v_err,
        'mean_void': mean_void,
        'err_void': err_void,
        'mean_cluster': mean_cluster,
        'err_cluster': err_cluster,
        't_statistic': t_stat,
        'p_value': p_value
    }
    
    results_file = os.path.join(results_dir, 'cgc_dwarf_analysis.npz')
    np.savez(results_file,
             v_void=v_void,
             v_cluster=v_cluster,
             results=results_flat)
    print(f"✓ Results saved to: {results_file}")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("                       FINAL RESULTS")
    print("=" * 70)
    
    # Use the already-extracted variables (no need to re-extract from results)
    print(f"""
    Sample Statistics:
    ──────────────────────────────────────────────────────
      Void Dwarfs:     N = {len(v_void):>5}    <V_rot> = {mean_void:.2f} ± {err_void:.2f} km/s
      Cluster Dwarfs:  N = {len(v_cluster):>5}    <V_rot> = {mean_cluster:.2f} ± {err_cluster:.2f} km/s
    
    Key Result:
    ──────────────────────────────────────────────────────
      Δv (observed)  = {delta_v:+.2f} ± {delta_v_err:.2f} km/s
      
      CGC prediction:  +12.00 ± 3.00 km/s
      ΛCDM prediction:   0.00 ± 0.50 km/s
    
    Statistical Significance:
    ──────────────────────────────────────────────────────
      Welch's t-statistic: {t_stat:.3f}
      p-value:             {p_value:.6f}
      
      Significance:  {'p < 0.001 (HIGHLY SIGNIFICANT)' if p_value < 0.001 else 
                      'p < 0.01 (VERY SIGNIFICANT)' if p_value < 0.01 else
                      'p < 0.05 (SIGNIFICANT)' if p_value < 0.05 else
                      'NOT SIGNIFICANT (p ≥ 0.05)'}
    """)
    
    # Determine which prediction is supported
    cgc_sigma = abs(delta_v - 12) / np.sqrt(delta_v_err**2 + 3**2)
    lcdm_sigma = abs(delta_v - 0) / delta_v_err if delta_v_err > 0 else np.inf
    
    print("    Theory Comparison:")
    print("    ──────────────────────────────────────────────────────")
    print(f"      Distance from CGC (+12):  {cgc_sigma:.2f}σ")
    print(f"      Distance from ΛCDM (0):   {lcdm_sigma:.2f}σ")
    
    if cgc_sigma < 2 and lcdm_sigma > 2:
        verdict = "CGC PREDICTION SUPPORTED"
        verdict_symbol = "✓✓✓"
    elif lcdm_sigma < 2 and cgc_sigma > 2:
        verdict = "ΛCDM PREDICTION SUPPORTED"
        verdict_symbol = "───"
    elif cgc_sigma < 2 and lcdm_sigma < 2:
        verdict = "INCONCLUSIVE (both within 2σ)"
        verdict_symbol = "???"
    else:
        verdict = "NEITHER MODEL FAVORED"
        verdict_symbol = "✗✗✗"
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   VERDICT:  {verdict_symbol}  {verdict:<42} ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ===== Generate Plots =====
    print("[PLOTTING] Generating Visualizations...")
    print("-" * 40)
    
    try:
        from plot_results import plot_velocity_histograms
        plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
        plot_velocity_histograms(v_void, v_cluster, results_flat, plots_dir)
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
        print("  Run plot_results.py separately after fixing matplotlib.")
    
    print("\n" + "=" * 70)
    print("                    ANALYSIS COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = run_full_analysis()
