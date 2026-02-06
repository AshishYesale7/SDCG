#!/usr/bin/env python3
"""
Mass-Matched Analysis for Void vs Cluster Dwarf Galaxies
=========================================================
Controls for Tully-Fisher bias by matching samples in HI mass.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent
FILTERED_DIR = SCRIPT_DIR / "filtered"

def main():
    # Check if filtered data exists
    void_path = FILTERED_DIR / 'void_dwarfs.csv'
    cluster_path = FILTERED_DIR / 'cluster_dwarfs.csv'
    
    if not void_path.exists() or not cluster_path.exists():
        print("="*70)
        print("MASS-MATCHED ANALYSIS: VOID vs CLUSTER DWARF GALAXIES")
        print("="*70)
        print()
        print("ERROR: Filtered data files not found!")
        print(f"  Expected: {void_path}")
        print(f"  Expected: {cluster_path}")
        print()
        print("Please run dwarf_tests/filter_dwarf_sample.py first to generate the data.")
        return
    
    # Load filtered samples
    void = pd.read_csv(void_path)
    cluster = pd.read_csv(cluster_path)

    print("="*70)
    print("MASS-MATCHED ANALYSIS: VOID vs CLUSTER DWARF GALAXIES")
    print("="*70)
    print()

    print("BEFORE MATCHING:")
    print(f"  Void:    N={len(void):,}, <logMHI>={void['logMHI'].mean():.2f}, <v_rot>={void['v_rot'].mean():.1f} km/s")
    print(f"  Cluster: N={len(cluster):,}, <logMHI>={cluster['logMHI'].mean():.2f}, <v_rot>={cluster['v_rot'].mean():.1f} km/s")
    delta_v_raw = void['v_rot'].mean() - cluster['v_rot'].mean()
    print(f"  Raw Δv = {delta_v_raw:+.2f} km/s (BIASED by mass difference!)")
    print()

    # Mass-matching: restrict to overlapping mass range
    mass_min = max(void['logMHI'].quantile(0.1), cluster['logMHI'].quantile(0.1))
    mass_max = min(void['logMHI'].quantile(0.9), cluster['logMHI'].quantile(0.9))

    print(f"Matching in mass range: {mass_min:.2f} < log(M_HI) < {mass_max:.2f}")
    print()

    void_matched = void[(void['logMHI'] >= mass_min) & (void['logMHI'] <= mass_max)]
    cluster_matched = cluster[(cluster['logMHI'] >= mass_min) & (cluster['logMHI'] <= mass_max)]

    print("AFTER MATCHING:")
    print(f"  Void:    N={len(void_matched):,}, <logMHI>={void_matched['logMHI'].mean():.2f}, <v_rot>={void_matched['v_rot'].mean():.1f} km/s")
    print(f"  Cluster: N={len(cluster_matched):,}, <logMHI>={cluster_matched['logMHI'].mean():.2f}, <v_rot>={cluster_matched['v_rot'].mean():.1f} km/s")
    print()

    # Compute Δv
    delta_v = void_matched['v_rot'].mean() - cluster_matched['v_rot'].mean()
    se = np.sqrt(void_matched['v_rot'].var()/len(void_matched) + cluster_matched['v_rot'].var()/len(cluster_matched))

    # T-test
    t_stat, p_value = stats.ttest_ind(void_matched['v_rot'], cluster_matched['v_rot'], equal_var=False)

    print("="*70)
    print("MASS-MATCHED RESULT")
    print("="*70)
    print(f"  Δv = {delta_v:+.2f} ± {se:.2f} km/s")
    print(f"  t-stat = {t_stat:.3f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("  *** STATISTICALLY SIGNIFICANT (p < 0.05) ***")
    else:
        print("  Not statistically significant")
    print()

    # Compare to predictions
    pred_unconstrained = 4.49  # From screening calculation
    pred_constrained = 0.55
    pred_lcdm = 0.0

    print("INTERPRETATION (after mass-matching):")
    print(f"  vs Unconstrained SDCG (+4.49 km/s): tension = {abs(delta_v - pred_unconstrained)/se:.1f}σ")
    print(f"  vs Lyα-constrained (+0.55 km/s):    tension = {abs(delta_v - pred_constrained)/se:.1f}σ")
    print(f"  vs ΛCDM (0 km/s):                   tension = {abs(delta_v - pred_lcdm)/se:.1f}σ")
    print()

    # Mass-binned analysis
    print("="*70)
    print("MASS-BINNED ANALYSIS (independent of mass-matching)")
    print("="*70)
    print()
    print("Each bin independently controls for mass by comparing at fixed M_HI")
    print()

    bins = [7.0, 8.0, 8.5, 9.0, 9.5]
    print(f"{'Mass Bin':<15} {'N_void':<10} {'N_cluster':<12} {'Δv (km/s)':<18} {'Signif.':<8}")
    print("-"*65)

    binned_results = []
    for i in range(len(bins)-1):
        m_lo, m_hi = bins[i], bins[i+1]
        v_bin = void[(void['logMHI'] >= m_lo) & (void['logMHI'] < m_hi)]
        c_bin = cluster[(cluster['logMHI'] >= m_lo) & (cluster['logMHI'] < m_hi)]
        
        if len(v_bin) > 10 and len(c_bin) > 10:
            dv = v_bin['v_rot'].mean() - c_bin['v_rot'].mean()
            se_bin = np.sqrt(v_bin['v_rot'].var()/len(v_bin) + c_bin['v_rot'].var()/len(c_bin))
            sig = dv / se_bin
            print(f"{m_lo:.1f}-{m_hi:.1f}         {len(v_bin):<10} {len(c_bin):<12} {dv:+.2f} ± {se_bin:.2f}     {sig:.1f}σ")
            binned_results.append((m_lo, m_hi, dv, se_bin, len(v_bin), len(c_bin)))
        else:
            print(f"{m_lo:.1f}-{m_hi:.1f}         {len(v_bin):<10} {len(c_bin):<12} (insufficient)")

    print()
    print("="*70)
    print("COMBINED BIN ANALYSIS (inverse-variance weighted)")
    print("="*70)

    if binned_results:
        weights = [1/se**2 for (_, _, dv, se, _, _) in binned_results]
        weighted_mean = sum(w * dv for (_, _, dv, se, _, _), w in zip(binned_results, weights)) / sum(weights)
        combined_se = 1/np.sqrt(sum(weights))
        
        print(f"  Weighted-mean Δv = {weighted_mean:+.2f} ± {combined_se:.2f} km/s")
        print()
        print("  Interpretation of weighted-mean:")
        print(f"    vs Unconstrained SDCG (+4.49 km/s): tension = {abs(weighted_mean - 4.49)/combined_se:.1f}σ")
        print(f"    vs Lyα-constrained (+0.55 km/s):    tension = {abs(weighted_mean - 0.55)/combined_se:.1f}σ")
        print(f"    vs ΛCDM (0 km/s):                   tension = {abs(weighted_mean - 0)/combined_se:.1f}σ")

    print()
    print("="*70)
    print("SUMMARY FOR THESIS")
    print("="*70)
    print()
    print("After controlling for the Tully-Fisher relation via mass-matching:")
    print(f"  • Mass-matched Δv = {delta_v:+.2f} ± {se:.2f} km/s")
    if len(binned_results) > 0:
        print(f"  • Bin-weighted Δv = {weighted_mean:+.2f} ± {combined_se:.2f} km/s")
    print()
    
    if delta_v > 0:
        print("  RESULT: Void dwarfs rotate FASTER than cluster dwarfs")
        print("  This is CONSISTENT with SDCG screening prediction")
    else:
        print("  RESULT: Void dwarfs rotate SLOWER than cluster dwarfs")
        print("  This would CONTRADICT SDCG screening prediction")

if __name__ == "__main__":
    main()
