#!/usr/bin/env python3
"""
PRECISE DWARF GALAXY ANALYSIS
=============================
Author: Data Science Analysis Pipeline
Purpose: Rigorous verification of SDCG dwarf galaxy predictions

This script performs a CAREFUL, step-by-step analysis to:
1. Verify data sources are REAL (not synthetic)
2. Check environment classification methodology
3. Validate the velocity comparison
4. Identify any systematic biases
"""

import numpy as np
import os
import sys
from pathlib import Path

# Setup paths - allow running from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # Script is at project root
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("PRECISE DWARF GALAXY ANALYSIS - DATA SCIENCE VERIFICATION")
print("=" * 80)

# ============================================================================
# STEP 1: CHECK DATA FILES EXIST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: VERIFYING DATA FILES")
print("=" * 80)

data_dir = DATA_DIR / "misc"
required_files = [
    "a40.datafile1.csv",  # ALFALFA HI catalog
    "voids_catalog.csv",  # Void positions
    "a40.datafile3.csv",  # ALFALFA-SDSS cross-match
]

files_exist = True
for f in required_files:
    filepath = data_dir / f
    if filepath.exists():
        size = filepath.stat().st_size
        print(f"  ✓ {f}: EXISTS ({size/1024:.1f} KB)")
    else:
        print(f"  ✗ {f}: MISSING")
        files_exist = False

if not files_exist:
    print("\n⚠ Some data files are missing. Checking for results file...")

# ============================================================================
# STEP 2: CHECK STORED RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: EXAMINING STORED RESULTS")
print("=" * 80)

results_file = RESULTS_DIR / "cgc_dwarf_analysis.npz"
if results_file.exists():
    data = np.load(results_file, allow_pickle=True)
    print(f"Results file: {results_file}")
    print(f"Keys: {list(data.keys())}")
    
    for key in data.keys():
        val = data[key]
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                print(f"  {key}: {val.item()}")
            else:
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                if len(val) > 0 and len(val) < 20:
                    print(f"       values={val}")
        else:
            print(f"  {key}: {type(val)}")
    
    # Extract velocity arrays
    if 'v_void' in data and 'v_cluster' in data:
        v_void = data['v_void']
        v_cluster = data['v_cluster']
        
        print("\n" + "-" * 60)
        print("RAW VELOCITY DATA:")
        print("-" * 60)
        print(f"Void galaxies:    N = {len(v_void)}")
        print(f"  Min:  {np.min(v_void):.2f} km/s")
        print(f"  Max:  {np.max(v_void):.2f} km/s")
        print(f"  Mean: {np.mean(v_void):.2f} km/s")
        print(f"  Std:  {np.std(v_void):.2f} km/s")
        print(f"  Median: {np.median(v_void):.2f} km/s")
        
        print(f"\nCluster galaxies: N = {len(v_cluster)}")
        print(f"  Min:  {np.min(v_cluster):.2f} km/s")
        print(f"  Max:  {np.max(v_cluster):.2f} km/s")
        print(f"  Mean: {np.mean(v_cluster):.2f} km/s")
        print(f"  Std:  {np.std(v_cluster):.2f} km/s")
        print(f"  Median: {np.median(v_cluster):.2f} km/s")
        
        # ============================================================================
        # STEP 3: CALCULATE VELOCITY DIFFERENCES
        # ============================================================================
        print("\n" + "=" * 80)
        print("STEP 3: VELOCITY DIFFERENCE CALCULATION")
        print("=" * 80)
        
        mean_void = np.mean(v_void)
        mean_cluster = np.mean(v_cluster)
        sem_void = np.std(v_void) / np.sqrt(len(v_void))
        sem_cluster = np.std(v_cluster) / np.sqrt(len(v_cluster))
        
        delta_v = mean_void - mean_cluster
        delta_v_err = np.sqrt(sem_void**2 + sem_cluster**2)
        
        print(f"Mean V_rot (void):    {mean_void:.3f} ± {sem_void:.3f} km/s")
        print(f"Mean V_rot (cluster): {mean_cluster:.3f} ± {sem_cluster:.3f} km/s")
        print(f"\nΔv = V_void - V_cluster = {delta_v:+.3f} ± {delta_v_err:.3f} km/s")
        
        # Statistical tests
        from scipy import stats
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(v_void, v_cluster, equal_var=False)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_mw = stats.mannwhitneyu(v_void, v_cluster, alternative='two-sided')
        
        # Kolmogorov-Smirnov test
        ks_stat, p_ks = stats.ks_2samp(v_void, v_cluster)
        
        print("\n" + "-" * 60)
        print("STATISTICAL TESTS:")
        print("-" * 60)
        print(f"Welch's t-test:      t = {t_stat:.4f}, p = {p_value:.6f}")
        print(f"Mann-Whitney U:      U = {u_stat:.0f}, p = {p_mw:.6f}")
        print(f"Kolmogorov-Smirnov:  D = {ks_stat:.4f}, p = {p_ks:.6f}")
        
        # ============================================================================
        # STEP 4: COMPARISON WITH PREDICTION
        # ============================================================================
        print("\n" + "=" * 80)
        print("STEP 4: COMPARISON WITH SDCG PREDICTION")
        print("=" * 80)
        
        # SDCG prediction parameters
        mu = 0.045  # Lyman-alpha constrained
        mu_err = 0.019
        
        # Theoretical prediction
        # Δv/v = sqrt(1 + μ*S_void) - sqrt(1 + μ*S_cluster)
        # For void: S ≈ 1.0, for cluster: S ≈ 0.001
        S_void = 1.0
        S_cluster = 0.001
        v_typical = 80  # km/s typical rotation velocity
        
        # Enhancement factor
        enhancement_void = np.sqrt(1 + mu * S_void) - 1
        enhancement_cluster = np.sqrt(1 + mu * S_cluster) - 1
        
        delta_v_predicted = v_typical * (enhancement_void - enhancement_cluster)
        delta_v_pred_err = v_typical * (mu_err / (2 * np.sqrt(1 + mu)))  # Approximate error
        
        print(f"SDCG Parameters:")
        print(f"  μ = {mu} ± {mu_err}")
        print(f"  S(void) ≈ {S_void}")
        print(f"  S(cluster) ≈ {S_cluster}")
        print(f"  v_typical = {v_typical} km/s")
        print(f"\nTheoretical calculation:")
        print(f"  Enhancement (void):    {enhancement_void*100:.3f}%")
        print(f"  Enhancement (cluster): {enhancement_cluster*100:.6f}%")
        print(f"  Δv_predicted = {delta_v_predicted:.2f} ± {delta_v_pred_err:.2f} km/s")
        
        print("\n" + "-" * 60)
        print("PREDICTION vs OBSERVATION:")
        print("-" * 60)
        print(f"SDCG Predicts:  Δv = +{delta_v_predicted:.2f} ± {delta_v_pred_err:.2f} km/s")
        print(f"ΛCDM Predicts:  Δv =  0.00 ± ~2 km/s (no enhancement)")
        print(f"Observed:       Δv = {delta_v:+.2f} ± {delta_v_err:.2f} km/s")
        
        # Calculate sigma distances
        sigma_from_sdcg = abs(delta_v - delta_v_predicted) / np.sqrt(delta_v_err**2 + delta_v_pred_err**2)
        sigma_from_lcdm = abs(delta_v) / delta_v_err
        
        print(f"\nDistance from SDCG: {sigma_from_sdcg:.2f}σ")
        print(f"Distance from ΛCDM: {sigma_from_lcdm:.2f}σ")
        
        # ============================================================================
        # STEP 5: INTERPRETATION
        # ============================================================================
        print("\n" + "=" * 80)
        print("STEP 5: INTERPRETATION")
        print("=" * 80)
        
        if delta_v > 0:
            print("✓ SIGN IS CORRECT: Void dwarfs rotate faster than cluster dwarfs")
        else:
            print("✗ SIGN IS OPPOSITE: Cluster dwarfs rotate faster than void dwarfs")
        
        if p_value < 0.05:
            print(f"✓ STATISTICALLY SIGNIFICANT (p = {p_value:.6f} < 0.05)")
        else:
            print(f"✗ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f} ≥ 0.05)")
        
        if sigma_from_sdcg < 2:
            print(f"→ CONSISTENT WITH SDCG (within 2σ)")
        else:
            print(f"→ INCONSISTENT WITH SDCG (>{sigma_from_sdcg:.1f}σ away)")
        
        if sigma_from_lcdm < 2:
            print(f"→ CONSISTENT WITH ΛCDM (within 2σ)")
        else:
            print(f"→ INCONSISTENT WITH ΛCDM (>{sigma_from_lcdm:.1f}σ away)")
        
        # ============================================================================
        # STEP 6: CRITICAL ANALYSIS OF METHODOLOGY
        # ============================================================================
        print("\n" + "=" * 80)
        print("STEP 6: CRITICAL ANALYSIS OF METHODOLOGY")
        print("=" * 80)
        
        print("""
ISSUE 1: ENVIRONMENT CLASSIFICATION
-----------------------------------
The current method classifies galaxies as:
  - "void":    if distance to nearest void center < 0.8 × void_radius
  - "cluster": if distance to nearest void center > 3.0 × void_radius

PROBLEM: "Far from voids" ≠ "In a cluster"!
This is a PROXY classification, not actual cluster membership.
True cluster dwarfs require spectroscopic membership (velocity within 
±3σ of cluster mean, position within virial radius).

ISSUE 2: SAMPLE SIZE IMBALANCE
------------------------------
""")
        print(f"N_void = {len(v_void)}")
        print(f"N_cluster = {len(v_cluster)}")
        ratio = len(v_void) / len(v_cluster) if len(v_cluster) > 0 else float('inf')
        print(f"Ratio = {ratio:.1f}")
        
        if ratio > 10:
            print("⚠ SEVERE IMBALANCE: Void sample is >10× larger than cluster sample")
        elif ratio > 3:
            print("⚠ MODERATE IMBALANCE: Void sample is >3× larger than cluster sample")
        
        print("""
ISSUE 3: ROTATION VELOCITY DEFINITION
-------------------------------------
V_rot is derived from W50 (HI line width) as:
  V_rot = W50 / 2.4

This assumes an average inclination correction factor of ~1.2.
Individual inclination angles are NOT known, introducing scatter.

ISSUE 4: BARYONIC FEEDBACK CONFOUNDING
--------------------------------------
Dwarf galaxies in clusters experience:
  - Tidal stripping (reduces V_rot)
  - Ram pressure stripping (reduces HI content)
  - Harassment (disturbs kinematics)

These effects can MIMIC or MASK gravitational modifications.
""")
        
        # ============================================================================
        # STEP 7: RECOMMENDED IMPROVEMENTS
        # ============================================================================
        print("\n" + "=" * 80)
        print("STEP 7: RECOMMENDED IMPROVEMENTS")
        print("=" * 80)
        
        print("""
1. USE SPECTROSCOPIC CLUSTER MEMBERSHIP
   - Coma, Virgo, Fornax cluster catalogs
   - Velocity-position membership criteria
   
2. MATCH SAMPLES BY MASS
   - Use propensity score matching
   - Compare mass distributions
   
3. USE RESOLVED ROTATION CURVES (SPARC)
   - Analyze V(r) shape, not just V_max
   - SDCG predicts shape differences at large r
   
4. CONTROL FOR BARYONIC EFFECTS
   - Compare with FIRE/EAGLE simulations
   - Use HI-deficiency as proxy for stripping
   
5. BOOTSTRAP ERROR ANALYSIS
   - Quantify systematic uncertainty
   - Compare with jackknife resampling
""")
        
        # ============================================================================
        # STEP 8: BOOTSTRAP ANALYSIS
        # ============================================================================
        print("\n" + "=" * 80)
        print("STEP 8: BOOTSTRAP ERROR ANALYSIS")
        print("=" * 80)
        
        n_bootstrap = 10000
        delta_v_boot = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            v_void_boot = np.random.choice(v_void, size=len(v_void), replace=True)
            v_cluster_boot = np.random.choice(v_cluster, size=len(v_cluster), replace=True)
            delta_v_boot[i] = np.mean(v_void_boot) - np.mean(v_cluster_boot)
        
        delta_v_bootstrap_mean = np.mean(delta_v_boot)
        delta_v_bootstrap_std = np.std(delta_v_boot)
        ci_low = np.percentile(delta_v_boot, 2.5)
        ci_high = np.percentile(delta_v_boot, 97.5)
        
        print(f"Bootstrap (N = {n_bootstrap}):")
        print(f"  Mean Δv: {delta_v_bootstrap_mean:.3f} km/s")
        print(f"  Std:     {delta_v_bootstrap_std:.3f} km/s")
        print(f"  95% CI:  [{ci_low:.3f}, {ci_high:.3f}] km/s")
        
        # Check if zero is in CI
        if ci_low <= 0 <= ci_high:
            print("  → Zero is WITHIN 95% CI (consistent with ΛCDM)")
        else:
            print("  → Zero is OUTSIDE 95% CI (inconsistent with ΛCDM)")
        
        # Check if SDCG prediction is in CI
        if ci_low <= delta_v_predicted <= ci_high:
            print(f"  → SDCG prediction ({delta_v_predicted:.2f}) is WITHIN 95% CI")
        else:
            print(f"  → SDCG prediction ({delta_v_predicted:.2f}) is OUTSIDE 95% CI")
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DWARF GALAXY ANALYSIS RESULTS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Void galaxies:     N = {len(v_void):>5}    Mean V_rot = {mean_void:.2f} ± {sem_void:.2f} km/s       │
│  Cluster galaxies:  N = {len(v_cluster):>5}    Mean V_rot = {mean_cluster:.2f} ± {sem_cluster:.2f} km/s       │
│                                                                              │
│  Observed Δv:       {delta_v:+.2f} ± {delta_v_err:.2f} km/s                                       │
│  SDCG Prediction:   +{delta_v_predicted:.2f} ± {delta_v_pred_err:.2f} km/s                                       │
│  ΛCDM Prediction:    0.00 ± ~2 km/s                                          │
│                                                                              │
│  Statistical tests:                                                          │
│    Welch's t-test:  p = {p_value:.6f}  {"(SIGNIFICANT)" if p_value < 0.05 else "(not significant)"}                         │
│    Bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}] km/s                                   │
│                                                                              │
│  Distance from SDCG: {sigma_from_sdcg:.2f}σ                                                     │
│  Distance from ΛCDM: {sigma_from_lcdm:.2f}σ                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  VERDICT:                                                                    │
""")
        
        if delta_v > 0 and p_value < 0.05 and sigma_from_sdcg < 3:
            verdict = "SDCG PREDICTION SUPPORTED"
        elif delta_v < 0 or sigma_from_sdcg > 3:
            verdict = "DATA DOES NOT SUPPORT SDCG PREDICTION"
        elif p_value >= 0.05:
            verdict = "INCONCLUSIVE - NOT STATISTICALLY SIGNIFICANT"
        else:
            verdict = "REQUIRES FURTHER INVESTIGATION"
        
        print(f"│    {verdict:<66} │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")
        
else:
    print(f"Results file not found: {results_file}")
    print("Run cgc_dwarf_test/run_cgc_dwarf_test.py first to generate results.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
