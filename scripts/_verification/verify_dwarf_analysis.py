#!/usr/bin/env python3
"""Comprehensive Dwarf Galaxy Analysis Verification"""
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 70)
print("COMPREHENSIVE DWARF GALAXY ANALYSIS VERIFICATION")
print("=" * 70)

# STEP 1: Load data
results_dir = Path("results")
dwarf_file = results_dir / "cgc_dwarf_analysis.npz"

if dwarf_file.exists():
    data = np.load(dwarf_file, allow_pickle=True)
    print("\nKeys in data file:", list(data.keys()))
    
    # Try different key names
    v_void = None
    v_cluster = None
    
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape'):
            print(f"  {key}: shape={arr.shape if arr.ndim > 0 else 'scalar'}, value={arr if arr.ndim == 0 else arr[:5]}")
    
    # Extract velocities
    if 'v_void' in data:
        v_void = data['v_void']
    elif 'void_velocities' in data:
        v_void = data['void_velocities']
        
    if 'v_cluster' in data:
        v_cluster = data['v_cluster']
    elif 'cluster_velocities' in data:
        v_cluster = data['cluster_velocities']
    
    if v_void is not None and v_cluster is not None:
        print("\n" + "=" * 70)
        print("VELOCITY STATISTICS")
        print("=" * 70)
        print(f"Void dwarfs:    N={len(v_void)}, mean={np.mean(v_void):.2f}, std={np.std(v_void):.2f}")
        print(f"Cluster dwarfs: N={len(v_cluster)}, mean={np.mean(v_cluster):.2f}, std={np.std(v_cluster):.2f}")
        
        delta_v = np.mean(v_void) - np.mean(v_cluster)
        print(f"\nMEASURED: Δv = {delta_v:+.2f} km/s")
        
        t_stat, p_val = stats.ttest_ind(v_void, v_cluster)
        print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")
        
        print("\n" + "=" * 70)
        print("SDCG PREDICTION (Thesis v12: μ_fit = 0.47, μ_eff(void) = 0.149)")
        print("=" * 70)
        # Thesis v12: μ_fit = 0.47 ± 0.03 (MCMC best-fit, 6σ detection)
        # μ_eff(void) = μ_fit × S_avg ≈ 0.47 × 0.31 = 0.149
        mu_eff_void = 0.149  # Effective coupling in voids
        g_z0 = 0.3
        v_typical = 80
        delta_v_pred = v_typical * (np.sqrt(1 + mu_eff_void * g_z0) - 1)
        print(f"With μ_eff(void)={mu_eff_void}, g(z=0)={g_z0}:")
        print(f"PREDICTED: Δv = +{delta_v_pred:.2f} km/s (void dwarfs faster)")
        
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"Prediction: +{delta_v_pred:.1f} km/s")
        print(f"Observed:   {delta_v:+.1f} km/s")
        
        if delta_v > 0:
            print("\n✓ SIGN IS CORRECT - void dwarfs are faster")
        else:
            print("\n✗ SIGN IS OPPOSITE - cluster dwarfs appear faster")
else:
    print(f"File not found: {dwarf_file}")

# STEP 2: Check the data generation script
print("\n" + "=" * 70)
print("CHECKING DATA SOURCE")
print("=" * 70)

script_path = Path("cgc_dwarf_test/run_cgc_dwarf_test.py")
if script_path.exists():
    with open(script_path) as f:
        content = f.read()
    
    print(f"Script: {script_path}")
    print(f"Lines: {len(content.splitlines())}")
    
    # Check for synthetic data indicators
    indicators = {
        'np.random': 'Uses random number generation',
        'simulate': 'Contains simulation code',
        'mock': 'Contains mock data',
        'synthetic': 'Contains synthetic data',
        'SPARC': 'References SPARC database',
        'real': 'References real data'
    }
    
    print("\nData source indicators:")
    for pattern, desc in indicators.items():
        if pattern.lower() in content.lower():
            print(f"  ✓ {desc}")
        else:
            print(f"  - {desc} (not found)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
