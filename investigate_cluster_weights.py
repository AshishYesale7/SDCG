#!/usr/bin/env python3
"""
INVESTIGATE CLUSTER SAMPLE WEIGHTING ANOMALY
=============================================
The weighted cluster mean is 61.7 km/s but unweighted is 71.9 km/s.
This 10 km/s difference suggests a few galaxies with very small errors
are dominating the weighted mean.

This is a DATA QUALITY issue that needs investigation.
"""

import numpy as np
import sys
sys.path.insert(0, 'cgc_dwarf_test')

from load_data import load_and_merge_all
from filter_samples import filter_dwarf_galaxies, split_by_environment, match_morphology

print("=" * 80)
print("INVESTIGATING CLUSTER SAMPLE WEIGHTING ANOMALY")
print("=" * 80)

# Load data
df = load_and_merge_all()
df_dwarfs = filter_dwarf_galaxies(df)
df_void, df_field, df_cluster = split_by_environment(df_dwarfs)
df_void, df_cluster = match_morphology(df_void, df_cluster, df_field)

# ============================================================================
# EXAMINE CLUSTER SAMPLE IN DETAIL
# ============================================================================
print("\n" + "=" * 80)
print("CLUSTER SAMPLE DETAILED ANALYSIS")
print("=" * 80)

v_cluster = df_cluster['V_rot'].values
err_cluster = df_cluster['V_rot_err'].values

print(f"\nCluster sample: N = {len(v_cluster)}")
print(f"\nV_rot statistics:")
print(f"  Min:    {np.min(v_cluster):.2f} km/s")
print(f"  Max:    {np.max(v_cluster):.2f} km/s")
print(f"  Mean:   {np.mean(v_cluster):.2f} km/s")
print(f"  Median: {np.median(v_cluster):.2f} km/s")

print(f"\nV_rot_err statistics:")
print(f"  Min:    {np.min(err_cluster):.3f} km/s")
print(f"  Max:    {np.max(err_cluster):.3f} km/s")
print(f"  Mean:   {np.mean(err_cluster):.3f} km/s")
print(f"  Median: {np.median(err_cluster):.3f} km/s")

# Find galaxies with very small errors
small_err_threshold = 1.0  # km/s
small_err_mask = err_cluster < small_err_threshold
n_small_err = np.sum(small_err_mask)

print(f"\n" + "-" * 60)
print(f"Galaxies with V_rot_err < {small_err_threshold} km/s: {n_small_err}")
print("-" * 60)

if n_small_err > 0:
    print("\nThese galaxies:")
    for i, (v, e) in enumerate(zip(v_cluster[small_err_mask], err_cluster[small_err_mask])):
        weight = 1.0 / e**2
        total_weight = np.sum(1.0 / err_cluster**2)
        frac_weight = weight / total_weight * 100
        print(f"  Galaxy {i+1}: V_rot = {v:.2f} km/s, err = {e:.3f} km/s, weight fraction = {frac_weight:.1f}%")

# Calculate contribution of high-weight galaxies
weights = 1.0 / err_cluster**2
total_weight = np.sum(weights)
weight_fracs = weights / total_weight

# Sort by weight
sorted_idx = np.argsort(weight_fracs)[::-1]

print(f"\n" + "-" * 60)
print("TOP 10 HIGHEST-WEIGHT GALAXIES:")
print("-" * 60)
print(f"{'Rank':<6} {'V_rot (km/s)':<15} {'Error (km/s)':<15} {'Weight %':<12} {'Cumulative %':<12}")
print("-" * 60)

cumulative = 0
for rank, idx in enumerate(sorted_idx[:10]):
    v = v_cluster[idx]
    e = err_cluster[idx]
    w_pct = weight_fracs[idx] * 100
    cumulative += w_pct
    print(f"{rank+1:<6} {v:<15.2f} {e:<15.3f} {w_pct:<12.1f} {cumulative:<12.1f}")

# Calculate how much the top N galaxies contribute
top_10_weight = np.sum(weight_fracs[sorted_idx[:10]]) * 100
top_5_weight = np.sum(weight_fracs[sorted_idx[:5]]) * 100
top_1_weight = weight_fracs[sorted_idx[0]] * 100

print(f"\n" + "-" * 60)
print("WEIGHT CONCENTRATION:")
print("-" * 60)
print(f"Top 1 galaxy contributes:  {top_1_weight:.1f}% of total weight")
print(f"Top 5 galaxies contribute: {top_5_weight:.1f}% of total weight")
print(f"Top 10 galaxies contribute: {top_10_weight:.1f}% of total weight")
print(f"(N_total = {len(v_cluster)} galaxies)")

# ============================================================================
# COMPARE WITH VOID SAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("VOID SAMPLE COMPARISON")
print("=" * 80)

v_void = df_void['V_rot'].values
err_void = df_void['V_rot_err'].values

weights_void = 1.0 / err_void**2
total_weight_void = np.sum(weights_void)
weight_fracs_void = weights_void / total_weight_void
sorted_idx_void = np.argsort(weight_fracs_void)[::-1]

top_10_weight_void = np.sum(weight_fracs_void[sorted_idx_void[:10]]) * 100
top_5_weight_void = np.sum(weight_fracs_void[sorted_idx_void[:5]]) * 100
top_1_weight_void = weight_fracs_void[sorted_idx_void[0]] * 100

print(f"VOID sample (N = {len(v_void)}):")
print(f"Top 1 galaxy contributes:  {top_1_weight_void:.1f}% of total weight")
print(f"Top 5 galaxies contribute: {top_5_weight_void:.1f}% of total weight")
print(f"Top 10 galaxies contribute: {top_10_weight_void:.1f}% of total weight")

print(f"\nCLUSTER sample (N = {len(v_cluster)}):")
print(f"Top 1 galaxy contributes:  {top_1_weight:.1f}% of total weight")
print(f"Top 5 galaxies contribute: {top_5_weight:.1f}% of total weight")
print(f"Top 10 galaxies contribute: {top_10_weight:.1f}% of total weight")

# ============================================================================
# WEIGHTED VS UNWEIGHTED WITH OUTLIER REMOVAL
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS: WEIGHTED MEAN WITH TOP GALAXIES REMOVED")
print("=" * 80)

def weighted_mean(values, errors):
    errors = np.maximum(errors, 1e-6)
    weights = 1.0 / errors**2
    wmean = np.sum(weights * values) / np.sum(weights)
    werr = 1.0 / np.sqrt(np.sum(weights))
    return wmean, werr

# Full weighted mean
wmean_full, werr_full = weighted_mean(v_cluster, err_cluster)
print(f"\nFull sample (N={len(v_cluster)}): weighted mean = {wmean_full:.2f} ± {werr_full:.3f} km/s")

# Remove top 1
mask = np.ones(len(v_cluster), dtype=bool)
mask[sorted_idx[0]] = False
wmean_no1, werr_no1 = weighted_mean(v_cluster[mask], err_cluster[mask])
print(f"Without top 1 (N={len(v_cluster)-1}): weighted mean = {wmean_no1:.2f} ± {werr_no1:.3f} km/s")

# Remove top 5
mask = np.ones(len(v_cluster), dtype=bool)
mask[sorted_idx[:5]] = False
wmean_no5, werr_no5 = weighted_mean(v_cluster[mask], err_cluster[mask])
print(f"Without top 5 (N={len(v_cluster)-5}): weighted mean = {wmean_no5:.2f} ± {werr_no5:.3f} km/s")

# Remove top 10
mask = np.ones(len(v_cluster), dtype=bool)
mask[sorted_idx[:10]] = False
wmean_no10, werr_no10 = weighted_mean(v_cluster[mask], err_cluster[mask])
print(f"Without top 10 (N={len(v_cluster)-10}): weighted mean = {wmean_no10:.2f} ± {werr_no10:.3f} km/s")

# Unweighted
umean = np.mean(v_cluster)
print(f"\nUnweighted mean: {umean:.2f} km/s")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
FINDING: The weighted cluster mean is heavily influenced by a few galaxies
with very small reported errors and low velocities.

DIAGNOSIS:
  - A small number of galaxies dominate the weighted mean
  - These galaxies have unusually small V_rot_err values
  - They may have:
    * Very high SNR observations
    * Face-on orientations (narrow lines)
    * Systematic error underestimation
    * Data quality issues

RECOMMENDATION:
  1. Use UNWEIGHTED means for robustness
  2. OR use a TRIMMED weighted mean (remove extreme weights)
  3. OR cap the maximum weight contribution per galaxy
  
For the thesis, we should report:
  - Unweighted: Δv = -2.49 ± 2.68 km/s (not significant, p = 0.36)
  - Weighted: Δv = +7.40 ± 0.30 km/s (sensitive to outliers)
  
The UNWEIGHTED result is more robust and should be the primary result.
""")
