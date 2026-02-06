#!/usr/bin/env python3
"""
CLEAN DATA ANALYSIS - Using ONLY Original Published Values
===========================================================

This script uses ONLY the original data values as stored in the repository,
WITHOUT any manufactured or estimated rotation velocities.

Original Data Fields:
- void_dwarfs.json: V_HI_km_s (heliocentric), sigma_HI_km_s (HI line width)
- local_group_dwarfs.json: sigma_v_km_s (stellar velocity dispersion)

We use sigma_HI or sigma_v as the kinematic measure, NOT rotation velocity,
unless actual V_rot is published in the source paper.

Author: CGC Analysis
Date: February 3, 2026
"""

import json
import os
import numpy as np
from scipy import stats

# Go up 3 levels: _data_processing -> scripts -> project_root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'dwarfs')

print("="*70)
print("CLEAN DATA ANALYSIS - ORIGINAL VALUES ONLY")
print("="*70)
print("\nPRINCIPLE: Use ONLY data as published, NO estimated values.\n")

# ============================================================================
# LOAD ORIGINAL DATA FILES
# ============================================================================

# Original void dwarfs (Pustilnik+2019, Kreckel+2011)
with open(os.path.join(DATA_DIR, 'void_dwarfs.json'), 'r') as f:
    void_original = json.load(f)

# Local Group dwarfs (McConnachie+2012)
with open(os.path.join(DATA_DIR, 'local_group_dwarfs.json'), 'r') as f:
    lg_original = json.load(f)

print("ORIGINAL DATA STRUCTURE:")
print("-"*70)
print(f"void_dwarfs.json columns: {void_original['columns']}")
print(f"local_group_dwarfs.json columns: {lg_original['columns']}")

# ============================================================================
# EXTRACT KINEMATICS FROM ORIGINAL DATA
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING ORIGINAL KINEMATIC MEASUREMENTS")
print("="*70)

# Void dwarfs - use sigma_HI (HI line width)
void_columns = void_original['columns']
void_name_idx = void_columns.index('Name')
void_sigma_idx = void_columns.index('sigma_HI_km_s')
void_delta_idx = void_columns.index('delta_local')

void_galaxies = []
for row in void_original['data']:
    name = row[void_name_idx]
    sigma_hi = row[void_sigma_idx]
    delta = row[void_delta_idx]
    void_galaxies.append({
        'name': name,
        'sigma': sigma_hi,  # HI line width (sigma)
        'delta': delta,
        'source': 'void_dwarfs.json'
    })

print(f"\nVoid dwarfs from original file: {len(void_galaxies)}")
for g in void_galaxies:
    print(f"  {g['name']:<15} σ_HI = {g['sigma']:.1f} km/s  (δ = {g['delta']:.2f})")

# Local Group - extract void and cluster
lg_columns = lg_original['columns']
lg_name_idx = lg_columns.index('Name')
lg_sigma_idx = lg_columns.index('sigma_v_km_s')
lg_env_idx = lg_columns.index('Environment')

lg_void = []
lg_cluster = []

for row in lg_original['data']:
    name = row[lg_name_idx]
    sigma_v = row[lg_sigma_idx]
    env = row[lg_env_idx]
    
    entry = {
        'name': name,
        'sigma': sigma_v,  # Stellar velocity dispersion
        'source': 'McConnachie+2012'
    }
    
    if env == 'void':
        lg_void.append(entry)
    elif env == 'cluster':
        lg_cluster.append(entry)

print(f"\nLocal Group void dwarfs: {len(lg_void)}")
for g in lg_void:
    print(f"  {g['name']:<15} σ_* = {g['sigma']:.1f} km/s")

print(f"\nLocal Group cluster dwarfs: {len(lg_cluster)}")
for g in lg_cluster:
    print(f"  {g['name']:<15} σ_* = {g['sigma']:.1f} km/s")

# ============================================================================
# ANALYSIS WITH ORIGINAL DATA ONLY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS USING ORIGINAL DATA (NO MANUFACTURED VALUES)")
print("="*70)

# Combine void samples
all_void_sigma = [g['sigma'] for g in void_galaxies + lg_void]
all_cluster_sigma = [g['sigma'] for g in lg_cluster]

print(f"\nCombined samples:")
print(f"  Void:    {len(all_void_sigma)} galaxies")
print(f"  Cluster: {len(all_cluster_sigma)} galaxies")

if len(all_void_sigma) >= 3 and len(all_cluster_sigma) >= 3:
    void_arr = np.array(all_void_sigma)
    cluster_arr = np.array(all_cluster_sigma)
    
    # Match sample sizes
    n_match = min(len(void_arr), len(cluster_arr))
    np.random.seed(42)
    void_matched = np.random.choice(void_arr, size=n_match, replace=False)
    cluster_matched = np.random.choice(cluster_arr, size=n_match, replace=False)
    
    # Statistics
    void_mean = np.mean(void_matched)
    void_sem = np.std(void_matched, ddof=1) / np.sqrt(n_match)
    
    cluster_mean = np.mean(cluster_matched)
    cluster_sem = np.std(cluster_matched, ddof=1) / np.sqrt(n_match)
    
    delta_sigma = void_mean - cluster_mean
    delta_err = np.sqrt(void_sem**2 + cluster_sem**2)
    
    # t-test
    t_stat, t_pval = stats.ttest_ind(void_matched, cluster_matched)
    
    print(f"\n┌{'─'*68}┐")
    print(f"│{'RESULTS (matched N = ' + str(n_match) + ')':^68}│")
    print(f"├{'─'*68}┤")
    print(f"│  Void mean σ:    {void_mean:5.1f} ± {void_sem:.1f} km/s{' '*35}│")
    print(f"│  Cluster mean σ: {cluster_mean:5.1f} ± {cluster_sem:.1f} km/s{' '*35}│")
    print(f"│  Δσ (void - cluster): {delta_sigma:+.1f} ± {delta_err:.1f} km/s{' '*28}│")
    print(f"│  t-test: t = {t_stat:.2f}, p = {t_pval:.4f}{' '*36}│")
    print(f"└{'─'*68}┘")
    
    if delta_sigma > 0:
        print(f"\n  Result: Void dwarfs have HIGHER velocity dispersion")
    else:
        print(f"\n  Result: Cluster dwarfs have HIGHER velocity dispersion")
else:
    print("\n  ERROR: Not enough data for comparison")

# ============================================================================
# IMPORTANT NOTE
# ============================================================================

print("\n" + "="*70)
print("CRITICAL DATA INTEGRITY NOTE")
print("="*70)
print("""
The 'verified_void_dwarfs.json' and 'verified_cluster_dwarfs.json' files
contain V_rot values that appear to have been ESTIMATED or MANUFACTURED
by scripts, not taken directly from published papers.

EVIDENCE:
- 38/38 void V_rot values are round numbers (suspicious)
- Original void_dwarfs.json has sigma_HI, NOT V_rot
- Values do not match any published tables exactly

ACTION REQUIRED:
1. Verify each V_rot value against the original publication
2. Remove any values that cannot be traced to published data
3. Use only sigma_HI or sigma_v if V_rot not available

For thesis purposes, consider:
- Using sigma (velocity dispersion) instead of V_rot
- Or citing only galaxies with published rotation curves
- Document all data sources with table/figure references
""")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
To maintain scientific integrity, the thesis should:

1. REMOVE all manufactured V_rot values
2. USE ONLY original published measurements:
   - sigma_HI from Pustilnik+2019
   - sigma_v from McConnachie+2012
   - Published rotation curves from LITTLE THINGS, VGS

3. If comparing with SDCG prediction:
   - SDCG predicts ΔV_rot ≈ +12 km/s
   - For dispersion-supported dwarfs: V_rot ≈ 2σ
   - Therefore expect Δσ ≈ +6 km/s (if SDCG applies)
""")
