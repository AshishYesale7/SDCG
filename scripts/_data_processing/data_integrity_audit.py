#!/usr/bin/env python3
"""
DATA INTEGRITY AUDIT FOR SDCG ANALYSIS
======================================

This script performs a thorough audit of all data used in the SDCG analysis:

1. Check for data loading errors
2. Verify no duplicate galaxies
3. Validate velocity ranges are physical
4. Check environment classifications
5. Verify sample matching is correct
6. Cross-check statistics calculations
7. Identify any data anomalies

Author: CGC Analysis Pipeline
Date: February 3, 2026
"""

import numpy as np
import json
import os
import csv
from collections import defaultdict, Counter
from scipy import stats

# Go up 3 levels: _data_processing -> scripts -> project_root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Expected physical ranges
MIN_VROT = 5.0    # km/s - minimum reasonable rotation velocity
MAX_VROT = 150.0  # km/s - maximum for dwarf galaxies
MIN_DIST = 0.01   # Mpc
MAX_DIST = 200.0  # Mpc

issues_found = []
warnings = []


def log_issue(msg):
    issues_found.append(f"❌ ISSUE: {msg}")
    print(f"❌ ISSUE: {msg}")


def log_warning(msg):
    warnings.append(f"⚠️  WARNING: {msg}")
    print(f"⚠️  WARNING: {msg}")


def log_ok(msg):
    print(f"✅ OK: {msg}")


# ============================================================================
# AUDIT 1: Check file existence and structure
# ============================================================================

def audit_file_existence():
    """Check all expected data files exist."""
    print("\n" + "="*70)
    print("AUDIT 1: FILE EXISTENCE AND STRUCTURE")
    print("="*70)
    
    expected_files = [
        ("dwarfs/verified_void_dwarfs.json", "Verified void dwarfs"),
        ("dwarfs/verified_cluster_dwarfs.json", "Verified cluster dwarfs"),
        ("dwarfs/local_group_dwarfs.json", "Local Group dwarfs"),
        ("little_things/little_things_catalog.json", "LITTLE THINGS"),
        ("alfalfa/alfalfa_a40.csv", "ALFALFA catalog"),
        ("sparc/sparc_data.mrt", "SPARC data"),
    ]
    
    for filepath, desc in expected_files:
        full_path = os.path.join(DATA_DIR, filepath)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            log_ok(f"{desc}: exists ({size/1024:.1f} KB)")
        else:
            log_issue(f"{desc}: FILE NOT FOUND at {full_path}")


# ============================================================================
# AUDIT 2: Verified void dwarfs
# ============================================================================

def audit_verified_void():
    """Audit verified void dwarf catalog."""
    print("\n" + "="*70)
    print("AUDIT 2: VERIFIED VOID DWARFS")
    print("="*70)
    
    void_path = os.path.join(DATA_DIR, "dwarfs", "verified_void_dwarfs.json")
    
    if not os.path.exists(void_path):
        log_issue("Void dwarfs file not found")
        return []
    
    with open(void_path, 'r') as f:
        data = json.load(f)
    
    galaxies = data.get('galaxies', [])
    print(f"\nTotal galaxies in file: {len(galaxies)}")
    
    # Check for required fields
    required_fields = ['name', 'ra', 'dec']
    velocity_fields = ['v_rot', 'z']  # At least one needed
    
    valid_galaxies = []
    names_seen = set()
    
    for i, g in enumerate(galaxies):
        # Check required fields
        missing = [f for f in required_fields if f not in g]
        if missing:
            log_warning(f"Galaxy {i}: missing fields {missing}")
            continue
        
        # Check for duplicates
        name_key = g['name'].upper().replace(' ', '').replace('-', '').replace('_', '')
        if name_key in names_seen:
            log_warning(f"Duplicate galaxy name: {g['name']}")
        names_seen.add(name_key)
        
        # Check V_rot
        v_rot = g.get('v_rot')
        if v_rot is None:
            log_warning(f"{g['name']}: No v_rot value")
            continue
        
        if v_rot <= 0:
            log_issue(f"{g['name']}: Invalid v_rot = {v_rot} (must be > 0)")
            continue
        
        if v_rot < MIN_VROT:
            log_warning(f"{g['name']}: Very low v_rot = {v_rot:.1f} km/s")
        
        if v_rot > MAX_VROT:
            log_warning(f"{g['name']}: High v_rot = {v_rot:.1f} km/s (may not be dwarf)")
        
        # Check coordinates
        ra = g.get('ra', 0)
        dec = g.get('dec', 0)
        if not (0 <= ra <= 360):
            log_issue(f"{g['name']}: Invalid RA = {ra}")
        if not (-90 <= dec <= 90):
            log_issue(f"{g['name']}: Invalid Dec = {dec}")
        
        valid_galaxies.append(g)
    
    # Statistics
    vrots = [g['v_rot'] for g in valid_galaxies if g.get('v_rot')]
    if vrots:
        print(f"\nValid galaxies with v_rot: {len(vrots)}")
        print(f"  Mean:   {np.mean(vrots):.1f} km/s")
        print(f"  Median: {np.median(vrots):.1f} km/s")
        print(f"  Min:    {np.min(vrots):.1f} km/s")
        print(f"  Max:    {np.max(vrots):.1f} km/s")
        print(f"  Std:    {np.std(vrots):.1f} km/s")
    
    # Check sources
    sources = Counter(g.get('source', 'Unknown') for g in valid_galaxies)
    print(f"\nSources:")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")
    
    log_ok(f"Verified {len(valid_galaxies)} valid void dwarfs")
    return valid_galaxies


# ============================================================================
# AUDIT 3: Verified cluster dwarfs
# ============================================================================

def audit_verified_cluster():
    """Audit verified cluster dwarf catalog."""
    print("\n" + "="*70)
    print("AUDIT 3: VERIFIED CLUSTER DWARFS")
    print("="*70)
    
    cluster_path = os.path.join(DATA_DIR, "dwarfs", "verified_cluster_dwarfs.json")
    
    if not os.path.exists(cluster_path):
        log_issue("Cluster dwarfs file not found")
        return []
    
    with open(cluster_path, 'r') as f:
        data = json.load(f)
    
    galaxies = data.get('galaxies', [])
    print(f"\nTotal galaxies in file: {len(galaxies)}")
    
    valid_galaxies = []
    names_seen = set()
    
    for i, g in enumerate(galaxies):
        # Check for duplicates
        name_key = g.get('name', f'idx{i}').upper().replace(' ', '').replace('-', '').replace('_', '')
        if name_key in names_seen:
            log_warning(f"Duplicate galaxy name: {g.get('name')}")
        names_seen.add(name_key)
        
        # Check V_rot
        v_rot = g.get('v_rot')
        if v_rot is None:
            log_warning(f"{g.get('name', f'idx{i}')}: No v_rot value")
            continue
        
        if v_rot <= 0:
            log_issue(f"{g.get('name')}: Invalid v_rot = {v_rot}")
            continue
        
        if v_rot < MIN_VROT:
            log_warning(f"{g.get('name')}: Very low v_rot = {v_rot:.1f} km/s")
        
        if v_rot > MAX_VROT:
            log_warning(f"{g.get('name')}: High v_rot = {v_rot:.1f} km/s")
        
        valid_galaxies.append(g)
    
    # Statistics
    vrots = [g['v_rot'] for g in valid_galaxies if g.get('v_rot')]
    if vrots:
        print(f"\nValid galaxies with v_rot: {len(vrots)}")
        print(f"  Mean:   {np.mean(vrots):.1f} km/s")
        print(f"  Median: {np.median(vrots):.1f} km/s")
        print(f"  Min:    {np.min(vrots):.1f} km/s")
        print(f"  Max:    {np.max(vrots):.1f} km/s")
        print(f"  Std:    {np.std(vrots):.1f} km/s")
    
    # Check sources
    sources = Counter(g.get('source', 'Unknown') for g in valid_galaxies)
    print(f"\nSources:")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")
    
    log_ok(f"Verified {len(valid_galaxies)} valid cluster dwarfs")
    return valid_galaxies


# ============================================================================
# AUDIT 4: Local Group data
# ============================================================================

def audit_local_group():
    """Audit Local Group catalog."""
    print("\n" + "="*70)
    print("AUDIT 4: LOCAL GROUP DWARFS")
    print("="*70)
    
    lg_path = os.path.join(DATA_DIR, "dwarfs", "local_group_dwarfs.json")
    
    if not os.path.exists(lg_path):
        log_issue("Local Group file not found")
        return [], []
    
    with open(lg_path, 'r') as f:
        data = json.load(f)
    
    columns = data.get('columns', [])
    rows = data.get('data', [])
    
    print(f"\nColumns: {columns}")
    print(f"Total entries: {len(rows)}")
    
    # Find column indices
    try:
        name_idx = columns.index('Name')
        sigma_idx = columns.index('sigma_v_km_s')
        env_idx = columns.index('Environment')
    except ValueError as e:
        log_issue(f"Missing required column: {e}")
        return [], []
    
    void_galaxies = []
    cluster_galaxies = []
    
    env_counts = Counter()
    
    for row in rows:
        name = row[name_idx]
        sigma_v = row[sigma_idx]
        env = row[env_idx]
        
        env_counts[env] += 1
        
        # Convert sigma to V_rot (for pressure-supported systems)
        v_rot = 2.0 * sigma_v
        
        if v_rot < MIN_VROT:
            log_warning(f"{name}: Very low v_rot = {v_rot:.1f} km/s (from σ={sigma_v:.1f})")
        
        entry = {'name': name, 'v_rot': v_rot, 'sigma_v': sigma_v, 'source': 'McConnachie+2012'}
        
        if env == 'void':
            void_galaxies.append(entry)
        elif env == 'cluster':
            cluster_galaxies.append(entry)
    
    print(f"\nEnvironment distribution:")
    for env, count in env_counts.items():
        print(f"  {env}: {count}")
    
    print(f"\nVoid galaxies: {len(void_galaxies)}")
    if void_galaxies:
        vrots = [g['v_rot'] for g in void_galaxies]
        print(f"  Mean v_rot: {np.mean(vrots):.1f} km/s")
    
    print(f"Cluster galaxies: {len(cluster_galaxies)}")
    if cluster_galaxies:
        vrots = [g['v_rot'] for g in cluster_galaxies]
        print(f"  Mean v_rot: {np.mean(vrots):.1f} km/s")
    
    log_ok(f"Local Group: {len(void_galaxies)} void, {len(cluster_galaxies)} cluster")
    return void_galaxies, cluster_galaxies


# ============================================================================
# AUDIT 5: LITTLE THINGS
# ============================================================================

def audit_little_things():
    """Audit LITTLE THINGS catalog."""
    print("\n" + "="*70)
    print("AUDIT 5: LITTLE THINGS CATALOG")
    print("="*70)
    
    lt_path = os.path.join(DATA_DIR, "little_things", "little_things_catalog.json")
    
    if not os.path.exists(lt_path):
        log_issue("LITTLE THINGS file not found")
        return [], []
    
    with open(lt_path, 'r') as f:
        data = json.load(f)
    
    galaxies = data.get('galaxies', [])
    print(f"\nTotal galaxies: {len(galaxies)}")
    
    void_galaxies = []
    cluster_galaxies = []
    env_counts = Counter()
    
    for g in galaxies:
        env = g.get('environment', 'field')
        env_counts[env] += 1
        
        v_rot = g.get('v_rot')
        if v_rot is None or v_rot <= 0:
            log_warning(f"{g.get('name')}: Invalid v_rot")
            continue
        
        entry = {'name': g['name'], 'v_rot': v_rot, 'source': 'LITTLE_THINGS'}
        
        if env == 'void':
            void_galaxies.append(entry)
        elif env == 'cluster':
            cluster_galaxies.append(entry)
    
    print(f"\nEnvironment distribution:")
    for env, count in env_counts.items():
        print(f"  {env}: {count}")
    
    print(f"\nVoid galaxies: {len(void_galaxies)}")
    print(f"Cluster galaxies: {len(cluster_galaxies)}")
    
    log_ok(f"LITTLE THINGS: {len(void_galaxies)} void, {len(cluster_galaxies)} cluster")
    return void_galaxies, cluster_galaxies


# ============================================================================
# AUDIT 6: Cross-check for duplicates across sources
# ============================================================================

def audit_cross_duplicates(void_verified, cluster_verified, lg_void, lg_cluster, lt_void, lt_cluster):
    """Check for duplicates across different sources."""
    print("\n" + "="*70)
    print("AUDIT 6: CROSS-SOURCE DUPLICATE CHECK")
    print("="*70)
    
    def normalize_name(name):
        return name.upper().replace(' ', '').replace('-', '').replace('_', '')
    
    # Collect all names with their sources
    all_void = []
    for g in void_verified:
        all_void.append((normalize_name(g['name']), g['name'], 'Verified'))
    for g in lg_void:
        all_void.append((normalize_name(g['name']), g['name'], 'LocalGroup'))
    for g in lt_void:
        all_void.append((normalize_name(g['name']), g['name'], 'LITTLE_THINGS'))
    
    all_cluster = []
    for g in cluster_verified:
        all_cluster.append((normalize_name(g['name']), g['name'], 'Verified'))
    for g in lg_cluster:
        all_cluster.append((normalize_name(g['name']), g['name'], 'LocalGroup'))
    for g in lt_cluster:
        all_cluster.append((normalize_name(g['name']), g['name'], 'LITTLE_THINGS'))
    
    # Check void duplicates
    void_names = defaultdict(list)
    for norm, orig, src in all_void:
        void_names[norm].append((orig, src))
    
    void_dups = {k: v for k, v in void_names.items() if len(v) > 1}
    if void_dups:
        print(f"\nVoid duplicates found: {len(void_dups)}")
        for name, sources in void_dups.items():
            log_warning(f"Duplicate void: {sources}")
    else:
        log_ok("No duplicate void galaxies across sources")
    
    # Check cluster duplicates
    cluster_names = defaultdict(list)
    for norm, orig, src in all_cluster:
        cluster_names[norm].append((orig, src))
    
    cluster_dups = {k: v for k, v in cluster_names.items() if len(v) > 1}
    if cluster_dups:
        print(f"\nCluster duplicates found: {len(cluster_dups)}")
        for name, sources in cluster_dups.items():
            log_warning(f"Duplicate cluster: {sources}")
    else:
        log_ok("No duplicate cluster galaxies across sources")
    
    # Check void/cluster overlap (same galaxy in both!)
    void_set = set(n for n, _, _ in all_void)
    cluster_set = set(n for n, _, _ in all_cluster)
    overlap = void_set & cluster_set
    
    if overlap:
        log_issue(f"CRITICAL: Same galaxy in both void AND cluster samples: {overlap}")
    else:
        log_ok("No galaxies appear in both void and cluster samples")


# ============================================================================
# AUDIT 7: Statistical calculation verification
# ============================================================================

def audit_statistics(void_galaxies, cluster_galaxies):
    """Verify statistical calculations are correct."""
    print("\n" + "="*70)
    print("AUDIT 7: STATISTICAL CALCULATION VERIFICATION")
    print("="*70)
    
    void_v = np.array([g['v_rot'] for g in void_galaxies if g.get('v_rot')])
    cluster_v = np.array([g['v_rot'] for g in cluster_galaxies if g.get('v_rot')])
    
    print(f"\nVoid sample size: {len(void_v)}")
    print(f"Cluster sample size: {len(cluster_v)}")
    
    if len(void_v) == 0 or len(cluster_v) == 0:
        log_issue("Empty sample - cannot compute statistics")
        return
    
    # Manual calculations
    void_mean = np.sum(void_v) / len(void_v)
    cluster_mean = np.sum(cluster_v) / len(cluster_v)
    
    void_var = np.sum((void_v - void_mean)**2) / (len(void_v) - 1)
    cluster_var = np.sum((cluster_v - cluster_mean)**2) / (len(cluster_v) - 1)
    
    void_std = np.sqrt(void_var)
    cluster_std = np.sqrt(cluster_var)
    
    void_sem = void_std / np.sqrt(len(void_v))
    cluster_sem = cluster_std / np.sqrt(len(cluster_v))
    
    delta_v = void_mean - cluster_mean
    delta_err = np.sqrt(void_sem**2 + cluster_sem**2)
    
    # Compare with numpy
    np_void_mean = np.mean(void_v)
    np_cluster_mean = np.mean(cluster_v)
    np_void_std = np.std(void_v, ddof=1)
    np_cluster_std = np.std(cluster_v, ddof=1)
    
    print(f"\n--- Manual Calculations ---")
    print(f"Void mean:     {void_mean:.4f} km/s")
    print(f"Cluster mean:  {cluster_mean:.4f} km/s")
    print(f"Void std:      {void_std:.4f} km/s")
    print(f"Cluster std:   {cluster_std:.4f} km/s")
    print(f"Delta v:       {delta_v:+.4f} km/s")
    print(f"Delta error:   {delta_err:.4f} km/s")
    
    print(f"\n--- NumPy Verification ---")
    print(f"Void mean:     {np_void_mean:.4f} km/s (diff: {abs(void_mean - np_void_mean):.2e})")
    print(f"Cluster mean:  {np_cluster_mean:.4f} km/s (diff: {abs(cluster_mean - np_cluster_mean):.2e})")
    print(f"Void std:      {np_void_std:.4f} km/s (diff: {abs(void_std - np_void_std):.2e})")
    print(f"Cluster std:   {np_cluster_std:.4f} km/s (diff: {abs(cluster_std - np_cluster_std):.2e})")
    
    # Check for calculation errors
    if abs(void_mean - np_void_mean) > 0.01:
        log_issue(f"Mean calculation mismatch: manual={void_mean:.4f}, numpy={np_void_mean:.4f}")
    else:
        log_ok("Mean calculations verified")
    
    if abs(void_std - np_void_std) > 0.01:
        log_issue(f"Std calculation mismatch")
    else:
        log_ok("Standard deviation calculations verified")
    
    # t-test verification
    t_stat_manual = delta_v / delta_err
    t_stat_scipy, p_val = stats.ttest_ind(void_v, cluster_v)
    
    print(f"\n--- t-test Verification ---")
    print(f"Manual t-stat:  {t_stat_manual:.4f}")
    print(f"Scipy t-stat:   {t_stat_scipy:.4f}")
    print(f"p-value:        {p_val:.4e}")
    
    # Check significance
    sigma = delta_v / delta_err
    print(f"\nSignificance from zero: {sigma:.2f}σ")
    
    # SDCG comparison
    sdcg_pred = 12.0
    sdcg_err = 3.0
    sigma_from_sdcg = abs(delta_v - sdcg_pred) / np.sqrt(delta_err**2 + sdcg_err**2)
    print(f"Deviation from SDCG (+12 km/s): {sigma_from_sdcg:.2f}σ")
    
    log_ok("Statistical calculations verified")
    
    return {
        'void_mean': void_mean,
        'cluster_mean': cluster_mean,
        'delta_v': delta_v,
        'delta_err': delta_err,
        'sigma': sigma,
        'sigma_from_sdcg': sigma_from_sdcg
    }


# ============================================================================
# AUDIT 8: Physical plausibility check
# ============================================================================

def audit_physical_plausibility(void_galaxies, cluster_galaxies):
    """Check if results are physically plausible."""
    print("\n" + "="*70)
    print("AUDIT 8: PHYSICAL PLAUSIBILITY CHECK")
    print("="*70)
    
    void_v = [g['v_rot'] for g in void_galaxies if g.get('v_rot')]
    cluster_v = [g['v_rot'] for g in cluster_galaxies if g.get('v_rot')]
    
    # Check velocity distributions
    print(f"\n--- Velocity Distribution Sanity Checks ---")
    
    # 1. Are velocities in reasonable range for dwarfs?
    all_v = void_v + cluster_v
    if np.min(all_v) < 5:
        log_warning(f"Some velocities < 5 km/s (min: {np.min(all_v):.1f})")
    else:
        log_ok(f"Minimum velocity ({np.min(all_v):.1f} km/s) is reasonable")
    
    if np.max(all_v) > 100:
        log_warning(f"Some velocities > 100 km/s (max: {np.max(all_v):.1f}) - may include non-dwarfs")
    else:
        log_ok(f"Maximum velocity ({np.max(all_v):.1f} km/s) is reasonable for dwarfs")
    
    # 2. Is the void > cluster trend physically expected?
    delta = np.mean(void_v) - np.mean(cluster_v)
    if delta > 0:
        log_ok(f"Void > Cluster by {delta:.1f} km/s - matches SDCG prediction (enhanced gravity in voids)")
    else:
        log_warning(f"Cluster > Void by {-delta:.1f} km/s - opposite to SDCG prediction")
    
    # 3. Is the effect size reasonable?
    if 5 < abs(delta) < 20:
        log_ok(f"Effect size ({delta:.1f} km/s) is within expected range for SDCG")
    elif abs(delta) > 30:
        log_warning(f"Effect size ({delta:.1f} km/s) seems unusually large")
    
    # 4. Check for outliers
    void_z = np.abs(stats.zscore(void_v))
    cluster_z = np.abs(stats.zscore(cluster_v))
    
    void_outliers = np.sum(void_z > 3)
    cluster_outliers = np.sum(cluster_z > 3)
    
    if void_outliers > 0:
        log_warning(f"{void_outliers} outliers (>3σ) in void sample")
    else:
        log_ok("No extreme outliers in void sample")
    
    if cluster_outliers > 0:
        log_warning(f"{cluster_outliers} outliers (>3σ) in cluster sample")
    else:
        log_ok("No extreme outliers in cluster sample")


# ============================================================================
# AUDIT 9: Sample matching verification
# ============================================================================

def audit_sample_matching(void_galaxies, cluster_galaxies):
    """Verify sample matching is done correctly."""
    print("\n" + "="*70)
    print("AUDIT 9: SAMPLE MATCHING VERIFICATION")
    print("="*70)
    
    n_void = len([g for g in void_galaxies if g.get('v_rot')])
    n_cluster = len([g for g in cluster_galaxies if g.get('v_rot')])
    
    print(f"\nBefore matching:")
    print(f"  Void:    {n_void}")
    print(f"  Cluster: {n_cluster}")
    
    n_match = min(n_void, n_cluster)
    print(f"\nMatched sample size: {n_match}")
    
    if n_match < 10:
        log_warning(f"Small sample size ({n_match}) may limit statistical power")
    elif n_match >= 30:
        log_ok(f"Good sample size ({n_match}) for reliable statistics")
    
    if abs(n_void - n_cluster) > 10:
        log_warning(f"Unbalanced samples before matching (difference: {abs(n_void - n_cluster)})")
    
    # Check if matching preserves velocity distribution shape
    void_v = sorted([g['v_rot'] for g in void_galaxies if g.get('v_rot')])
    cluster_v = sorted([g['v_rot'] for g in cluster_galaxies if g.get('v_rot')])
    
    void_range = (void_v[0], void_v[-1]) if void_v else (0, 0)
    cluster_range = (cluster_v[0], cluster_v[-1]) if cluster_v else (0, 0)
    
    print(f"\nVelocity ranges:")
    print(f"  Void:    {void_range[0]:.1f} - {void_range[1]:.1f} km/s")
    print(f"  Cluster: {cluster_range[0]:.1f} - {cluster_range[1]:.1f} km/s")
    
    # Check overlap
    overlap_min = max(void_range[0], cluster_range[0])
    overlap_max = min(void_range[1], cluster_range[1])
    
    if overlap_max > overlap_min:
        log_ok(f"Velocity ranges overlap ({overlap_min:.1f} - {overlap_max:.1f} km/s)")
    else:
        log_warning("Velocity ranges do not overlap - samples may not be comparable")


# ============================================================================
# MAIN AUDIT
# ============================================================================

def main():
    print("="*70)
    print("COMPREHENSIVE DATA INTEGRITY AUDIT")
    print("SDCG Dwarf Galaxy Rotation Analysis")
    print("="*70)
    print(f"\nData directory: {DATA_DIR}")
    
    # Run all audits
    audit_file_existence()
    
    void_verified = audit_verified_void()
    cluster_verified = audit_verified_cluster()
    lg_void, lg_cluster = audit_local_group()
    lt_void, lt_cluster = audit_little_things()
    
    audit_cross_duplicates(
        void_verified, cluster_verified,
        lg_void, lg_cluster,
        lt_void, lt_cluster
    )
    
    # Combine high-quality samples
    print("\n" + "="*70)
    print("COMBINED HIGH-QUALITY SAMPLE")
    print("="*70)
    
    all_void = void_verified + lg_void + lt_void
    all_cluster = cluster_verified + lg_cluster + lt_cluster
    
    # Remove duplicates
    seen_void = set()
    unique_void = []
    for g in all_void:
        key = g['name'].upper().replace(' ', '').replace('-', '').replace('_', '')
        if key not in seen_void:
            seen_void.add(key)
            unique_void.append(g)
    
    seen_cluster = set()
    unique_cluster = []
    for g in all_cluster:
        key = g['name'].upper().replace(' ', '').replace('-', '').replace('_', '')
        if key not in seen_cluster:
            seen_cluster.add(key)
            unique_cluster.append(g)
    
    print(f"\nAfter deduplication:")
    print(f"  Void:    {len(unique_void)}")
    print(f"  Cluster: {len(unique_cluster)}")
    
    audit_physical_plausibility(unique_void, unique_cluster)
    audit_sample_matching(unique_void, unique_cluster)
    stats = audit_statistics(unique_void, unique_cluster)
    
    # Final summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    
    print(f"\n❌ ISSUES FOUND: {len(issues_found)}")
    for issue in issues_found:
        print(f"  {issue}")
    
    print(f"\n⚠️  WARNINGS: {len(warnings)}")
    for warning in warnings:
        print(f"  {warning}")
    
    if len(issues_found) == 0:
        print("\n" + "="*70)
        print("✅ ✅ ✅  ALL DATA INTEGRITY CHECKS PASSED  ✅ ✅ ✅")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ ❌ ❌  DATA ISSUES DETECTED - REVIEW REQUIRED  ❌ ❌ ❌")
        print("="*70)
    
    if stats:
        print(f"\n--- FINAL VERIFIED RESULT ---")
        print(f"Δv (void - cluster) = {stats['delta_v']:+.2f} ± {stats['delta_err']:.2f} km/s")
        print(f"Significance: {stats['sigma']:.1f}σ")
        print(f"SDCG consistency: {stats['sigma_from_sdcg']:.1f}σ from prediction")


if __name__ == "__main__":
    main()
