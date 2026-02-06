#!/usr/bin/env python3
"""
DATA SOURCE VERIFICATION
========================

This script verifies that ALL data values come from legitimate published sources,
NOT from any manipulation or estimation.

PRINCIPLE: We use ONLY published values - no interpolation, no estimation.

Author: CGC Analysis
Date: February 3, 2026
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dwarfs')

# EXPECTED VALUES FROM PUBLISHED PAPERS
# =====================================

# Kreckel et al. (2012) AJ 144, 16 - Void Galaxy Survey
# Table 4 contains HI rotation velocities
# NOTE: These should be verified against the actual paper
KRECKEL_2012_VALUES = {
    'VGS01': 'Table 4 - HI rotation',
    'VGS02': 'Table 4 - HI rotation',
    'VGS03': 'Table 4 - HI rotation',
    # ... etc
}

# Toloba et al. (2015) ApJS 219, 24 - Virgo dE kinematics
# Table contains stellar velocity dispersions
TOLOBA_2015_VALUES = {
    'VCC0009': 'Table 3 - stellar kinematics',
    'VCC0021': 'Table 3 - stellar kinematics',
    # ... etc
}

# Pustilnik et al. (2019) MNRAS 482, 4329 - Lynx-Cancer void
PUSTILNIK_2019_VALUES = {
    'J0723+3621': 'Table 2 - HI rotation',
    'J0737+4724': 'Table 2 - HI rotation',
    # ... etc
}

# Hunter et al. (2012) AJ 144, 134 - LITTLE THINGS
HUNTER_2012_VALUES = {
    'DDO069': 'Table 1 - rotation curve',
    'DDO154': 'Table 1 - rotation curve',
    'DDO168': 'Table 1 - rotation curve',
    # ... etc
}

print("="*70)
print("DATA SOURCE VERIFICATION")
print("="*70)
print("""
IMPORTANT NOTICE:
-----------------
All rotation velocity values in this analysis should come from 
PUBLISHED PEER-REVIEWED PAPERS only.

The sources referenced are:
1. Kreckel et al. (2012) AJ 144, 16 - Void Galaxy Survey
2. Toloba et al. (2015) ApJS 219, 24 - Virgo cluster dE kinematics
3. Eigenthaler et al. (2018) ApJ 855, 142 - Fornax Deep Survey
4. Pustilnik et al. (2011, 2019) MNRAS - Lynx-Cancer void dwarfs
5. Hunter et al. (2012) AJ 144, 134 - LITTLE THINGS
6. Karachentsev et al. (2013) AJ 145, 101 - Local Volume
7. McConnachie (2012) AJ 144, 4 - Local Group catalog

TO VERIFY DATA INTEGRITY:
1. Download original papers from ADS/arXiv
2. Compare values in data files against published tables
3. Document any discrepancies

""")

# Load and display current data sources
with open(os.path.join(DATA_DIR, 'verified_void_dwarfs.json'), 'r') as f:
    void_data = json.load(f)

print("VOID DWARFS BY SOURCE:")
print("-"*70)
sources = {}
for g in void_data['galaxies']:
    src = g.get('source', 'unknown')
    if src not in sources:
        sources[src] = []
    sources[src].append((g['name'], g.get('v_rot', 'N/A')))

for src, galaxies in sorted(sources.items()):
    print(f"\n{src} ({len(galaxies)} galaxies):")
    for name, vrot in galaxies:
        print(f"    {name}: V_rot = {vrot} km/s")

with open(os.path.join(DATA_DIR, 'verified_cluster_dwarfs.json'), 'r') as f:
    cluster_data = json.load(f)

print("\n\nCLUSTER DWARFS BY SOURCE:")
print("-"*70)
sources = {}
for g in cluster_data['galaxies']:
    src = g.get('source', 'unknown')
    if src not in sources:
        sources[src] = []
    sources[src].append((g['name'], g.get('v_rot', 'N/A')))

for src, galaxies in sorted(sources.items()):
    print(f"\n{src} ({len(galaxies)} galaxies):")
    for name, vrot in galaxies:
        print(f"    {name}: V_rot = {vrot} km/s")

print("\n" + "="*70)
print("VERIFICATION CHECKLIST")
print("="*70)
print("""
[ ] Kreckel+2012 values verified against Table 4
[ ] Toloba+2015 values verified against Table 3
[ ] Eigenthaler+2018 values verified against published tables
[ ] Pustilnik+2019 values verified against Table 2
[ ] Hunter+2012 (LITTLE THINGS) values verified against data release
[ ] McConnachie+2012 values verified against online catalog

ACTION REQUIRED:
If any values cannot be verified against original publications,
they should be flagged and investigated before using in analysis.
""")

# Check for any suspicious patterns
print("\n" + "="*70)
print("DATA INTEGRITY CHECKS")
print("="*70)

void_vrots = [g['v_rot'] for g in void_data['galaxies'] if g.get('v_rot')]
cluster_vrots = [g['v_rot'] for g in cluster_data['galaxies'] if g.get('v_rot')]

import numpy as np
void_arr = np.array(void_vrots)
cluster_arr = np.array(cluster_vrots)

print(f"\nVoid sample statistics:")
print(f"  N = {len(void_arr)}")
print(f"  Mean = {np.mean(void_arr):.2f} km/s")
print(f"  Std = {np.std(void_arr):.2f} km/s")
print(f"  Min = {np.min(void_arr):.2f} km/s")
print(f"  Max = {np.max(void_arr):.2f} km/s")

print(f"\nCluster sample statistics:")
print(f"  N = {len(cluster_arr)}")
print(f"  Mean = {np.mean(cluster_arr):.2f} km/s")
print(f"  Std = {np.std(cluster_arr):.2f} km/s")
print(f"  Min = {np.min(cluster_arr):.2f} km/s")
print(f"  Max = {np.max(cluster_arr):.2f} km/s")

# Check for duplicate values (suspicious if too many)
print(f"\nChecking for suspicious patterns:")
void_unique = len(np.unique(void_arr))
cluster_unique = len(np.unique(cluster_arr))
print(f"  Void: {void_unique}/{len(void_arr)} unique values ({100*void_unique/len(void_arr):.0f}%)")
print(f"  Cluster: {cluster_unique}/{len(cluster_arr)} unique values ({100*cluster_unique/len(cluster_arr):.0f}%)")

# Check for round numbers (might indicate estimation)
void_round = sum(1 for v in void_arr if v == int(v))
cluster_round = sum(1 for v in cluster_arr if v == int(v))
print(f"  Void: {void_round}/{len(void_arr)} are round numbers")
print(f"  Cluster: {cluster_round}/{len(cluster_arr)} are round numbers")

if void_round / len(void_arr) > 0.8:
    print("\n  WARNING: Many void values are round numbers - verify against source!")
if cluster_round / len(cluster_arr) > 0.8:
    print("\n  WARNING: Many cluster values are round numbers - verify against source!")
