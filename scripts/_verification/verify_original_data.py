#!/usr/bin/env python3
"""
Verify Original Data - No Modifications
========================================

This script ONLY reads and displays the original data values.
It does NOT modify any data.

Author: CGC Analysis
Date: February 3, 2026
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dwarfs')

print("="*70)
print("DATA VERIFICATION - ORIGINAL VALUES ONLY (NO MODIFICATIONS)")
print("="*70)

# Check void dwarfs
void_path = os.path.join(DATA_DIR, 'verified_void_dwarfs.json')
with open(void_path, 'r') as f:
    void_data = json.load(f)

print('\nVOID DWARFS (Original Data):')
print('-'*70)
n_with_vrot = 0
n_without_vrot = 0

for g in void_data['galaxies']:
    vrot = g.get('v_rot')
    source = g.get('source', 'unknown')
    if vrot is not None and vrot > 0:
        n_with_vrot += 1
        print(f"  {g['name']:<20} V_rot = {vrot:6.1f} km/s   [{source}]")
    else:
        n_without_vrot += 1
        print(f"  {g['name']:<20} V_rot = MISSING          [{source}]")

print(f'\nSummary: {len(void_data["galaxies"])} total | {n_with_vrot} with V_rot | {n_without_vrot} missing V_rot')

# Check cluster dwarfs
cluster_path = os.path.join(DATA_DIR, 'verified_cluster_dwarfs.json')
with open(cluster_path, 'r') as f:
    cluster_data = json.load(f)

print('\n\nCLUSTER DWARFS (Original Data):')
print('-'*70)
n_with_vrot_c = 0
n_without_vrot_c = 0

for g in cluster_data['galaxies']:
    vrot = g.get('v_rot')
    source = g.get('source', 'unknown')
    if vrot is not None and vrot > 0:
        n_with_vrot_c += 1
        print(f"  {g['name']:<20} V_rot = {vrot:6.1f} km/s   [{source}]")
    else:
        n_without_vrot_c += 1
        print(f"  {g['name']:<20} V_rot = MISSING          [{source}]")

print(f'\nSummary: {len(cluster_data["galaxies"])} total | {n_with_vrot_c} with V_rot | {n_without_vrot_c} missing V_rot')

# Check LITTLE THINGS
lt_path = os.path.join(os.path.dirname(DATA_DIR), 'little_things', 'little_things_catalog.json')
if os.path.exists(lt_path):
    with open(lt_path, 'r') as f:
        lt_data = json.load(f)
    
    print('\n\nLITTLE THINGS (Original Data):')
    print('-'*70)
    n_void_lt = 0
    n_cluster_lt = 0
    n_field_lt = 0
    
    for g in lt_data['galaxies']:
        env = g.get('environment', 'field')
        vrot = g.get('v_rot', 0)
        if env == 'void':
            n_void_lt += 1
            print(f"  {g['name']:<20} V_rot = {vrot:6.1f} km/s   [void]")
        elif env == 'cluster':
            n_cluster_lt += 1
        else:
            n_field_lt += 1
    
    print(f'\nSummary: {len(lt_data["galaxies"])} total | {n_void_lt} void | {n_cluster_lt} cluster | {n_field_lt} field')

# Check Local Group
lg_path = os.path.join(DATA_DIR, 'local_group_dwarfs.json')
if os.path.exists(lg_path):
    with open(lg_path, 'r') as f:
        lg_data = json.load(f)
    
    print('\n\nLOCAL GROUP (Original Data):')
    print('-'*70)
    
    columns = lg_data.get('columns', [])
    rows = lg_data.get('data', [])
    
    name_idx = columns.index('Name') if 'Name' in columns else 0
    sigma_idx = columns.index('sigma_v_km_s') if 'sigma_v_km_s' in columns else 4
    env_idx = columns.index('Environment') if 'Environment' in columns else -1
    
    n_void_lg = 0
    n_cluster_lg = 0
    
    for row in rows:
        name = row[name_idx]
        sigma_v = row[sigma_idx]
        env = row[env_idx] if env_idx >= 0 else 'unknown'
        
        if env == 'void':
            n_void_lg += 1
            print(f"  {name:<20} sigma_v = {sigma_v:6.1f} km/s   [void]")
        elif env == 'cluster':
            n_cluster_lg += 1
    
    print(f'\nSummary: {len(rows)} total | {n_void_lg} void | {n_cluster_lg} cluster')

print('\n' + '='*70)
print('FINAL USABLE DATA COUNT (only galaxies with actual measurements)')
print('='*70)
print(f'  Void dwarfs with V_rot:    {n_with_vrot}')
print(f'  Cluster dwarfs with V_rot: {n_with_vrot_c}')
print(f'  Matched sample size:       {min(n_with_vrot, n_with_vrot_c)}')
print('='*70)
