#!/usr/bin/env python3
"""
Compile and Analyze Expanded Dwarf Galaxy Dataset for SDCG
==========================================================

This script:
1. Parses all successfully downloaded data files
2. Classifies galaxies by environment (void/cluster/field)
3. Filters for dwarf galaxies (Vflat < 120 km/s typically)
4. Compiles into unified dataset for SDCG analysis
5. Performs preliminary statistical analysis

Data Sources Used:
- ALFALFA α.40: 15,856 HI sources → filter to dwarfs in voids/clusters
- SPARC: 175 galaxies with rotation curves → filter dwarfs
- VGS: 60 void galaxies with environment classification
- Manual catalogs: 23 void + 21 cluster verified dwarfs
"""

import os
import json
import numpy as np
import math

# Project paths - go up 3 levels: _data_processing -> scripts -> project_root
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DWARFS_DIR = os.path.join(DATA_DIR, "dwarfs")

# ============================================================================
# ENVIRONMENT CLASSIFICATION
# ============================================================================

# Known galaxy clusters
CLUSTERS = {
    'Virgo':   {'ra': 187.70, 'dec': 12.39, 'dist': 16.5, 'r_deg': 8.0},
    'Fornax':  {'ra': 54.62, 'dec': -35.45, 'dist': 19.0, 'r_deg': 4.0},
    'Coma':    {'ra': 194.95, 'dec': 27.98, 'dist': 100.0, 'r_deg': 2.5},
    'Perseus': {'ra': 49.95, 'dec': 41.51, 'dist': 72.0, 'r_deg': 2.0},
    'Centaurus': {'ra': 192.2, 'dec': -41.31, 'dist': 52.0, 'r_deg': 2.0},
}

# Known cosmic voids (approximate centers)
VOIDS = {
    'Local_Void':    {'ra': 295, 'dec': 5, 'd_min': 1, 'd_max': 25, 'r_deg': 40},
    'Lynx_Cancer':   {'ra': 130, 'dec': 40, 'd_min': 10, 'd_max': 35, 'r_deg': 25},
    'Eridanus':      {'ra': 55, 'dec': -20, 'd_min': 10, 'd_max': 30, 'r_deg': 20},
    'CVn_Void':      {'ra': 190, 'dec': 35, 'd_min': 3, 'd_max': 15, 'r_deg': 15},
    'Bootes':        {'ra': 218, 'dec': 46, 'd_min': 200, 'd_max': 350, 'r_deg': 10},
}


def angular_sep(ra1, dec1, ra2, dec2):
    """Angular separation in degrees (small angle approximation)"""
    dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) / 2))
    ddec = dec1 - dec2
    return math.sqrt(dra**2 + ddec**2)


def classify_environment(ra, dec, dist_mpc):
    """Classify galaxy environment"""
    if dist_mpc is None or math.isnan(dist_mpc):
        return 'field'
    
    # Check clusters
    for name, cl in CLUSTERS.items():
        sep = angular_sep(ra, dec, cl['ra'], cl['dec'])
        if sep < cl['r_deg'] and abs(dist_mpc - cl['dist']) < 15:
            return 'cluster'
    
    # Check voids
    for name, void in VOIDS.items():
        sep = angular_sep(ra, dec, void['ra'], void['dec'])
        if sep < void['r_deg'] and void['d_min'] < dist_mpc < void['d_max']:
            return 'void'
    
    return 'field'


# ============================================================================
# DATA PARSERS
# ============================================================================

def parse_sparc_mrt():
    """Parse SPARC MRT format file"""
    filepath = os.path.join(DATA_DIR, 'sparc', 'sparc_data.mrt')
    
    if not os.path.exists(filepath):
        print("  SPARC file not found")
        return []
    
    galaxies = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find data section (after header)
    in_data = False
    for line in lines:
        # Skip header until we find data lines (start with galaxy name)
        if line.strip() and not line.startswith(' ') and not line.startswith('Title') and \
           not line.startswith('Authors') and not line.startswith('Table') and \
           not line.startswith('=') and not line.startswith('Byte') and \
           not line.startswith('-') and not line.startswith('Note'):
            in_data = True
        
        if not in_data:
            continue
        
        if len(line) < 90:
            continue
        
        try:
            # Parse fixed-width format based on header description
            name = line[0:11].strip()
            hubble_type = int(line[11:13].strip()) if line[11:13].strip() else 10
            dist = float(line[13:19].strip()) if line[13:19].strip() else None
            inc = float(line[26:30].strip()) if line[26:30].strip() else 45.0
            vflat = float(line[86:91].strip()) if line[86:91].strip() and line[86:91].strip() != '0.0' else None
            
            if not name or name.startswith('Note') or name.startswith('---'):
                continue
            
            # Skip if no rotation velocity
            if vflat is None or vflat == 0:
                continue
            
            # Estimate RA/Dec from galaxy name patterns (approximate)
            # For now, classify based on distance and Hubble type
            # Dwarf types: Im (10), BCD (11), Sm (9), Sdm (8)
            is_dwarf = hubble_type >= 8 or vflat < 100
            
            # Use generic position for classification (SPARC doesn't include coords in this file)
            # We'll mark as 'field' and filter by velocity
            
            galaxies.append({
                'name': name,
                'distance_mpc': dist,
                'v_rot': vflat,
                'hubble_type': hubble_type,
                'inclination': inc,
                'is_dwarf': is_dwarf,
                'environment': 'field',  # Default, will update if we can identify
                'source': 'SPARC'
            })
            
        except (ValueError, IndexError) as e:
            continue
    
    # Filter for dwarfs
    dwarfs = [g for g in galaxies if g['is_dwarf']]
    print(f"  SPARC: {len(galaxies)} total, {len(dwarfs)} dwarfs (Vflat < 100 km/s or type >= Sdm)")
    
    return dwarfs


def parse_alfalfa():
    """Parse ALFALFA CSV - filter for dwarfs"""
    filepath = os.path.join(DATA_DIR, 'alfalfa', 'alfalfa_a40.csv')
    
    if not os.path.exists(filepath):
        print("  ALFALFA file not found")
        return []
    
    galaxies = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find header
    header = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('AGCNr'):
            header = line.strip().split(',')
            data_start = i + 1
            break
    
    if not header:
        print("  Could not find ALFALFA header")
        return []
    
    col_idx = {col: i for i, col in enumerate(header)}
    
    void_count = 0
    cluster_count = 0
    
    for line in lines[data_start:]:
        try:
            parts = line.strip().split(',')
            if len(parts) < 10:
                continue
            
            agc = parts[col_idx.get('AGCNr', 0)]
            ra = float(parts[col_idx.get('RAdeg_HI', 2)])
            dec = float(parts[col_idx.get('Decdeg_HI', 3)])
            v_helio = float(parts[col_idx.get('Vhelio', 6)]) if parts[col_idx.get('Vhelio', 6)] else None
            w50 = float(parts[col_idx.get('W50', 7)]) if parts[col_idx.get('W50', 7)] else None
            dist_str = parts[col_idx.get('Dist', 13)]
            dist = float(dist_str) if dist_str and dist_str.strip('"') else None
            
            # Filter for valid entries
            if v_helio and v_helio < 0:
                continue
            if w50 is None:
                continue
            
            # Filter for dwarfs (W50 < 150 km/s is typical for dwarfs)
            if w50 > 150:
                continue
            
            # Skip very distant galaxies
            if dist and dist > 100:
                continue
            
            # Convert W50 to V_rot (approximate)
            # V_rot ≈ W50 / (2 * sin(i)), assuming average i ≈ 57°
            v_rot = w50 / (2 * math.sin(math.radians(57)))
            
            # Classify environment
            env = classify_environment(ra, dec, dist)
            
            if env == 'void':
                void_count += 1
            elif env == 'cluster':
                cluster_count += 1
            
            galaxies.append({
                'name': f"AGC{agc}",
                'ra': ra,
                'dec': dec,
                'distance_mpc': dist,
                'v_helio': v_helio,
                'w50': w50,
                'v_rot': v_rot,
                'environment': env,
                'source': 'ALFALFA'
            })
            
        except (ValueError, IndexError):
            continue
    
    print(f"  ALFALFA: {len(galaxies)} dwarf candidates")
    print(f"    - Void: {void_count}")
    print(f"    - Cluster: {cluster_count}")
    print(f"    - Field: {len(galaxies) - void_count - cluster_count}")
    
    return galaxies


def load_manual_catalogs():
    """Load manually curated void and cluster dwarf catalogs"""
    galaxies = []
    
    # Void dwarfs
    void_file = os.path.join(DWARFS_DIR, 'verified_void_dwarfs.json')
    if os.path.exists(void_file):
        with open(void_file, 'r') as f:
            data = json.load(f)
        for g in data.get('galaxies', []):
            galaxies.append({
                'name': g['name'],
                'ra': g.get('ra'),
                'dec': g.get('dec'),
                'z': g.get('z'),
                'delta': g.get('delta'),
                'distance_mpc': g.get('z', 0) * 3000 / 70 if g.get('z') else None,  # H0=70
                'environment': 'void',
                'source': g.get('source', 'manual_void')
            })
        print(f"  Manual void dwarfs: {len(data.get('galaxies', []))}")
    
    # Cluster dwarfs
    cluster_file = os.path.join(DWARFS_DIR, 'verified_cluster_dwarfs.json')
    if os.path.exists(cluster_file):
        with open(cluster_file, 'r') as f:
            data = json.load(f)
        for g in data.get('galaxies', []):
            galaxies.append({
                'name': g['name'],
                'ra': g.get('ra'),
                'dec': g.get('dec'),
                'distance_mpc': g.get('dist'),
                'v_rot': g.get('v_rot'),
                'cluster': g.get('cluster'),
                'environment': 'cluster',
                'source': g.get('source', 'manual_cluster')
            })
        print(f"  Manual cluster dwarfs: {len(data.get('galaxies', []))}")
    
    return galaxies


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*70)
    print("    EXPANDED DWARF GALAXY DATASET COMPILATION")
    print("    For SDCG Void vs Cluster Analysis")
    print("="*70)
    
    all_galaxies = []
    
    # Parse each data source
    print("\n[1] Parsing SPARC database...")
    sparc = parse_sparc_mrt()
    all_galaxies.extend(sparc)
    
    print("\n[2] Parsing ALFALFA catalog...")
    alfalfa = parse_alfalfa()
    all_galaxies.extend(alfalfa)
    
    print("\n[3] Loading manual catalogs...")
    manual = load_manual_catalogs()
    all_galaxies.extend(manual)
    
    # Deduplicate by name (simple approach)
    seen_names = set()
    unique = []
    for g in all_galaxies:
        name = g['name'].upper().replace(' ', '')
        if name not in seen_names:
            seen_names.add(name)
            unique.append(g)
    
    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal unique galaxies: {len(unique)}")
    
    # Count by environment
    void = [g for g in unique if g['environment'] == 'void']
    cluster = [g for g in unique if g['environment'] == 'cluster']
    field = [g for g in unique if g['environment'] == 'field']
    
    print(f"\nEnvironment breakdown:")
    print(f"  - Void:    {len(void):5d}")
    print(f"  - Cluster: {len(cluster):5d}")
    print(f"  - Field:   {len(field):5d}")
    
    # Count with rotation velocities
    with_vrot = [g for g in unique if g.get('v_rot')]
    print(f"\nWith rotation velocities: {len(with_vrot)}")
    
    # Subset for SDCG analysis (void vs cluster with V_rot)
    void_with_vrot = [g for g in void if g.get('v_rot')]
    cluster_with_vrot = [g for g in cluster if g.get('v_rot')]
    
    print(f"\nFor SDCG void vs cluster test:")
    print(f"  - Void dwarfs with V_rot:    {len(void_with_vrot)}")
    print(f"  - Cluster dwarfs with V_rot: {len(cluster_with_vrot)}")
    
    # Statistical summary
    if void_with_vrot and cluster_with_vrot:
        void_vrots = [g['v_rot'] for g in void_with_vrot]
        cluster_vrots = [g['v_rot'] for g in cluster_with_vrot]
        
        void_mean = np.mean(void_vrots)
        void_std = np.std(void_vrots) / np.sqrt(len(void_vrots))
        cluster_mean = np.mean(cluster_vrots)
        cluster_std = np.std(cluster_vrots) / np.sqrt(len(cluster_vrots))
        
        delta_v = void_mean - cluster_mean
        delta_err = np.sqrt(void_std**2 + cluster_std**2)
        significance = abs(delta_v) / delta_err if delta_err > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"PRELIMINARY STATISTICAL ANALYSIS")
        print(f"{'='*70}")
        print(f"\nMean rotation velocities:")
        print(f"  Void:    {void_mean:.1f} ± {void_std:.1f} km/s (N={len(void_vrots)})")
        print(f"  Cluster: {cluster_mean:.1f} ± {cluster_std:.1f} km/s (N={len(cluster_vrots)})")
        print(f"\nDifference (void - cluster):")
        print(f"  Δv = {delta_v:+.1f} ± {delta_err:.1f} km/s")
        print(f"  Significance: {significance:.1f}σ")
        
        print(f"\nSDCG Prediction: +12 ± 3 km/s")
        print(f"Observation:     {delta_v:+.1f} ± {delta_err:.1f} km/s")
        
        if delta_v > 0:
            print(f"\n✓ Sign is CORRECT (void dwarfs rotate faster)")
        else:
            print(f"\n✗ Sign is unexpected (cluster dwarfs rotate faster)")
    
    # Save compiled dataset
    output = {
        'description': 'Expanded dwarf galaxy dataset for SDCG analysis',
        'date': '2026-02-03',
        'total': len(unique),
        'void_count': len(void),
        'cluster_count': len(cluster),
        'field_count': len(field),
        'with_vrot': len(with_vrot),
        'sources': {
            'SPARC': len(sparc),
            'ALFALFA': len(alfalfa),
            'manual': len(manual)
        },
        'galaxies': unique
    }
    
    output_file = os.path.join(DATA_DIR, 'expanded_dwarf_dataset.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"Dataset saved to: {output_file}")
    print(f"{'='*70}")
    
    # Create analysis subset
    analysis_subset = {
        'void': void_with_vrot,
        'cluster': cluster_with_vrot,
        'summary': {
            'void_mean': float(np.mean([g['v_rot'] for g in void_with_vrot])) if void_with_vrot else None,
            'cluster_mean': float(np.mean([g['v_rot'] for g in cluster_with_vrot])) if cluster_with_vrot else None,
        }
    }
    
    subset_file = os.path.join(DATA_DIR, 'sdcg_analysis_subset.json')
    with open(subset_file, 'w') as f:
        json.dump(analysis_subset, f, indent=2, default=str)
    
    print(f"Analysis subset saved to: {subset_file}")
    
    return unique


if __name__ == '__main__':
    main()
