#!/usr/bin/env python3
"""
Comprehensive Dwarf Galaxy Dataset Expansion Script
====================================================

This script downloads and compiles dwarf galaxy rotation curve data from 
multiple verified astronomical databases to expand beyond the current 72 galaxies.

Target: 300+ galaxies with environment classification (void/cluster/field)

Data Sources with Working Download Links:
-----------------------------------------
1. SPARC (175 galaxies) - astroweb.cwru.edu/SPARC/
2. LITTLE THINGS (41 dwarfs) - VizieR J/AJ/144/134
3. ALFALFA α.100 (31,000+ HI sources) - egg.astro.cornell.edu
4. VGS - Void Galaxy Survey (60 galaxies) - VizieR J/AJ/144/16
5. Local Volume Galaxy catalog (869 galaxies) - VizieR J/AJ/145/101
6. NGVS - Virgo Cluster (64 dwarfs) - VizieR J/A+A/667/A76
7. Fornax Deep Survey (FDS) - Multiple papers
8. Updated Nearby Galaxy Catalog - VizieR

Author: SDCG Analysis Pipeline
Date: February 2026
"""

import os
import sys
import json
import requests
import numpy as np
from io import StringIO, BytesIO
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DWARFS_DIR = os.path.join(DATA_DIR, "dwarfs")
os.makedirs(DWARFS_DIR, exist_ok=True)

# ============================================================================
# VERIFIED DATA SOURCE URLS
# ============================================================================

DATA_SOURCES = {
    # SPARC - Spitzer Photometry and Accurate Rotation Curves
    'SPARC': {
        'url': 'http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt',
        'url_alt': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/152/157/sparc&-out.max=unlimited',
        'reference': 'Lelli, McGaugh & Schombert (2016) AJ 152, 157',
        'n_galaxies': 175,
        'description': 'High-quality rotation curves with 3.6μm photometry'
    },
    
    # LITTLE THINGS - VLA HI survey of dwarf irregulars
    'LITTLE_THINGS': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/144/134/galaxies&-out.max=unlimited',
        'url_tsv': 'https://vizier.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/AJ/144/134/galaxies',
        'reference': 'Hunter et al. (2012) AJ 144, 134',
        'n_galaxies': 41,
        'description': 'Deep VLA B/C/D array HI maps of nearby dwarfs'
    },
    
    # ALFALFA - Arecibo Legacy Fast ALFA Survey
    'ALFALFA_100': {
        'url': 'https://egg.astro.cornell.edu/alfalfa/data/a100files/a100.code1.230517.csv',
        'url_alt': 'https://egg.astro.cornell.edu/alfalfa/data/a40files/a40.datafile1.csv',
        'reference': 'Haynes et al. (2018) ApJ 861, 49',
        'n_galaxies': 31502,
        'description': '100% ALFALFA extragalactic HI source catalog'
    },
    
    # VGS - Void Galaxy Survey (Kreckel+2012)
    'VGS': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/144/16/vgs&-out.max=unlimited',
        'url_tsv': 'https://vizier.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/AJ/144/16/vgs',
        'hi_url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/144/16/table4&-out.max=unlimited',
        'reference': 'Kreckel et al. (2012) AJ 144, 16',
        'n_galaxies': 60,
        'description': 'Void galaxies with optical and HI properties'
    },
    
    # Local Volume Galaxy catalog
    'LVG': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/145/101/lvg&-out.max=unlimited',
        'url_tsv': 'https://vizier.cds.unistra.fr/viz-bin/nph-Cat/tsv?J/AJ/145/101/lvg',
        'reference': 'Karachentsev et al. (2013) AJ 145, 101',
        'n_galaxies': 869,
        'description': 'Updated Nearby Galaxy Catalog (D < 11 Mpc)'
    },
    
    # NGVS - Virgo Cluster dwarfs
    'NGVS_VIRGO': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/A+A/667/A76/tabled1&-out.max=unlimited',
        'reference': 'Junais et al. (2022) A&A 667, A76',
        'n_galaxies': 64,
        'description': 'Virgo cluster dwarf galaxies (NGVS)'
    },
    
    # THINGS - The HI Nearby Galaxy Survey
    'THINGS': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/AJ/136/2563/table1&-out.max=unlimited',
        'reference': 'Walter et al. (2008) AJ 136, 2563',
        'n_galaxies': 34,
        'description': 'High-resolution HI maps of nearby galaxies'
    },
    
    # VCC - Virgo Cluster Catalog
    'VCC': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=VII/73/vcc&-out.max=unlimited',
        'reference': 'Binggeli et al. (1985) AJ 90, 1681',
        'n_galaxies': 2096,
        'description': 'Complete Virgo Cluster Catalog'
    },
    
    # SDSS Void Galaxies (Pan+2012)
    'SDSS_VOIDS': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/MNRAS/421/926&-out.max=unlimited',
        'reference': 'Pan et al. (2012) MNRAS 421, 926',
        'n_galaxies': 'varies',
        'description': 'SDSS void galaxy catalog'
    },
    
    # Fornax Cluster Catalog  
    'FCC': {
        'url': 'https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/A+AS/121/507&-out.max=unlimited',
        'reference': 'Ferguson (1989) AJ 98, 367',
        'n_galaxies': 340,
        'description': 'Fornax Cluster Catalog'
    },
}

# ============================================================================
# ENVIRONMENT CLASSIFICATION
# ============================================================================

def classify_environment(ra, dec, distance_mpc, density_delta=None):
    """
    Classify galaxy environment as void, cluster, or field.
    
    Methods:
    1. If density contrast (delta) is provided, use it directly
    2. Otherwise, use proximity to known structures
    
    Thresholds:
    - Void: delta < -0.5 (underdense by >50%)
    - Cluster: delta > 100 or within cluster virial radius
    - Field: everything else
    """
    if density_delta is not None:
        if density_delta < -0.5:
            return 'void'
        elif density_delta > 100:
            return 'cluster'
        else:
            return 'field'
    
    # Known cluster centers (approximate)
    clusters = {
        'Virgo': {'ra': 187.7, 'dec': 12.4, 'dist': 16.5, 'r_vir': 2.2},
        'Fornax': {'ra': 54.6, 'dec': -35.5, 'dist': 19.0, 'r_vir': 0.7},
        'Coma': {'ra': 194.9, 'dec': 27.9, 'dist': 100.0, 'r_vir': 2.9},
        'Perseus': {'ra': 49.9, 'dec': 41.5, 'dist': 72.0, 'r_vir': 2.0},
    }
    
    # Known voids
    voids = {
        'Local_Void': {'ra': 295.0, 'dec': 5.0, 'dist_range': (1, 25)},
        'Lynx_Cancer': {'ra': 130.0, 'dec': 40.0, 'dist_range': (10, 30)},
        'Eridanus': {'ra': 55.0, 'dec': -35.0, 'dist_range': (10, 25)},
        'CVn_Void': {'ra': 190.0, 'dec': 35.0, 'dist_range': (3, 15)},
    }
    
    # Check cluster membership
    for name, cl in clusters.items():
        ang_sep = np.sqrt((ra - cl['ra'])**2 + (dec - cl['dec'])**2)
        if distance_mpc and cl['dist']:
            # Physical separation in Mpc
            phys_sep = (ang_sep * np.pi / 180.0) * cl['dist']
            if abs(distance_mpc - cl['dist']) < 10 and phys_sep < cl['r_vir']:
                return 'cluster'
    
    # Check void membership (simplified)
    for name, void in voids.items():
        ang_sep = np.sqrt((ra - void['ra'])**2 + (dec - void['dec'])**2)
        if distance_mpc:
            d_min, d_max = void['dist_range']
            if d_min < distance_mpc < d_max and ang_sep < 30:
                return 'void'
    
    return 'field'


# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================

def download_vizier_table(source_key, save_path):
    """Download data from VizieR in VOTable or TSV format"""
    source = DATA_SOURCES.get(source_key)
    if not source:
        print(f"  ✗ Unknown source: {source_key}")
        return None
    
    # Try primary URL
    try:
        print(f"  Trying VizieR: {source['url'][:60]}...")
        response = requests.get(source['url'], timeout=30)
        if response.status_code == 200 and len(response.content) > 100:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded {source_key} ({len(response.content)} bytes)")
            return save_path
    except Exception as e:
        print(f"  Warning: {e}")
    
    # Try alternate URL
    if 'url_alt' in source:
        try:
            print(f"  Trying alternate: {source['url_alt'][:60]}...")
            response = requests.get(source['url_alt'], timeout=30)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {source_key} from alternate")
                return save_path
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Try TSV format
    if 'url_tsv' in source:
        try:
            print(f"  Trying TSV format...")
            response = requests.get(source['url_tsv'], timeout=30)
            if response.status_code == 200:
                save_path_tsv = save_path.replace('.xml', '.tsv')
                with open(save_path_tsv, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {source_key} as TSV")
                return save_path_tsv
        except Exception as e:
            print(f"  Warning: {e}")
    
    print(f"  ✗ Could not download {source_key}")
    return None


def download_sparc():
    """Download SPARC database"""
    print("\n[1] Downloading SPARC (Lelli+2016)...")
    
    # Direct download from CWRU
    urls = [
        'http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt',
        'http://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.txt',
    ]
    
    sparc_dir = os.path.join(DATA_DIR, 'sparc')
    os.makedirs(sparc_dir, exist_ok=True)
    
    # Try MRT format
    for url in urls:
        try:
            print(f"  Trying: {url[:50]}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                filename = url.split('/')[-1]
                filepath = os.path.join(sparc_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {filename}")
                return filepath
        except Exception as e:
            print(f"  Warning: {e}")
    
    # Try VizieR mirror
    vizier_url = 'https://vizier.cds.unistra.fr/viz-bin/nph-Cat/txt.gz?J/AJ/152/157'
    try:
        print(f"  Trying VizieR mirror...")
        response = requests.get(vizier_url, timeout=30)
        if response.status_code == 200:
            filepath = os.path.join(sparc_dir, 'sparc_vizier.txt.gz')
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded SPARC from VizieR")
            return filepath
    except Exception as e:
        print(f"  Warning: {e}")
    
    print("  ✗ Could not download SPARC - trying manual construction...")
    return None


def download_alfalfa():
    """Download ALFALFA α.100 catalog"""
    print("\n[2] Downloading ALFALFA (Haynes+2018)...")
    
    alfalfa_dir = os.path.join(DATA_DIR, 'alfalfa')
    os.makedirs(alfalfa_dir, exist_ok=True)
    
    # α.100 catalog (most complete)
    urls = [
        ('https://egg.astro.cornell.edu/alfalfa/data/a100files/a100.code1.230517.csv', 'alfalfa_a100.csv'),
        ('https://egg.astro.cornell.edu/alfalfa/data/a40files/a40.datafile1.csv', 'alfalfa_a40.csv'),
        ('https://egg.astro.cornell.edu/alfalfa/data/gridsafiles/alfalfa.gridsa_160317.csv', 'alfalfa_gridsa.csv'),
    ]
    
    for url, filename in urls:
        try:
            print(f"  Trying: {filename}...")
            response = requests.get(url, timeout=60)
            if response.status_code == 200 and len(response.content) > 1000:
                filepath = os.path.join(alfalfa_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {filename} ({len(response.content)//1024} KB)")
                return filepath
        except Exception as e:
            print(f"  Warning: {e}")
    
    print("  ✗ Could not download ALFALFA")
    return None


def download_little_things():
    """Download LITTLE THINGS survey data"""
    print("\n[3] Downloading LITTLE THINGS (Hunter+2012)...")
    
    save_path = os.path.join(DWARFS_DIR, 'little_things.xml')
    result = download_vizier_table('LITTLE_THINGS', save_path)
    
    if not result:
        # Try direct TSV
        try:
            url = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/144/134/galaxies&-out.max=100'
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                save_path_tsv = os.path.join(DWARFS_DIR, 'little_things.tsv')
                with open(save_path_tsv, 'w') as f:
                    f.write(response.text)
                print(f"  ✓ Downloaded LITTLE THINGS as TSV")
                return save_path_tsv
        except Exception as e:
            print(f"  Warning: {e}")
    
    return result


def download_vgs():
    """Download Void Galaxy Survey data"""
    print("\n[4] Downloading VGS (Kreckel+2012)...")
    
    vgs_dir = os.path.join(DATA_DIR, 'vgs')
    os.makedirs(vgs_dir, exist_ok=True)
    
    # Optical properties
    save_path = os.path.join(vgs_dir, 'vgs_optical.xml')
    result1 = download_vizier_table('VGS', save_path)
    
    # HI properties
    try:
        url_hi = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/144/16/table4&-out.max=100'
        response = requests.get(url_hi, timeout=30)
        if response.status_code == 200:
            hi_path = os.path.join(vgs_dir, 'vgs_hi.tsv')
            with open(hi_path, 'w') as f:
                f.write(response.text)
            print(f"  ✓ Downloaded VGS HI data")
    except Exception as e:
        print(f"  Warning: {e}")
    
    return result1


def download_local_volume():
    """Download Local Volume Galaxy catalog"""
    print("\n[5] Downloading LVG (Karachentsev+2013)...")
    
    save_path = os.path.join(DWARFS_DIR, 'local_volume.xml')
    return download_vizier_table('LVG', save_path)


def download_virgo_dwarfs():
    """Download Virgo cluster dwarf data from multiple sources"""
    print("\n[6] Downloading Virgo Cluster dwarfs...")
    
    virgo_dir = os.path.join(DATA_DIR, 'virgo')
    os.makedirs(virgo_dir, exist_ok=True)
    
    # NGVS
    save_path = os.path.join(virgo_dir, 'ngvs_dwarfs.xml')
    result = download_vizier_table('NGVS_VIRGO', save_path)
    
    # VCC
    try:
        url_vcc = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=VII/73/vcc&-out.max=500&Vt=%3C15'
        response = requests.get(url_vcc, timeout=30)
        if response.status_code == 200:
            vcc_path = os.path.join(virgo_dir, 'vcc_bright.tsv')
            with open(vcc_path, 'w') as f:
                f.write(response.text)
            print(f"  ✓ Downloaded VCC bright galaxies")
    except Exception as e:
        print(f"  Warning: {e}")
    
    return result


def download_fornax_dwarfs():
    """Download Fornax cluster dwarf data"""
    print("\n[7] Downloading Fornax Cluster dwarfs...")
    
    fornax_dir = os.path.join(DATA_DIR, 'fornax')
    os.makedirs(fornax_dir, exist_ok=True)
    
    # FDS (Eigenthaler+2018, Venhola+2018)
    try:
        # Fornax Dwarf catalog
        url = 'https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A+A/620/A165/tableb1&-out.max=500'
        response = requests.get(url, timeout=30)
        if response.status_code == 200 and len(response.content) > 100:
            filepath = os.path.join(fornax_dir, 'fds_dwarfs.tsv')
            with open(filepath, 'w') as f:
                f.write(response.text)
            print(f"  ✓ Downloaded FDS dwarfs")
            return filepath
    except Exception as e:
        print(f"  Warning: {e}")
    
    # Try FCC
    save_path = os.path.join(fornax_dir, 'fcc.xml')
    return download_vizier_table('FCC', save_path)


def download_things():
    """Download THINGS survey data"""
    print("\n[8] Downloading THINGS (Walter+2008)...")
    
    save_path = os.path.join(DWARFS_DIR, 'things.xml')
    return download_vizier_table('THINGS', save_path)


# ============================================================================
# COMPILE MASTER DATASET
# ============================================================================

def load_existing_data():
    """Load existing 72-galaxy dataset"""
    existing = {'void': [], 'cluster': [], 'field': []}
    
    # Load void dwarfs
    void_file = os.path.join(DWARFS_DIR, 'void_dwarfs.json')
    if os.path.exists(void_file):
        with open(void_file, 'r') as f:
            data = json.load(f)
            if 'data' in data:
                for row in data['data']:
                    existing['void'].append({
                        'name': row[0],
                        'ra': row[1],
                        'dec': row[2],
                        'distance_mpc': row[3],
                        'v_rot': row[5] if len(row) > 5 else None,
                        'source': 'existing_void'
                    })
    
    # Load local group dwarfs
    lg_file = os.path.join(DWARFS_DIR, 'local_group_dwarfs.json')
    if os.path.exists(lg_file):
        with open(lg_file, 'r') as f:
            data = json.load(f)
            if 'data' in data:
                for row in data['data']:
                    existing['field'].append({
                        'name': row[0],
                        'ra': row[1],
                        'dec': row[2],
                        'distance_mpc': row[3] / 1000.0 if row[3] > 100 else row[3],  # kpc to Mpc
                        'source': 'existing_local_group'
                    })
    
    total = sum(len(v) for v in existing.values())
    print(f"\nLoaded {total} existing galaxies")
    return existing


def generate_summary_report(all_galaxies):
    """Generate summary of expanded dataset"""
    
    void_count = len([g for g in all_galaxies if g.get('environment') == 'void'])
    cluster_count = len([g for g in all_galaxies if g.get('environment') == 'cluster'])
    field_count = len([g for g in all_galaxies if g.get('environment') == 'field'])
    
    # Count by source
    sources = {}
    for g in all_galaxies:
        src = g.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    
    report = f"""
================================================================================
                    EXPANDED DWARF GALAXY DATASET SUMMARY
================================================================================

TOTAL GALAXIES: {len(all_galaxies)}

ENVIRONMENT BREAKDOWN:
  - Void galaxies:    {void_count:4d}
  - Cluster galaxies: {cluster_count:4d}
  - Field galaxies:   {field_count:4d}

SOURCES:
"""
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        report += f"  - {src}: {count}\n"
    
    report += """
================================================================================
DATA SOURCES ATTEMPTED:
"""
    for key, info in DATA_SOURCES.items():
        report += f"\n{key}:\n"
        report += f"  Reference: {info['reference']}\n"
        report += f"  Expected: ~{info['n_galaxies']} galaxies\n"
        report += f"  URL: {info['url'][:70]}...\n"
    
    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("    DWARF GALAXY DATASET EXPANSION")
    print("    Target: 300+ galaxies with environment classification")
    print("="*70)
    
    # Track downloads
    downloaded = {}
    failed = []
    
    # 1. SPARC
    result = download_sparc()
    if result:
        downloaded['SPARC'] = result
    else:
        failed.append('SPARC')
    
    # 2. ALFALFA
    result = download_alfalfa()
    if result:
        downloaded['ALFALFA'] = result
    else:
        failed.append('ALFALFA')
    
    # 3. LITTLE THINGS
    result = download_little_things()
    if result:
        downloaded['LITTLE_THINGS'] = result
    else:
        failed.append('LITTLE_THINGS')
    
    # 4. VGS
    result = download_vgs()
    if result:
        downloaded['VGS'] = result
    else:
        failed.append('VGS')
    
    # 5. Local Volume
    result = download_local_volume()
    if result:
        downloaded['LVG'] = result
    else:
        failed.append('LVG')
    
    # 6. Virgo dwarfs
    result = download_virgo_dwarfs()
    if result:
        downloaded['VIRGO'] = result
    else:
        failed.append('VIRGO')
    
    # 7. Fornax dwarfs
    result = download_fornax_dwarfs()
    if result:
        downloaded['FORNAX'] = result
    else:
        failed.append('FORNAX')
    
    # 8. THINGS
    result = download_things()
    if result:
        downloaded['THINGS'] = result
    else:
        failed.append('THINGS')
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"\n✓ Successfully downloaded: {len(downloaded)}")
    for key, path in downloaded.items():
        print(f"  - {key}: {path}")
    
    print(f"\n✗ Failed: {len(failed)}")
    for key in failed:
        print(f"  - {key}")
    
    # Load existing data
    existing = load_existing_data()
    
    # Save download manifest
    manifest = {
        'date': '2026-02-03',
        'downloaded': downloaded,
        'failed': failed,
        'existing_count': sum(len(v) for v in existing.values()),
        'sources': {k: v['reference'] for k, v in DATA_SOURCES.items()}
    }
    
    manifest_path = os.path.join(DATA_DIR, 'download_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Parse downloaded files to extract rotation velocities
2. Cross-match with void catalogs for environment classification
3. Apply quality cuts (S/N > 5, inclination 30°-80°)
4. Deduplicate across catalogs
5. Run SDCG analysis on expanded dataset

For manual downloads, visit:
- SPARC: http://astroweb.cwru.edu/SPARC/
- ALFALFA: https://egg.astro.cornell.edu/alfalfa/data/
- VizieR: https://vizier.cds.unistra.fr/
""")
    
    return downloaded, failed


if __name__ == '__main__':
    main()
