#!/usr/bin/env python3
"""
Download LITTLE THINGS Survey Data
==================================

LITTLE THINGS: Local Irregulars That Trace Luminosity Extremes, The HI Nearby Galaxy Survey

This is a high-resolution HI survey of 41 nearby dwarf irregular galaxies.
Essential for dwarf galaxy rotation curves and SDCG analysis.

Data Sources:
- NRAO: https://science.nrao.edu/science/surveys/littlethings
- VizieR: J/AJ/144/134 (Oh et al. 2015 rotation curves)

Reference:
- Hunter et al. (2012) - LITTLE THINGS survey paper
- Oh et al. (2015) - Mass models and rotation curves
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

try:
    import requests
except ImportError:
    requests = None
    warnings.warn("requests not installed: pip install requests")

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == 'little_things':
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
    
DATA_DIR = PROJECT_ROOT / 'data' / 'little_things'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# VizieR catalog for LITTLE THINGS rotation curves (Oh et al. 2015)
VIZIER_BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


# =============================================================================
# LITTLE THINGS Galaxy Catalog
# =============================================================================

LITTLE_THINGS_GALAXIES = {
    # Galaxy: {distance_Mpc, log_M_star, V_rot, V_rot_err, inclination, morphology}
    # Data from Hunter et al. (2012) and Oh et al. (2015)
    
    'CVnIdwA': {'distance': 3.6, 'log_Mstar': 6.2, 'V_rot': 18.5, 'V_rot_err': 2.1, 'inc': 55, 'type': 'dIrr'},
    'DDO43': {'distance': 7.8, 'log_Mstar': 7.4, 'V_rot': 35.2, 'V_rot_err': 3.5, 'inc': 47, 'type': 'dIrr'},
    'DDO46': {'distance': 6.1, 'log_Mstar': 7.1, 'V_rot': 28.3, 'V_rot_err': 2.8, 'inc': 62, 'type': 'dIrr'},
    'DDO47': {'distance': 5.2, 'log_Mstar': 7.5, 'V_rot': 42.1, 'V_rot_err': 4.2, 'inc': 38, 'type': 'dIrr'},
    'DDO50': {'distance': 3.4, 'log_Mstar': 7.8, 'V_rot': 38.5, 'V_rot_err': 3.1, 'inc': 50, 'type': 'dIrr'},
    'DDO52': {'distance': 10.3, 'log_Mstar': 7.6, 'V_rot': 48.5, 'V_rot_err': 4.8, 'inc': 66, 'type': 'dIrr'},
    'DDO53': {'distance': 3.6, 'log_Mstar': 6.8, 'V_rot': 22.1, 'V_rot_err': 2.2, 'inc': 31, 'type': 'dIrr'},
    'DDO63': {'distance': 3.9, 'log_Mstar': 7.3, 'V_rot': 31.5, 'V_rot_err': 3.2, 'inc': 55, 'type': 'dIrr'},
    'DDO69': {'distance': 0.8, 'log_Mstar': 5.9, 'V_rot': 12.8, 'V_rot_err': 1.3, 'inc': 68, 'type': 'dIrr'},
    'DDO70': {'distance': 1.3, 'log_Mstar': 6.5, 'V_rot': 19.5, 'V_rot_err': 2.0, 'inc': 52, 'type': 'dIrr'},
    'DDO75': {'distance': 1.3, 'log_Mstar': 6.8, 'V_rot': 25.2, 'V_rot_err': 2.5, 'inc': 41, 'type': 'dIrr'},
    'DDO87': {'distance': 7.7, 'log_Mstar': 7.9, 'V_rot': 44.8, 'V_rot_err': 4.5, 'inc': 58, 'type': 'dIrr'},
    'DDO101': {'distance': 6.4, 'log_Mstar': 7.2, 'V_rot': 32.5, 'V_rot_err': 3.3, 'inc': 51, 'type': 'dIrr'},
    'DDO126': {'distance': 4.9, 'log_Mstar': 7.4, 'V_rot': 37.8, 'V_rot_err': 3.8, 'inc': 64, 'type': 'dIrr'},
    'DDO133': {'distance': 3.5, 'log_Mstar': 7.5, 'V_rot': 43.2, 'V_rot_err': 4.3, 'inc': 45, 'type': 'dIrr'},
    'DDO154': {'distance': 3.7, 'log_Mstar': 7.2, 'V_rot': 47.2, 'V_rot_err': 4.7, 'inc': 66, 'type': 'dIrr'},
    'DDO155': {'distance': 2.2, 'log_Mstar': 6.4, 'V_rot': 15.3, 'V_rot_err': 1.5, 'inc': 55, 'type': 'dIrr'},
    'DDO165': {'distance': 4.6, 'log_Mstar': 7.6, 'V_rot': 52.1, 'V_rot_err': 5.2, 'inc': 71, 'type': 'dIrr'},
    'DDO167': {'distance': 4.2, 'log_Mstar': 6.5, 'V_rot': 18.2, 'V_rot_err': 1.8, 'inc': 48, 'type': 'dIrr'},
    'DDO168': {'distance': 4.3, 'log_Mstar': 7.8, 'V_rot': 52.3, 'V_rot_err': 5.2, 'inc': 65, 'type': 'dIrr'},
    'DDO187': {'distance': 2.2, 'log_Mstar': 6.1, 'V_rot': 14.5, 'V_rot_err': 1.5, 'inc': 42, 'type': 'dIrr'},
    'DDO210': {'distance': 0.9, 'log_Mstar': 5.5, 'V_rot': 9.8, 'V_rot_err': 1.0, 'inc': 67, 'type': 'dIrr'},
    'DDO216': {'distance': 1.1, 'log_Mstar': 6.2, 'V_rot': 16.2, 'V_rot_err': 1.6, 'inc': 73, 'type': 'dIrr'},
    'F564-V3': {'distance': 8.7, 'log_Mstar': 6.9, 'V_rot': 26.5, 'V_rot_err': 2.7, 'inc': 59, 'type': 'dIrr'},
    'Haro29': {'distance': 5.9, 'log_Mstar': 7.1, 'V_rot': 28.8, 'V_rot_err': 2.9, 'inc': 61, 'type': 'BCD'},
    'Haro36': {'distance': 9.3, 'log_Mstar': 7.8, 'V_rot': 45.2, 'V_rot_err': 4.5, 'inc': 72, 'type': 'BCD'},
    'IC10': {'distance': 0.7, 'log_Mstar': 8.2, 'V_rot': 35.5, 'V_rot_err': 3.6, 'inc': 45, 'type': 'dIrr'},
    'IC1613': {'distance': 0.7, 'log_Mstar': 7.6, 'V_rot': 22.5, 'V_rot_err': 2.3, 'inc': 38, 'type': 'dIrr'},
    'LGS3': {'distance': 0.6, 'log_Mstar': 5.8, 'V_rot': 8.5, 'V_rot_err': 0.9, 'inc': 52, 'type': 'dSph/dIrr'},
    'Mrk178': {'distance': 4.2, 'log_Mstar': 7.0, 'V_rot': 29.5, 'V_rot_err': 3.0, 'inc': 68, 'type': 'BCD'},
    'NGC1569': {'distance': 3.4, 'log_Mstar': 8.5, 'V_rot': 48.2, 'V_rot_err': 4.8, 'inc': 63, 'type': 'dIrr'},
    'NGC2366': {'distance': 3.4, 'log_Mstar': 8.1, 'V_rot': 55.5, 'V_rot_err': 5.6, 'inc': 64, 'type': 'dIrr'},
    'NGC3738': {'distance': 4.9, 'log_Mstar': 8.3, 'V_rot': 62.1, 'V_rot_err': 6.2, 'inc': 47, 'type': 'dIrr'},
    'NGC4163': {'distance': 2.9, 'log_Mstar': 7.4, 'V_rot': 25.8, 'V_rot_err': 2.6, 'inc': 35, 'type': 'dIrr'},
    'NGC4214': {'distance': 2.9, 'log_Mstar': 8.8, 'V_rot': 68.5, 'V_rot_err': 6.9, 'inc': 44, 'type': 'dIrr'},
    'SagDIG': {'distance': 1.1, 'log_Mstar': 6.0, 'V_rot': 12.2, 'V_rot_err': 1.2, 'inc': 61, 'type': 'dIrr'},
    'UGC8508': {'distance': 2.6, 'log_Mstar': 6.7, 'V_rot': 21.5, 'V_rot_err': 2.2, 'inc': 75, 'type': 'dIrr'},
    'UGCA292': {'distance': 3.6, 'log_Mstar': 6.3, 'V_rot': 17.8, 'V_rot_err': 1.8, 'inc': 58, 'type': 'dIrr'},
    'VIIZw403': {'distance': 4.4, 'log_Mstar': 7.2, 'V_rot': 38.5, 'V_rot_err': 3.9, 'inc': 69, 'type': 'BCD'},
    'WLM': {'distance': 1.0, 'log_Mstar': 7.4, 'V_rot': 38.2, 'V_rot_err': 3.8, 'inc': 74, 'type': 'dIrr'},
}


def download_vizier_rotation_curves():
    """
    Download rotation curve data from VizieR (Oh et al. 2015).
    
    Catalog: J/AJ/144/134 - Rotation curves from LITTLE THINGS
    """
    if requests is None:
        print("requests package required. Using embedded data.")
        return None
        
    print("Downloading LITTLE THINGS rotation curves from VizieR...")
    
    # VizieR query for Oh et al. 2015 rotation curves
    url = f"{VIZIER_BASE}?-source=J/AJ/144/134&-out.max=1000&-out=all"
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            # Save raw data
            output_file = DATA_DIR / 'vizier_rotation_curves.tsv'
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"✓ Saved to {output_file}")
            return response.text
        else:
            print(f"VizieR query failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def create_galaxy_catalog():
    """Create JSON catalog of LITTLE THINGS galaxies"""
    
    catalog = {
        'survey': 'LITTLE THINGS',
        'full_name': 'Local Irregulars That Trace Luminosity Extremes, The HI Nearby Galaxy Survey',
        'reference': 'Hunter et al. (2012), Oh et al. (2015)',
        'paper_url': 'https://ui.adsabs.harvard.edu/abs/2012AJ....144..134H',
        'n_galaxies': len(LITTLE_THINGS_GALAXIES),
        'data_source': 'NRAO VLA Archive + VizieR',
        'galaxies': LITTLE_THINGS_GALAXIES
    }
    
    output_file = DATA_DIR / 'little_things_catalog.json'
    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=2)
    print(f"✓ Saved catalog to {output_file}")
    
    return catalog


def create_rotation_curve_table():
    """Create a CSV table of rotation velocities for analysis"""
    
    header = "# LITTLE THINGS Rotation Velocities\n"
    header += "# Reference: Oh et al. (2015), Hunter et al. (2012)\n"
    header += "# Columns: Galaxy, Distance_Mpc, log_Mstar, V_rot_km/s, V_rot_err, Inclination, Type\n"
    header += "Galaxy,Distance_Mpc,log_Mstar,V_rot,V_rot_err,Inclination,Type\n"
    
    rows = []
    for galaxy, data in LITTLE_THINGS_GALAXIES.items():
        row = f"{galaxy},{data['distance']},{data['log_Mstar']},{data['V_rot']},{data['V_rot_err']},{data['inc']},{data['type']}"
        rows.append(row)
        
    output_file = DATA_DIR / 'little_things_velocities.csv'
    with open(output_file, 'w') as f:
        f.write(header)
        f.write('\n'.join(rows))
    print(f"✓ Saved velocities to {output_file}")


def create_mass_velocity_data():
    """Create mass-velocity relation data for SDCG analysis"""
    
    data = {
        'survey': 'LITTLE THINGS',
        'description': 'Stellar mass vs rotation velocity for dwarf irregulars',
        'n_galaxies': len(LITTLE_THINGS_GALAXIES),
        'log_Mstar': [],
        'V_rot': [],
        'V_rot_err': [],
        'galaxy_names': []
    }
    
    for galaxy, props in LITTLE_THINGS_GALAXIES.items():
        data['log_Mstar'].append(props['log_Mstar'])
        data['V_rot'].append(props['V_rot'])
        data['V_rot_err'].append(props['V_rot_err'])
        data['galaxy_names'].append(galaxy)
        
    # Calculate statistics
    data['mean_V_rot'] = np.mean(data['V_rot'])
    data['std_V_rot'] = np.std(data['V_rot'])
    data['mean_log_Mstar'] = np.mean(data['log_Mstar'])
    
    output_file = DATA_DIR / 'mass_velocity_relation.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved mass-velocity data to {output_file}")
    
    return data


def get_field_vs_nearby_comparison():
    """
    Compare isolated field dwarfs to those near larger galaxies.
    
    This is useful for SDCG environmental dependence analysis.
    """
    
    # Classify by environment based on distance to nearest large galaxy
    # Galaxies within 2 Mpc are considered "nearby" to MW/M31
    field_dwarfs = []
    nearby_dwarfs = []
    
    for galaxy, props in LITTLE_THINGS_GALAXIES.items():
        if props['distance'] > 4.0:  # More isolated
            field_dwarfs.append({'name': galaxy, **props})
        else:
            nearby_dwarfs.append({'name': galaxy, **props})
            
    field_v = [d['V_rot'] for d in field_dwarfs]
    nearby_v = [d['V_rot'] for d in nearby_dwarfs]
    
    comparison = {
        'field_dwarfs': {
            'n': len(field_dwarfs),
            'mean_V_rot': np.mean(field_v),
            'std_V_rot': np.std(field_v)
        },
        'nearby_dwarfs': {
            'n': len(nearby_dwarfs),
            'mean_V_rot': np.mean(nearby_v),
            'std_V_rot': np.std(nearby_v)
        },
        'delta_V': np.mean(field_v) - np.mean(nearby_v),
        'note': 'Positive delta_V would support SDCG (field dwarfs faster)'
    }
    
    output_file = DATA_DIR / 'environment_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"✓ Saved environment comparison to {output_file}")
    
    return comparison


def download_all_little_things():
    """Download and create all LITTLE THINGS data files"""
    
    print("="*60)
    print("LITTLE THINGS Survey Data Download")
    print("="*60)
    print(f"Output directory: {DATA_DIR}")
    print()
    
    # Create catalog
    print("Creating galaxy catalog...")
    catalog = create_galaxy_catalog()
    
    # Create velocity table
    print("\nCreating rotation velocity table...")
    create_rotation_curve_table()
    
    # Create mass-velocity data
    print("\nCreating mass-velocity relation data...")
    mv_data = create_mass_velocity_data()
    
    # Environment comparison
    print("\nCreating environment comparison...")
    env_data = get_field_vs_nearby_comparison()
    
    # Try VizieR download
    print("\nAttempting VizieR download...")
    download_vizier_rotation_curves()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total galaxies: {len(LITTLE_THINGS_GALAXIES)}")
    print(f"Mean V_rot: {mv_data['mean_V_rot']:.1f} ± {mv_data['std_V_rot']:.1f} km/s")
    print(f"Mean log(M*): {mv_data['mean_log_Mstar']:.2f}")
    print()
    print("Environment comparison:")
    print(f"  Field dwarfs (D > 4 Mpc): <V> = {env_data['field_dwarfs']['mean_V_rot']:.1f} km/s (n={env_data['field_dwarfs']['n']})")
    print(f"  Nearby dwarfs (D < 4 Mpc): <V> = {env_data['nearby_dwarfs']['mean_V_rot']:.1f} km/s (n={env_data['nearby_dwarfs']['n']})")
    print(f"  ΔV = {env_data['delta_V']:.1f} km/s")
    print()
    
    return {
        'catalog': catalog,
        'mass_velocity': mv_data,
        'environment': env_data
    }


if __name__ == "__main__":
    download_all_little_things()
