#!/usr/bin/env python3
"""
Download All Cosmological Simulation Data
=========================================

Master script to download and aggregate data from all major
cosmological simulations for SDCG tidal stripping analysis:

- IllustrisTNG: https://www.tng-project.org/
- EAGLE: http://icc.dur.ac.uk/Eagle/
- FIRE-2: https://fire.northwestern.edu/
- SIMBA: http://simba.roe.ac.uk/

Usage:
    python download_all.py [--tng-key YOUR_API_KEY]
    
Or in Python:
    >>> from simulations.cosmological_simulations import download_all_simulations
    >>> data = download_all_simulations()
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'simulations'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import simulation modules
try:
    from .illustris_tng import IllustrisTNGAccess, download_tng_stripping_data
    from .eagle_database import EAGLEDatabase, download_eagle_stripping_data
    from .fire_data import FIREDataAccess, download_fire_stripping_data
    from .simba_data import SIMBADataAccess, download_simba_stripping_data
except ImportError:
    # Running as script directly
    from illustris_tng import IllustrisTNGAccess, download_tng_stripping_data
    from eagle_database import EAGLEDatabase, download_eagle_stripping_data
    from fire_data import FIREDataAccess, download_fire_stripping_data
    from simba_data import SIMBADataAccess, download_simba_stripping_data


def download_all_simulations(tng_api_key: Optional[str] = None,
                            use_cache: bool = True) -> Dict:
    """
    Download stripping data from all major simulations.
    
    Args:
        tng_api_key: IllustrisTNG API key (get free at tng-project.org)
        use_cache: Whether to use cached data if available
        
    Returns:
        Dict with combined stripping statistics from all simulations
    """
    print("="*70)
    print("DOWNLOADING COSMOLOGICAL SIMULATION DATA FOR SDCG ANALYSIS")
    print("="*70)
    print(f"Output directory: {DATA_DIR}")
    print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'simulations': {}
    }
    
    # 1. IllustrisTNG
    print("\n" + "="*70)
    print("1. IllustrisTNG")
    print("="*70)
    
    try:
        if tng_api_key:
            tng_data = download_tng_stripping_data(api_key=tng_api_key)
            results['simulations']['IllustrisTNG'] = tng_data
        else:
            # Use published values
            print("No TNG API key provided. Using published values.")
            print("Get a FREE API key at: https://www.tng-project.org/data/")
            results['simulations']['IllustrisTNG'] = {
                'delta_v': 8.2,
                'delta_v_err': 1.5,
                'source': 'Published values (Joshi et al. 2021)',
                'note': 'Set TNG_API_KEY environment variable for live data'
            }
    except Exception as e:
        print(f"Warning: TNG access failed: {e}")
        results['simulations']['IllustrisTNG'] = {
            'error': str(e),
            'delta_v': 8.2,  # Fallback to published
            'delta_v_err': 1.5
        }
        
    # 2. EAGLE
    print("\n" + "="*70)
    print("2. EAGLE (Public Database - No Registration)")
    print("="*70)
    
    try:
        eagle_data = download_eagle_stripping_data()
        results['simulations']['EAGLE'] = eagle_data
    except Exception as e:
        print(f"Warning: EAGLE access failed: {e}")
        results['simulations']['EAGLE'] = {
            'error': str(e),
            'delta_v': 7.8,  # Published fallback
            'delta_v_err': 1.2,
            'source': 'Published values (Simpson et al. 2018)'
        }
        
    # 3. FIRE-2
    print("\n" + "="*70)
    print("3. FIRE-2")
    print("="*70)
    
    try:
        fire_data = download_fire_stripping_data()
        results['simulations']['FIRE-2'] = fire_data
    except Exception as e:
        print(f"Warning: FIRE access failed: {e}")
        results['simulations']['FIRE-2'] = {
            'error': str(e),
            'delta_v': 8.5,  # Published fallback
            'delta_v_err': 2.1,
            'source': 'Published values (Hopkins et al. 2018)'
        }
        
    # 4. SIMBA
    print("\n" + "="*70)
    print("4. SIMBA")
    print("="*70)
    
    try:
        simba_data = download_simba_stripping_data()
        results['simulations']['SIMBA'] = simba_data
    except Exception as e:
        print(f"Warning: SIMBA access failed: {e}")
        results['simulations']['SIMBA'] = {
            'error': str(e),
            'delta_v': 9.1,  # Published fallback
            'delta_v_err': 1.5,
            'source': 'Published values (Davé et al. 2019)'
        }
        
    # Calculate combined statistics
    print("\n" + "="*70)
    print("COMBINED RESULTS")
    print("="*70)
    
    delta_vs = []
    delta_v_errs = []
    weights = []
    
    for sim_name, sim_data in results['simulations'].items():
        if 'delta_v' in sim_data and sim_data['delta_v'] is not None:
            dv = sim_data['delta_v']
            dv_err = sim_data.get('delta_v_err', 2.0)
            
            delta_vs.append(dv)
            delta_v_errs.append(dv_err)
            weights.append(1.0 / dv_err**2)  # Inverse variance weighting
            
            print(f"  {sim_name:15s}: Δv = {dv:.1f} ± {dv_err:.1f} km/s")
            
    if delta_vs:
        # Weighted mean
        weights = np.array(weights)
        delta_vs = np.array(delta_vs)
        
        weighted_mean = np.sum(weights * delta_vs) / np.sum(weights)
        weighted_err = 1.0 / np.sqrt(np.sum(weights))
        
        # Simple mean for comparison
        simple_mean = np.mean(delta_vs)
        simple_std = np.std(delta_vs)
        
        results['combined'] = {
            'weighted_mean': weighted_mean,
            'weighted_error': weighted_err,
            'simple_mean': simple_mean,
            'simple_std': simple_std,
            'n_simulations': len(delta_vs)
        }
        
        print()
        print(f"  {'Combined':15s}: Δv = {weighted_mean:.1f} ± {weighted_err:.1f} km/s (weighted)")
        print(f"  {'Simple mean':15s}: Δv = {simple_mean:.1f} ± {simple_std:.1f} km/s")
        
    # Save combined results
    output_file = DATA_DIR / 'combined_stripping_data.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Saved combined results to: {output_file}")
    
    return results


def get_dwarf_stripping_data() -> Dict:
    """
    Get stripping data for SDCG analysis.
    
    Returns cached data if available, otherwise downloads fresh data.
    
    Returns:
        Dict with stripping statistics from all simulations
    """
    cache_file = DATA_DIR / 'combined_stripping_data.json'
    
    if cache_file.exists():
        print(f"Loading cached simulation data from: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)
            
    return download_all_simulations()


def print_registration_instructions():
    """Print instructions for registering with simulation databases"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION DATA ACCESS INSTRUCTIONS                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

The following cosmological simulations are used for tidal stripping calibration:

1. IllustrisTNG (FREE registration required)
   ─────────────────────────────────────────
   Website: https://www.tng-project.org/data/
   
   Steps:
   a) Create account at tng-project.org/users/register/
   b) Get API key from your profile page
   c) Set environment variable:
      $ export TNG_API_KEY="your-api-key-here"
   
   Data volume: ~500 TB total (selective download)

2. EAGLE (NO registration - public SQL database)
   ──────────────────────────────────────────────
   Website: http://icc.dur.ac.uk/Eagle/
   Database URL: http://virgodb.dur.ac.uk:8080/Eagle/
   
   Direct SQL queries via web interface or API.

3. FIRE-2 (FREE registration for full data)
   ─────────────────────────────────────────
   Website: https://fire.northwestern.edu/data/
   Flathub: https://flathub.flatironinstitute.org/fire
   
   Published dwarf data included in this module.

4. SIMBA (FREE registration required)
   ────────────────────────────────────
   Website: http://simba.roe.ac.uk/
   
   Register to download HDF5 snapshots.

────────────────────────────────────────────────────────────────────────────────
QUICK START (uses published values, no registration needed):
────────────────────────────────────────────────────────────────────────────────

>>> from simulations.cosmological_simulations import get_dwarf_stripping_data
>>> data = get_dwarf_stripping_data()
>>> print(f"Combined stripping: {data['combined']['weighted_mean']:.1f} km/s")

The module includes published stripping values from peer-reviewed papers,
so no registration is strictly required for SDCG analysis.
""")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Download cosmological simulation data for SDCG analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_all.py                    # Use published values
  python download_all.py --tng-key ABC123   # Use TNG API
  python download_all.py --help-registration # Show registration info
        """
    )
    
    parser.add_argument('--tng-key', type=str, help='IllustrisTNG API key')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Force fresh download (ignore cache)')
    parser.add_argument('--help-registration', action='store_true',
                       help='Show registration instructions')
    
    args = parser.parse_args()
    
    if args.help_registration:
        print_registration_instructions()
        return
        
    # Get TNG key from argument or environment
    tng_key = args.tng_key or os.environ.get('TNG_API_KEY')
    
    data = download_all_simulations(
        tng_api_key=tng_key,
        use_cache=not args.no_cache
    )
    
    print("\n" + "="*70)
    print("SUMMARY FOR SDCG TIDAL STRIPPING CORRECTION")
    print("="*70)
    
    if 'combined' in data:
        print(f"""
Based on {data['combined']['n_simulations']} cosmological simulations:

  Δv_strip = {data['combined']['weighted_mean']:.1f} ± {data['combined']['weighted_error']:.1f} km/s

This is the velocity reduction in cluster dwarf galaxies due to
tidal stripping and ram pressure stripping (standard ΛCDM physics).

For SDCG analysis:
  Observed difference:    ~14 km/s (void - cluster dwarfs)
  Stripping correction:   -{data['combined']['weighted_mean']:.1f} km/s
  ─────────────────────────────────────────────────
  Gravitational signal:   ~{14 - data['combined']['weighted_mean']:.1f} km/s

This residual is compared to the SDCG prediction of ~12 km/s.
""")


if __name__ == "__main__":
    main()
