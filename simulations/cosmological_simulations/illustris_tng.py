#!/usr/bin/env python3
"""
IllustrisTNG Simulation Data Access
===================================

IllustrisTNG is a suite of large-volume cosmological magnetohydrodynamical 
simulations. Data is freely available via web API.

Registration Required:
    1. Go to https://www.tng-project.org/data/
    2. Create a free account
    3. Get your API key from your profile page
    4. Set environment variable: export TNG_API_KEY="your-key-here"
    
Or save key to: ~/.tng_api_key

References:
    - Pillepich et al. (2018) - First results from IllustrisTNG
    - Nelson et al. (2019) - IllustrisTNG public data release
    
API Documentation: https://www.tng-project.org/data/docs/api/
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import warnings

try:
    import requests
except ImportError:
    requests = None
    warnings.warn("requests package not installed. Install with: pip install requests")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Setup paths first
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    # Load .env from project root
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    warnings.warn("python-dotenv not installed. Install with: pip install python-dotenv")

DATA_DIR = PROJECT_ROOT / 'data' / 'simulations' / 'tng'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# TNG API base URL
TNG_BASE_URL = "https://www.tng-project.org/api"

@dataclass
class TNGDwarfGalaxy:
    """Dwarf galaxy data from IllustrisTNG"""
    subhalo_id: int
    stellar_mass: float  # M_sun
    dm_mass: float  # M_sun
    half_mass_radius: float  # kpc
    v_max: float  # km/s
    v_disp: float  # km/s (stellar velocity dispersion)
    environment: str  # 'cluster', 'group', 'field'
    host_mass: Optional[float] = None  # Host halo mass if satellite
    distance_to_host: Optional[float] = None  # kpc
    infall_time: Optional[float] = None  # Gyr


class IllustrisTNGAccess:
    """
    Access IllustrisTNG simulation data via public API.
    
    Example usage:
        >>> tng = IllustrisTNGAccess()
        >>> tng.set_api_key("your-api-key")
        >>> dwarfs = tng.get_dwarf_galaxies(simulation='TNG100-1', min_mass=1e7, max_mass=1e9)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TNG access.
        
        Args:
            api_key: Your TNG API key. If None, tries to read from:
                    1. TNG_API_KEY environment variable
                    2. ~/.tng_api_key file
        """
        self.api_key = api_key or self._load_api_key()
        self.base_url = TNG_BASE_URL
        self.headers = {"api-key": self.api_key} if self.api_key else {}
        
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or file"""
        # Try environment variable
        key = os.environ.get('TNG_API_KEY')
        if key:
            return key
            
        # Try file
        key_file = Path.home() / '.tng_api_key'
        if key_file.exists():
            return key_file.read_text().strip()
            
        return None
        
    def set_api_key(self, api_key: str):
        """Set the API key"""
        self.api_key = api_key
        self.headers = {"api-key": api_key}
        
        # Optionally save to file
        key_file = Path.home() / '.tng_api_key'
        key_file.write_text(api_key)
        print(f"API key saved to {key_file}")
        
    def _check_api_key(self):
        """Check if API key is set"""
        if not self.api_key:
            raise ValueError(
                "TNG API key not set. Get one free at:\n"
                "https://www.tng-project.org/data/\n"
                "Then call: tng.set_api_key('your-key')\n"
                "Or set environment variable: export TNG_API_KEY='your-key'"
            )
            
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to TNG API"""
        if requests is None:
            raise ImportError("requests package required. Install with: pip install requests")
            
        self._check_api_key()
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 401:
            raise PermissionError("Invalid API key. Check your key at tng-project.org")
        elif response.status_code == 404:
            raise FileNotFoundError(f"Endpoint not found: {endpoint}")
        elif response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text}")
            
        return response.json()
        
    def list_simulations(self) -> List[Dict]:
        """
        List all available TNG simulations.
        
        Returns:
            List of simulation info dicts
        """
        return self._get("")
        
    def get_simulation_info(self, simulation: str = "TNG100-1") -> Dict:
        """
        Get info about a specific simulation.
        
        Args:
            simulation: Simulation name (e.g., 'TNG100-1', 'TNG50-1', 'TNG300-1')
            
        Returns:
            Simulation metadata
        """
        return self._get(simulation)
        
    def get_subhalos(self, simulation: str = "TNG100-1", snapshot: int = 99,
                     limit: int = 100, offset: int = 0,
                     mass_min: Optional[float] = None,
                     mass_max: Optional[float] = None) -> List[Dict]:
        """
        Query subhalos (galaxies) from a simulation snapshot.
        
        Args:
            simulation: Simulation name
            snapshot: Snapshot number (99 = z=0 for TNG)
            limit: Max number of results
            offset: Pagination offset
            mass_min: Minimum stellar mass in M_sun
            mass_max: Maximum stellar mass in M_sun
            
        Returns:
            List of subhalo data dicts
        """
        endpoint = f"{simulation}/snapshots/{snapshot}/subhalos/"
        
        params = {
            'limit': limit,
            'offset': offset
        }
        
        # Add mass filters if specified
        if mass_min is not None:
            params['mass_stars__gt'] = mass_min / 1e10  # TNG uses 10^10 M_sun units
        if mass_max is not None:
            params['mass_stars__lt'] = mass_max / 1e10
            
        return self._get(endpoint, params)
        
    def get_subhalo_details(self, simulation: str, snapshot: int, 
                           subhalo_id: int) -> Dict:
        """
        Get detailed data for a specific subhalo.
        
        Args:
            simulation: Simulation name
            snapshot: Snapshot number
            subhalo_id: Subhalo ID
            
        Returns:
            Detailed subhalo data
        """
        endpoint = f"{simulation}/snapshots/{snapshot}/subhalos/{subhalo_id}/"
        return self._get(endpoint)
        
    def get_dwarf_galaxies(self, simulation: str = "TNG100-1",
                          min_mass: float = 1e7, max_mass: float = 1e9,
                          limit: int = 500) -> List[TNGDwarfGalaxy]:
        """
        Get dwarf galaxies from TNG simulation.
        
        Args:
            simulation: Simulation name
            min_mass: Minimum stellar mass (M_sun)
            max_mass: Maximum stellar mass (M_sun)
            limit: Maximum number of galaxies
            
        Returns:
            List of TNGDwarfGalaxy objects
        """
        subhalos = self.get_subhalos(
            simulation=simulation,
            snapshot=99,  # z=0
            limit=limit,
            mass_min=min_mass,
            mass_max=max_mass
        )
        
        dwarf_galaxies = []
        
        for sh in subhalos.get('results', []):
            # Get detailed info
            try:
                details = self.get_subhalo_details(
                    simulation, 99, sh['id']
                )
                
                # Determine environment based on host properties
                if details.get('SubhaloGrNr', -1) >= 0:
                    # Get host halo mass
                    host_mass = details.get('Group_M_Crit200', 0) * 1e10  # M_sun
                    if host_mass > 1e14:
                        environment = 'cluster'
                    elif host_mass > 1e12:
                        environment = 'group'
                    else:
                        environment = 'field'
                else:
                    environment = 'field'
                    host_mass = None
                    
                dwarf = TNGDwarfGalaxy(
                    subhalo_id=sh['id'],
                    stellar_mass=details.get('SubhaloMassType', [0]*6)[4] * 1e10,  # Stars
                    dm_mass=details.get('SubhaloMassType', [0]*6)[1] * 1e10,  # DM
                    half_mass_radius=details.get('SubhaloHalfmassRadType', [0]*6)[4],  # kpc
                    v_max=details.get('SubhaloVmax', 0),  # km/s
                    v_disp=details.get('SubhaloVelDisp', 0),  # km/s
                    environment=environment,
                    host_mass=host_mass,
                    distance_to_host=details.get('SubhaloCM', None)  # Would need more calc
                )
                dwarf_galaxies.append(dwarf)
                
            except Exception as e:
                print(f"Warning: Could not process subhalo {sh['id']}: {e}")
                
        return dwarf_galaxies
        
    def calculate_stripping_statistics(self, simulation: str = "TNG100-1") -> Dict:
        """
        Calculate tidal stripping statistics for dwarf galaxies in clusters vs field.
        
        This is the key data needed for SDCG analysis.
        
        Returns:
            Dict with stripping statistics:
            - mean_v_cluster: Mean velocity in clusters
            - mean_v_field: Mean velocity in field
            - delta_v: Velocity difference (used for stripping correction)
            - n_cluster: Number of cluster dwarfs
            - n_field: Number of field dwarfs
        """
        dwarfs = self.get_dwarf_galaxies(simulation=simulation, limit=1000)
        
        cluster_v = [d.v_max for d in dwarfs if d.environment == 'cluster']
        field_v = [d.v_max for d in dwarfs if d.environment == 'field']
        
        if not cluster_v or not field_v:
            return {"error": "Insufficient data"}
            
        return {
            'simulation': simulation,
            'mean_v_cluster': np.mean(cluster_v),
            'mean_v_field': np.mean(field_v),
            'std_v_cluster': np.std(cluster_v),
            'std_v_field': np.std(field_v),
            'delta_v': np.mean(field_v) - np.mean(cluster_v),  # Stripping effect
            'delta_v_err': np.sqrt(np.var(cluster_v)/len(cluster_v) + 
                                   np.var(field_v)/len(field_v)),
            'n_cluster': len(cluster_v),
            'n_field': len(field_v)
        }
        
    def save_cache(self, data: Any, filename: str):
        """Save downloaded data to cache"""
        cache_path = DATA_DIR / filename
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved: {cache_path}")
        
    def load_cache(self, filename: str) -> Optional[Any]:
        """Load data from cache"""
        cache_path = DATA_DIR / filename
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None


def download_tng_stripping_data(api_key: Optional[str] = None) -> Dict:
    """
    Download IllustrisTNG stripping data for SDCG analysis.
    
    This function queries the TNG API to get velocity differences between
    cluster and field dwarf galaxies - the key observable for stripping corrections.
    
    Args:
        api_key: TNG API key (get free at tng-project.org)
        
    Returns:
        Dict with stripping statistics from TNG simulations
    """
    tng = IllustrisTNGAccess(api_key)
    
    print("="*60)
    print("Downloading IllustrisTNG Tidal Stripping Data")
    print("="*60)
    
    # Try to load from cache first
    cached = tng.load_cache('tng_stripping_data.json')
    if cached:
        print("Loaded from cache")
        return cached
        
    results = {}
    
    # Query different TNG simulations
    for sim in ['TNG100-1', 'TNG50-1']:
        try:
            print(f"\nQuerying {sim}...")
            stats = tng.calculate_stripping_statistics(sim)
            results[sim] = stats
            print(f"  Δv_strip = {stats['delta_v']:.1f} ± {stats['delta_v_err']:.1f} km/s")
        except Exception as e:
            print(f"  Warning: {e}")
            
    # Save to cache
    if results:
        tng.save_cache(results, 'tng_stripping_data.json')
        
    return results


# =============================================================================
# Main - Quick test
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
    
    tng = IllustrisTNGAccess()
    
    if tng.api_key:
        print("\n✓ API key found!")
        print("\nAvailable simulations:")
        try:
            sims = tng.list_simulations()
            for sim in sims[:5]:
                print(f"  - {sim.get('name', sim)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n" + "="*60)
        print("HOW TO GET STARTED:")
        print("="*60)
        print("""
1. Register FREE at: https://www.tng-project.org/data/

2. After registration, get your API key from your profile

3. Set your API key:
   
   Option A - Environment variable:
   $ export TNG_API_KEY="your-api-key-here"
   
   Option B - In Python:
   >>> tng = IllustrisTNGAccess()
   >>> tng.set_api_key("your-api-key-here")
   
4. Download stripping data:
   >>> from simulations.cosmological_simulations.illustris_tng import download_tng_stripping_data
   >>> data = download_tng_stripping_data()

Registration is FREE and provides access to ~500 TB of simulation data!
""")
