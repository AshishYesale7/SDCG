"""
Path Utilities for CGC Project
==============================

Provides consistent path handling across all scripts.

Usage:
    from path_utils import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, PLOTS_DIR
    
    # Or:
    import path_utils
    data_path = path_utils.get_data_path('planck/planck_TT_binned.txt')
"""

import os
from pathlib import Path

def find_project_root():
    """Find the project root directory by looking for key markers."""
    current = Path(__file__).resolve().parent
    
    # Walk up the directory tree looking for project markers
    markers = ['simulations', 'data', '.git', 'README.md']
    
    for _ in range(10):  # Max 10 levels up
        if all((current / marker).exists() for marker in ['simulations', 'data']):
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: use the environment variable or cwd
    if 'CGC_PROJECT_ROOT' in os.environ:
        return Path(os.environ['CGC_PROJECT_ROOT'])
    
    return Path.cwd()

# Main paths
PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = PROJECT_ROOT / 'plots'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
SIMULATIONS_DIR = PROJECT_ROOT / 'simulations'

# CGC module path
CGC_MODULE_DIR = SIMULATIONS_DIR / 'cgc'

def get_data_path(relative_path: str) -> Path:
    """Get absolute path to a data file."""
    return DATA_DIR / relative_path

def get_results_path(relative_path: str) -> Path:
    """Get absolute path to a results file."""
    return RESULTS_DIR / relative_path

def get_plots_path(relative_path: str) -> Path:
    """Get absolute path to a plots file."""
    return PLOTS_DIR / relative_path

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def setup_python_path():
    """Add project root to Python path for imports."""
    import sys
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

# Print info when run directly
if __name__ == '__main__':
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"PLOTS_DIR: {PLOTS_DIR}")
    print(f"CGC_MODULE_DIR: {CGC_MODULE_DIR}")
    print()
    print(f"Data dir exists: {DATA_DIR.exists()}")
    print(f"Results dir exists: {RESULTS_DIR.exists()}")
