#!/usr/bin/env python3
"""
Verify that all cosmological datasets are loaded correctly
"""

import sys
import os

# Add data directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))

try:
    from data_config import list_datasets, load_all_datasets
    
    print("="*60)
    print("VERIFYING COSMOLOGICAL DATASETS")
    print("="*60)
    
    # List datasets
    list_datasets()
    
    # Try to load all datasets
    print("\nLoading datasets...")
    data = load_all_datasets()
    
    print(f"\n✓ Successfully loaded {len(data)} datasets")
    
    # Print summary
    print("\nDataset summary:")
    print("-"*40)
    for name, dataset in data.items():
        if name == 'planck_params':
            print(f"{name:15s}: {len(dataset)} parameters")
        elif 'z' in dataset:
            print(f"{name:15s}: {len(dataset['z'])} points")
        elif 'ell' in dataset:
            print(f"{name:15s}: {len(dataset['ell'])} multipoles")
        elif 'H0' in dataset:
            print(f"{name:15s}: H0 = {dataset['H0']:.1f} ± {dataset['H0_err']:.1f}")
    
    # Check for key parameters
    if 'planck_params' in data:
        print(f"\nPlanck 2018 reference:")
        print(f"  H0 = {data['planck_params']['H0']['value']:.1f} ± {data['planck_params']['H0']['error']:.1f}")
        print(f"  S8 = {data['planck_params']['S8']['value']:.3f} ± {data['planck_params']['S8']['error']:.3f}")
    
    if 'sh0es' in data:
        print(f"SH0ES 2022: H0 = {data['sh0es']['H0']:.1f} ± {data['sh0es']['H0_err']:.1f}")
    
    print("\n" + "="*60)
    print("DATA VERIFICATION COMPLETE")
    print("="*60)
    
except ImportError as e:
    print(f"Error: {e}")
    print("\nMake sure to run: ./fetch_real_cosmology_data.sh")
except Exception as e:
    print(f"Error: {e}")
