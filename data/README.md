# Real Cosmological Datasets

This directory contains cosmological data for testing the CGC theory.

## Contents:

1. **Planck 2018 CMB** - TT power spectrum (binned)
2. **BOSS DR12 BAO** - Baryon Acoustic Oscillations at z=0.38, 0.51, 0.61
3. **Pantheon+ SNe** - Compressed supernova distances (0.01 < z < 2.3)
4. **SH0ES 2022** - Hubble constant from Cepheid calibration
5. **RSD Growth** - fσ8 measurements from various surveys
6. **Lyman-α Forest** - Flux power spectrum from eBOSS

## Usage:

```python
from data_config import load_all_datasets

# Load all data
data = load_all_datasets()

# Access specific datasets
planck = data['planck_tt']
bao = data['boss_bao']
sne = data['pantheon']
