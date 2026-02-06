# Simulations Directory

This folder contains all simulation-related components for the CGC/SDCG cosmology analysis.

## Directory Structure

```
simulations/
├── cosmological_simulations/   # NEW: Real simulation data access
│   ├── illustris_tng.py        # IllustrisTNG API (FREE registration)
│   ├── eagle_database.py       # EAGLE public SQL database
│   ├── fire_data.py            # FIRE-2 simulation data
│   ├── simba_data.py           # SIMBA simulation data
│   ├── download_all.py         # Master download script
│   └── README.md               # Setup instructions
│
├── LaCE/                       # Lyα Cosmology Emulator (git submodule)
│   ├── lace/                   # Core LaCE library
│   ├── notebooks/              # Example notebooks
│   └── data/                   # Emulator training data
│
├── class_cgc/                  # Modified CLASS Boltzmann Solver
│   ├── source/                 # C source code with CGC modifications
│   ├── python/                 # Python wrapper (classy)
│   └── test/                   # Test configurations
│
├── stripping_models/           # Tidal Stripping Calibrations
│   ├── stripping_correction.py      # IllustrisTNG/EAGLE/SIMBA models
│   └── tidal_stripping_explained.py # Stripping physics documentation
│
├── mcmc_config/                # MCMC Sampler Configurations
│   ├── cgc_cobaya.yaml              # Cobaya sampler config
│   └── cobayaConfiguration.yaml     # Extended Cobaya settings
│
├── cgc_data/                   # CGC Simulation Outputs
│   ├── growth_evolution.txt         # f(z) growth history
│   └── hubble_evolution.txt         # H(z) expansion history
│
├── lace_env/                   # LaCE Python Virtual Environment
│
├── sdcg_real_data_pipeline.py  # Main data analysis pipeline
├── REAL_DATA_ANALYSIS.md       # Analysis methodology
└── REAL_DATA_STRATEGY.md       # Data strategy documentation
```

## Cosmological Simulations Used

### Tidal Stripping Calibrations
The stripping models are calibrated using state-of-the-art hydrodynamical simulations:

| Simulation | Reference | Mass Loss | Velocity Effect |
|------------|-----------|-----------|-----------------|
| **EAGLE** | Schaye+ 2015 | 30-50% | 7-9 km/s |
| **IllustrisTNG** | Pillepich+ 2018 | 35-55% | 8-10 km/s |
| **FIRE-2** | Hopkins+ 2018 | 25-45% | 6-9 km/s |
| **SIMBA** | Davé+ 2019 | 30-45% | 7-8 km/s |

### CLASS Boltzmann Solver
Modified version of [CLASS](https://github.com/lesgourg/class_public) with CGC gravity modifications:
- Scale-dependent growth: G_eff(k,z,ρ)
- Environment-dependent screening
- Modified power spectrum output

### LaCE Emulator
[LaCE](https://github.com/igmhub/LaCE) - Lyα Cosmology Emulator:
- Trained on hydrodynamical simulations
- Fast P1D power spectrum predictions
- Used for Lyα forest constraint validation

## Usage

### Running CLASS with CGC
```bash
cd simulations/class_cgc
make clean && make
./class explanatory.ini
```

### Using Stripping Corrections
```python
from simulations.stripping_models.stripping_correction import (
    calculate_stripping_correction,
    STRIPPING_MODELS
)

# Use IllustrisTNG calibration
correction = calculate_stripping_correction(galaxy_data, simulation='IllustrisTNG')
```

### Cobaya MCMC
```bash
cobaya-run simulations/mcmc_config/cgc_cobaya.yaml
```

## References

1. **EAGLE**: Schaye, J., et al. (2015). MNRAS, 446, 521.
2. **IllustrisTNG**: Pillepich, A., et al. (2018). MNRAS, 473, 4077.
3. **FIRE-2**: Hopkins, P. F., et al. (2018). MNRAS, 480, 800.
4. **SIMBA**: Davé, R., et al. (2019). MNRAS, 486, 2827.
5. **CLASS**: Lesgourgues, J. (2011). arXiv:1104.2932
6. **LaCE**: Pedersen, C., et al. (2021). JCAP, 05, 033.
