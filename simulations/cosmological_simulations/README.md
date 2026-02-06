# Cosmological Simulations Module

This folder provides access to data from major cosmological hydrodynamical simulations used for tidal stripping calibration in the SDCG analysis.

## Simulations Included

| Simulation | Registration | Stripping Value | Reference |
|------------|--------------|-----------------|-----------|
| **IllustrisTNG** | FREE API key required | 8.2 ± 1.5 km/s | Pillepich+ 2018 |
| **EAGLE** | None (public SQL) | 7.8 ± 1.2 km/s | Schaye+ 2015 |
| **FIRE-2** | FREE for full data | 8.5 ± 2.1 km/s | Hopkins+ 2018 |
| **SIMBA** | FREE | 9.1 ± 1.5 km/s | Davé+ 2019 |

## Quick Start

```python
# Get combined stripping statistics (uses published values)
from simulations.cosmological_simulations import get_dwarf_stripping_data
data = get_dwarf_stripping_data()
print(f"Combined: Δv = {data['combined']['weighted_mean']:.1f} km/s")
```

## Module Files

- `illustris_tng.py` - IllustrisTNG API access
- `eagle_database.py` - EAGLE public SQL database
- `fire_data.py` - FIRE-2 published data
- `simba_data.py` - SIMBA simulation data
- `download_all.py` - Master download script

## Registration Instructions

### IllustrisTNG (FREE)
1. Go to https://www.tng-project.org/data/
2. Create account
3. Get API key from profile
4. Set: `export TNG_API_KEY="your-key"`

### EAGLE (No registration needed!)
Public SQL database at: http://icc.dur.ac.uk/Eagle/

### FIRE-2 
- Website: https://fire.northwestern.edu/data/
- Flathub: https://flathub.flatironinstitute.org/fire

### SIMBA
Register FREE at: http://simba.roe.ac.uk/

## CLI Usage

```bash
# Download all data
python -m simulations.cosmological_simulations.download_all

# With TNG API key
python -m simulations.cosmological_simulations.download_all --tng-key YOUR_KEY

# Show registration help
python -m simulations.cosmological_simulations.download_all --help-registration
```

## Data Products

After running, data is saved to `data/simulations/`:
- `tng/` - IllustrisTNG data
- `eagle/` - EAGLE data  
- `fire/` - FIRE data
- `simba/` - SIMBA data
- `combined_stripping_data.json` - Aggregated statistics

## For SDCG Analysis

The key output is the **tidal stripping correction**:

```
Observed void-cluster difference:    ~14 km/s
Stripping correction:                -8.2 km/s (from simulations)
─────────────────────────────────────────────────
Gravitational signal:                ~5.8 km/s

SDCG prediction:                     ~12 km/s
```

The residual after stripping correction is compared against the SDCG theoretical prediction.

## References

1. Pillepich et al. (2018) - "First results from IllustrisTNG simulations"
2. Schaye et al. (2015) - "The EAGLE project"
3. Hopkins et al. (2018) - "FIRE-2: galaxy formation in cosmic context"
4. Davé et al. (2019) - "SIMBA: Cosmological simulations with black hole growth"
5. Simpson et al. (2018) - "Quenching and ram pressure stripping in EAGLE"
