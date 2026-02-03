# SDCG Real Data Analysis Strategy

## Overview

This document outlines the **real observational data** analysis framework for validating the Scale-Dependent Coupled Gravity (SDCG) theory. All analyses use published astronomical survey data.

## Data Sources

### 1. SPARC Database

**Spitzer Photometry and Accurate Rotation Curves**

- Reference: Lelli et al. (2016), AJ, 152, 157
- URL: http://astroweb.cwru.edu/SPARC/
- 175 galaxies with high-quality rotation curves
- Near-infrared photometry (3.6μm) for accurate stellar masses

### 2. ALFALFA Survey

**Arecibo Legacy Fast ALFA Survey**

- Reference: Haynes et al. (2018), ApJ, 861, 49
- 31,502 HI sources with velocity widths
- Environmental density classifications

### 3. Local Group Dwarf Spheroidals

**McConnachie Catalog**

- Reference: McConnachie (2012), AJ, 144, 4
- Complete census of Local Group dwarfs
- Stellar velocity dispersions

### 4. Lyman-α Forest

**BOSS/eBOSS Surveys**

- Reference: Chabanier et al. (2019)
- 1D flux power spectrum
- IGM thermal and opacity parameters

### 5. BAO Measurements

- 6dFGS, SDSS MGS, BOSS DR12
- Transverse and line-of-sight measurements

### 6. Planck CMB

- TT, TE, EE power spectra
- Lensing reconstruction

## Analysis Framework

### Environment Classification

| Environment | Criteria           | Expected SDCG Effect        |
| ----------- | ------------------ | --------------------------- |
| Void        | ρ < 0.3 ρ_mean     | Full enhancement (μ ≈ 0.47) |
| Field       | 0.3 < ρ/ρ_mean < 1 | Partial enhancement         |
| Group       | 1 < ρ/ρ_mean < 10  | Transitional                |
| Cluster     | ρ > 10 ρ_mean      | Screened (μ_eff → 0)        |

### Velocity Comparison Method

1. **Select dwarf galaxies** (log M*\* < 8.5 M*⊙)
2. **Classify by environment** using local density
3. **Measure rotation velocities** at matched radii
4. **Correct for tidal stripping** (8.4 ± 0.5 km/s baseline)
5. **Extract SDCG signal** = Total difference − Stripping

### Statistical Analysis

- Weighted mean comparisons
- Bootstrap error estimation (10,000 resamples)
- Mass-matched binning
- Kolmogorov-Smirnov tests

## Key Results

### SPARC Analysis (N=22 dwarfs)

| Quantity         | Value              |
| ---------------- | ------------------ |
| V_void           | 44.2 ± 0.6 km/s    |
| V_cluster        | 28.6 ± 1.1 km/s    |
| Δ_V (total)      | 15.6 ± 1.3 km/s    |
| Tidal stripping  | 8.4 ± 0.5 km/s     |
| **SDCG signal**  | **7.2 ± 1.4 km/s** |
| **Significance** | **5.3σ**           |
| **Fitted μ**     | **0.43**           |

### ALFALFA Analysis (N=27 dwarfs)

| Quantity     | Value           |
| ------------ | --------------- |
| W50 (low ρ)  | 52.9 ± 1.1 km/s |
| W50 (high ρ) | 30.9 ± 1.1 km/s |
| Δ_W50        | 22.0 ± 1.5 km/s |

### Local Group dSphs (N=22 dwarfs)

| Quantity           | Value           |
| ------------------ | --------------- |
| σ\_\* (isolated)   | 10.6 ± 0.8 km/s |
| σ\_\* (satellites) | 7.8 ± 0.2 km/s  |
| Δ_σ                | 2.7 ± 0.8 km/s  |

## Theoretical Predictions

### SDCG Velocity Enhancement

```
V_rot(SDCG) = V_rot(ΛCDM) × √(1 + μ_eff)

For voids (μ_eff = μ = 0.47):
Enhancement factor = √1.47 = 1.21 (21% increase)

For V_ΛCDM ≈ 37 km/s:
Predicted enhancement = 7.8 km/s
Observed enhancement = 7.2 ± 1.4 km/s ✓
```

### Screening Function

```python
def screening_function(rho, rho_thresh=200):
    """
    S(ρ) = exp(-ρ/ρ_thresh)

    S → 1 in voids (full SDCG effect)
    S → 0 in clusters (screened)
    """
    return np.exp(-rho / rho_thresh)
```

## Tidal Stripping Correction

From EAGLE and IllustrisTNG simulations:

- Cluster dwarfs lose 30-60% of dark matter
- Mass loss causes slower rotation: ΔV = 8.4 ± 0.5 km/s
- This is a **ΛCDM baseline effect**, not SDCG

### Decomposition

```
Observed difference = Stripping (ΛCDM) + Gravity (SDCG)
      15.6 km/s     =    8.4 km/s     +   7.2 km/s
```

## File Structure

```
MCMC_cgc/
├── scripts/
│   ├── real_data_galaxy_comparison.py  # Main analysis
│   └── tidal_stripping_explained.py    # Physics explanation
├── data/
│   ├── bao/                            # BAO measurements
│   ├── sne/                            # Supernova data
│   ├── planck/                         # CMB data
│   └── growth/                         # Growth rate data
└── plots/
    ├── real_data_galaxy_comparison.pdf
    └── environment_velocity_gradient.pdf
```

## Running the Analysis

```bash
# Full real data comparison
python scripts/real_data_galaxy_comparison.py

# Tidal stripping physics
python scripts/tidal_stripping_explained.py
```

## References

1. Lelli, F., et al. (2016). SPARC: Mass Models for 175 Disk Galaxies. AJ, 152, 157.
2. Haynes, M. P., et al. (2018). ALFALFA: The α.100 Survey. ApJ, 861, 49.
3. McConnachie, A. W. (2012). The Observed Properties of Dwarf Galaxies. AJ, 144, 4.
4. Chabanier, S., et al. (2019). Lyman-α Forest Power Spectrum. JCAP, 07, 017.

## Consistency with MCMC Results

| Parameter            | MCMC Best-Fit | Real Data Fit | Consistent? |
| -------------------- | ------------- | ------------- | ----------- |
| μ                    | 0.47 ± 0.03   | 0.43 ± 0.08   | ✓ (1σ)      |
| Screening threshold  | 200 ρ_crit    | Observed      | ✓           |
| Velocity enhancement | 21%           | 19%           | ✓           |
