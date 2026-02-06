# Void vs. Cluster Dwarf Galaxy Rotation Test

## SDCG Falsification Test - Using REAL Observational Data

This directory contains the complete pipeline to test the SDCG prediction that **void dwarf galaxies should rotate faster** than cluster dwarf galaxies due to reduced chameleon screening.

> ⚠️ **REAL DATA ONLY**: This analysis requires real ALFALFA and SDSS observational data.
> Synthetic/mock data is NOT permitted for scientific validity.

### Prediction Summary

| SDCG Version | Predicted Δv | Status |
|--------------|--------------|--------|
| Unconstrained (μ = 0.41) | +12-15 km/s | Likely falsified |
| Lyα-Constrained (μ = 0.045) | +0.5 km/s | Within error bars |
| ΛCDM (μ = 0) | 0 km/s | Null hypothesis |

### Required Data

1. **SDSS Cosmic Void Catalog** (Pan et al., 2012)
   - 1,054 voids from SDSS DR7
   - Provides void centers and radii

2. **ALFALFA α.100 Catalog**
   - HI line widths (W50) for ~31,500 sources
   - Provides rotation velocity proxy

### Execution Steps

```bash
# 1. Download data
python download_catalogs.py

# 2. Cross-match and filter
python filter_dwarf_sample.py

# 3. Run analysis
python run_void_cluster_analysis.py

# 4. Generate results
python interpret_results.py
```

### Test Criteria

| Result (Δv) | Interpretation | SDCG Status |
|-------------|----------------|-------------|
| +12 ± 3 km/s | Void dwarfs faster | ✓ Confirmed |
| +0.5 ± 0.5 km/s | Small effect | ✓ Lyα-consistent |
| 0 ± 2 km/s | No difference | ✗ Falsified |
| -2.5 ± 5 km/s | Inverted signal | ✗ Falsified |

### Previous Result (Version 7)

Preliminary ALFALFA analysis found:
- **Δv = -2.49 ± 5.0 km/s**
- Consistent with zero (ΛCDM) within 1σ
- Rules out unconstrained SDCG (+15 km/s) at >3σ
- Lyα-constrained prediction (+0.5 km/s) is within error bars
