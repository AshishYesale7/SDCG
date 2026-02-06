# MCMC_cgc Project Status Report
Generated: 2026-02-06

## AUDIT SUMMARY

### Critical Issues Found: 3
1. **H0 Inconsistency** - Values range 61.7 - 76.3 km/s/Mpc across result files
   - v12 expected: 67.4 km/s/Mpc (Planck 2018)
   - Fixed: Old files archived to `results/_archive/`

2. **Parameter False Positives** - Most "h" parameter issues are in library files (scipy/matplotlib)
   - These are expected and not bugs in project code

3. **JSON Config Differences** - Some JSON files have different values for same parameters
   - Status: Most are different test runs, not bugs

### Files Archived (Incorrect/Obsolete)
- `_archive/cgc_mcmc_results.npz` (H0 = 76.33 - way off expected)
- `_archive/cgc_real_analysis_20260130_002836.npz` (early test)
- `_archive/cgc_results_fixed_20260130_004223.npz` (early test)
- `_archive/cgc_summary_20260129_235422.txt`
- `_archive/cgc_summary_fixed_20260130_004223.txt`

---

## CURRENT ACTIVE RESULT FILES

### Latest Analysis (USE THESE)
| File | Date | Key Results |
|------|------|-------------|
| `cgc_dwarf_mcmc_results.npz` | 2026-02-06 | Δv = -1.85 ± 2.92 km/s |
| `cgc_lace_comprehensive_v6.npz` | 2026-02-06 | μ = 0.41, n_g = 0.65 |
| `sdcg_definitive_analysis.npz` | 2026-02-01 | μ_no_lyα = 0.48, tension 3.9σ |
| `cgc_thesis_lyalpha_comparison.npz` | 2026-02-06 | Complete Lyα comparison |

### Production MCMC Chains
| File | Date | Description |
|------|------|-------------|
| `cgc_production_chains_20260204_225711.npz` | 2026-02-04 | Full production run |
| `cgc_long_mcmc_20260204_210022.npz` | 2026-02-04 | Extended MCMC chain |
| `mcmc_chains.npz` | 2026-02-04 | Standard chains, acceptance = 0.35 |

### EFT Prior Analysis
| File | Date | Description |
|------|------|-------------|
| `sdcg_eft_prior_20260203_131759.npz` | 2026-02-03 | EFT-constrained run 1 |
| `sdcg_eft_prior_20260203_135042.npz` | 2026-02-03 | EFT-constrained run 2 |
| `fixed_eft_mcmc_results.npz` | 2026-02-04 | Fixed EFT parameters |

---

## v12 PARAMETER REFERENCE

### Cosmology (Planck 2018)
```
H0 = 67.4 ± 0.5 km/s/Mpc
Ωm = 0.315 ± 0.007
Ωb = 0.0493
σ8 = 0.811 ± 0.006
S8 = 0.832 ± 0.013
n_s = 0.965
```

### CGC/SDCG Model
```
μ = 0.41 ± 0.04 (MCMC without Lyα)
μ < 0.012 (95% CL with Lyα)
n_g = 0.65 ± 0.05
z_trans = 2.43 ± 0.15
```

### Dwarf Galaxy Predictions
```
Δv_strip = 7.9 ± 0.9 km/s (from simulations)
Δv_observed = -2.49 ± 5.0 km/s
```

---

## DATA DIRECTORIES STATUS

| Directory | Status | Files | Size |
|-----------|--------|-------|------|
| `data/planck/` | ✓ OK | 2 | 0.16 MB |
| `data/bao/` | ✓ OK | 1 | 0.00 MB |
| `data/sne/` | ✓ OK | 5 | 32.30 MB |
| `data/growth/` | ✓ OK | 1 | 0.00 MB |
| `data/lyalpha/` | ✓ OK | 13 | 0.29 MB |
| `data/little_things/` | ✓ OK | 6 | 0.02 MB |
| `data/cgc_simulations/` | ✓ OK | 2 | 0.01 MB |

---

## FILES USING v12 PARAMETERS CORRECTLY
- `v12_parameters.py` - Single source of truth (NEW)
- `verify_equations.py` - H0=67.4, Ωm=0.315 ✓
- Core MCMC files in `cgc_theory/`

## FILES WITH NON-v12 VALUES (BY DESIGN)
- `fetch_real_cosmology_api.py` - Uses SH0ES H0=73.04 for Pantheon+ calibration
  - This is CORRECT - SH0ES measurement is 73.04

---

## RECOMMENDATIONS

1. **Import from `v12_parameters.py`** for all new code:
   ```python
   from v12_parameters import COSMO, CGC, DWARF
   H0 = COSMO['H0']  # 67.4
   ```

2. **Use latest result files** listed above

3. **Archive old runs** when creating new analyses

---

## FILE COUNTS
- Python files: 2795 (most in virtual environments)
- JSON configs: 47
- Data files: 294
- Result files (.npz): 44 (3 archived)
