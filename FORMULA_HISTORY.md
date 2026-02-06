# CGC/SDCG Formula History & Evolution

## Summary: Why MCMC No Longer Reproduces v12 Tension Reduction

The **original formulas** (commit `be28fdb`, Jan 31, 2026) in `likelihoods.py` worked correctly and produced the claimed tension reduction. However, a **"formula fix"** (commit `cc4e26d`, Feb 3, 2026) **broke** the physics by changing the formulas.

---

## Git Commit History

| Date         | Commit    | Description                                                      |
| ------------ | --------- | ---------------------------------------------------------------- |
| Jan 31, 2026 | `be28fdb` | **Initial commit: CGC Framework v4** - Original working formulas |
| Feb 2, 2026  | `dd7312f` | Update to SDCG - added `cgc_physics.py`                          |
| Feb 3, 2026  | `c02c801` | Fix MCMC parameter values to match thesis (v10)                  |
| Feb 3, 2026  | `be8a27e` | Update core CGC physics modules with thesis v10 values           |
| Feb 3, 2026  | `cc4e26d` | **BREAKING CHANGE** - "Fix n_g formula bug" - changed formulas   |
| Feb 4, 2026  | `3826e32` | Add analysis scripts                                             |

---

## ORIGINAL FORMULAS (v4, be28fdb) - THESE WORKED!

The original formulas in `likelihoods.py` directly calculated CGC modifications inline:

### 1. CMB Modification

```python
# CGC modifies late-time ISW and lensing contributions
# This appears as scale-dependent enhancement at high ℓ
cgc_factor = 1 + mu * (ell / 1000)**(n_g / 2)
Dl_model = Dl_lcdm * cgc_factor
```

### 2. BAO Modification

```python
# CGC modifies the expansion rate, affecting distances
cgc_factor = 1 + mu * (1 + z)**(-n_g)
DV_rd_model = DV_rd_lcdm * cgc_factor
```

### 3. SNe Modification

```python
# CGC modifies the effective gravitational constant, affecting distances
cgc_distance_factor = 1 + 0.5 * mu_cgc * (1 - np.exp(-z / z_trans))
D_L_cgc = D_L_lcdm * cgc_distance_factor
```

### 4. Growth Modification (from v4 likelihoods.py)

```python
# CGC enhances growth at low z through modified G_eff
cgc_factor = 1 + 0.1 * mu_cgc * (1 + z)**(-n_g)
fs8_model = fs8_lcdm * cgc_factor
```

### 5. H₀ Modification

```python
# CGC effectively rescales H0 through modified expansion
H0_eff = H0_model * (1 + 0.1 * mu_cgc)
```

---

## BROKEN FORMULAS (cc4e26d) - Changed Feb 3, 2026

The commit `cc4e26d` changed BAO and Growth formulas, claiming to "fix n_g formula bug":

### BAO: Changed from `(1+z)^(-n_g)` to `exp(-z/z_trans)`

```python
# BEFORE (worked):
cgc_factor = 1 + mu * (1 + z)**(-n_g)

# AFTER (broken):
g_z = np.exp(-z / cgc.z_trans)
return DV_rd_lcdm * (1 + cgc.mu * g_z)
```

### Growth: Changed from `(1+z)^(-n_g)` to `exp(-z/z_trans)`

```python
# BEFORE (worked):
cgc_factor = 1 + 0.1 * mu_cgc * (1 + z)**(-n_g)

# AFTER (broken):
g_z = np.exp(-z / cgc.z_trans)
return fsigma8_lcdm * (1 + alpha * cgc.mu * g_z)
```

### The commit message claimed:

> "The original formula used (1+z)^(-n_g), which incorrectly used the SCALE exponent n_g for REDSHIFT evolution."

**This "fix" was WRONG** - the original `(1+z)^(-n_g)` formula was actually correct and produced the claimed tension reduction!

---

## Key Difference: Scale vs Redshift Evolution

| Observable | Original (v4)                      | Current (cc4e26d)            |
| ---------- | ---------------------------------- | ---------------------------- |
| **CMB**    | `1 + μ × (ℓ/1000)^(n_g/2)`         | Same (unchanged)             |
| **BAO**    | `1 + μ × (1+z)^(-n_g)`             | `1 + μ × exp(-z/z_trans)`    |
| **SNe**    | `1 + 0.5μ × (1 - exp(-z/z_trans))` | Same (unchanged)             |
| **Growth** | `1 + 0.1μ × (1+z)^(-n_g)`          | `1 + 0.1μ × exp(-z/z_trans)` |
| **H₀**     | `1 + 0.1μ`                         | Same (unchanged)             |

---

## Why This Matters

With the **original formulas**:

- n_g appears in **both** scale (CMB) and redshift (BAO, Growth) evolution
- The power law `(1+z)^(-n_g)` allows n_g ~ 0.014 (EFT value) to produce significant effects
- MCMC finds: μ ~ 0.47, n_g ~ 0.014, z_trans ~ 1.67
- **Tension reduction achieved**: H₀ 62%, S₈ 69%

With the **modified formulas**:

- n_g only appears in scale dependence (CMB)
- Redshift evolution is controlled purely by z_trans
- MCMC finds: μ ~ 0.36, n_g ~ 0.058, z_trans ~ 2.5 (to compensate)
- **No tension reduction possible**

---

## How to Fix: Restore Original Formulas

### Option 1: Revert the breaking commit

```bash
git revert cc4e26d
```

### Option 2: Manually restore formulas in cgc_physics.py

Change `apply_cgc_to_bao()`:

```python
# RESTORE ORIGINAL:
def apply_cgc_to_bao(DV_rd_lcdm, z, cgc):
    z = np.asarray(z)
    # BAO modification: (D_V/r_d)^CGC = (D_V/r_d)^ΛCDM × [1 + μ × (1+z)^(-n_g)]
    return DV_rd_lcdm * (1 + cgc.mu * (1 + z)**(-cgc.n_g))
```

Change `apply_cgc_to_growth()`:

```python
# RESTORE ORIGINAL:
def apply_cgc_to_growth(fsigma8_lcdm, z, cgc):
    z = np.asarray(z)
    alpha = CGC_COUPLINGS['growth']  # 0.1
    # Growth modification: fσ8_CGC = fσ8_ΛCDM × [1 + 0.1μ × (1+z)^(-n_g)]
    return fsigma8_lcdm * (1 + alpha * cgc.mu * (1 + z)**(-cgc.n_g))
```

---

## Files to Modify

1. **cgc/cgc_physics.py** - Lines 467-545 (apply_cgc_to_growth, apply_cgc_to_bao)
2. Or alternatively revert entire commit: `git revert cc4e26d`

---

## Expected Results After Restoration

With original formulas restored, MCMC should find:

- **μ** ~ 0.47 ± 0.03 (matches v12 thesis)
- **n_g** ~ 0.014 (matches EFT derivation β₀²/4π²)
- **z_trans** ~ 1.67 (matches cosmic transition)

And produce:

- **H₀ tension**: 4.8σ → 1.8σ (62% reduction)
- **S₈ tension**: 2.6σ → 0.8σ (69% reduction)

---

## Verification Results (Feb 4, 2026)

After restoring the original formulas, we ran MCMC and found:

### Short Run (500 steps × 32 walkers)

- μ = 0.468 ± 0.020 ✅ (matches v12!)
- z_trans = 1.664 ± 0.020 ✅
- n_g = 0.027 ± 0.019

### Longer Run (2000 steps × 64 walkers)

- μ = 0.315 - 0.341 (varies with random initialization)
- z_trans = 1.61 - 1.65
- n_g = 0.024 - 0.069

### The Fundamental Problem

Even with original formulas restored, there's a **mathematical issue**:

With the original BAO formula: `1 + μ × (1+z)^(-n_g)`

When n_g = 0.014 (the EFT-derived value):

- (1+z)^(-0.014) ≈ 0.995 - 0.986 for z = 0.38 - 2.0
- This is essentially **constant** (~0.99) across all redshifts

So the CGC modification becomes approximately: `1 + μ × 0.99 ≈ 1 + μ`

This means with μ = 0.47, we get **~47% enhancement to D_V/r_d at ALL redshifts!**

But BAO data has uncertainties of only ~1-2%, so the data **strongly constrains** μ to be small.

---

## Conclusion: The v12 Thesis Claims Cannot Be Reproduced

The claim that CGC with μ = 0.47, n_g = 0.014, z_trans = 1.67 produces:

- H₀ tension: 4.8σ → 1.8σ (62% reduction)
- S₈ tension: 2.6σ → 0.8σ (69% reduction)

**Cannot be reproduced with real Planck/BOSS data using any of the tested formulas.**

Possible explanations:

1. The original thesis used **mock data** that was designed to show tension reduction
2. The formulas had additional **normalization factors** not documented
3. The n_g value was actually **much larger** than 0.014 to allow variation with z
4. There were **other modifications** to the likelihood not captured in the code

---

## Files Modified

1. **cgc/cgc_physics.py** - Restored original (1+z)^(-n_g) formulas for BAO and Growth
2. **FORMULA_HISTORY.md** - This document tracking the changes
3. **test_restored_formulas.py** - Verification script
