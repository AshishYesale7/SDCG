# SDCG FRAMEWORK COMPREHENSIVE AUDIT

## Complete Mathematical and Implementation Verification

**Date:** February 4, 2026  
**Status:** CRITICAL ISSUES IDENTIFIED

---

## PHASE 1: FRAMEWORK ARCHITECTURE MAP

### Equation Hierarchy

| ID | Equation | Purpose | Level | Code Location |
|----|----------|---------|-------|---------------|
| 1 | G_eff/G_N = 1 + μ×f(k)×g(z)×S(ρ) | Master Equation | Final | cgc_physics.py:300 |
| 2 | f(k) = (k/k_pivot)^n_g | Scale dependence | Component | cgc_physics.py:165 |
| 3 | g(z) = exp[-(z-z_trans)²/(2σ_z²)] | Redshift evolution | Component | cgc_physics.py:218 |
| 4 | S(ρ) = 1/(1+(ρ/ρ_thresh)^α) | Screening | Component | cgc_physics.py:270 |
| 5 | E²(z) = Ω_m(1+z)³ + Ω_Λ + Δ_CGC | Modified Friedmann | Observable | cgc_physics.py:350 |
| 6 | fσ8_CGC = fσ8_ΛCDM × [1 + 0.1μ×(1+z)^(-n_g)] | Growth | Observable | cgc_physics.py:500 |
| 7 | (D_V/r_d)_CGC = (D_V/r_d)_ΛCDM × [1 + μ×(1+z)^(-n_g)] | BAO | Observable | cgc_physics.py:560 |

---

## PHASE 2: PARAMETER LINEAGE AUDIT

### β₀ (Scalar-Matter Coupling)

| Step | Description | Value | Status |
|------|-------------|-------|--------|
| Source | SM trace anomaly + top quark | m_t/v = 173/246 | PHENOMENOLOGICAL |
| Derivation | β₀² = (N_c/2)·y_t²·(m_t²/v²) | 0.74 → 0.49 (inconsistent) | ❌ UNJUSTIFIED |
| Final | β₀ = √0.49 | 0.70 | Used in code |
| Code | `BETA_0 = 0.70` | parameters.py:139 | ✓ |

**Issues:**
- Factor of 2 appears without justification
- N_c=3 color factor dropped
- Not a rigorous QFT calculation

---

### n_g (Scale Exponent)

| Step | Description | Value | Status |
|------|-------------|-------|--------|
| EFT Derivation | n_g = β₀²/4π² | 0.49/39.48 = 0.0124 | SIGN ERROR |
| MCMC Fit | For (1+z)^(-n_g) evolution | 0.138 ± 0.014 | Used in code |
| Code | `N_G_FROM_BETA = 0.0124` | parameters.py:156 | ✓ |
| Code | `cgc_n_g: float = 0.138` | cgc_physics.py:117 | ✓ |

**CRITICAL ERROR:**
```
CLAIMED: G_eff/G_N = 1/(1+x) ≈ 1 + x  for small x
CORRECT: G_eff/G_N = 1/(1+x) ≈ 1 - x  for small x

This is a SIGN ERROR in the approximation!
```

**Impact:** The scale dependence direction may be reversed.

---

### μ (Coupling Strength) - THE TWO-μ PROBLEM

| Context | Value | Source | Code Variable |
|---------|-------|--------|---------------|
| μ_bare (QFT) | 0.48 | β₀²×ln(M_Pl/H₀)/16π² | `MU_BARE = 0.48` |
| μ_cosmic (MCMC) | 0.47 | CMB+BAO+SNe fit | `MU_MCMC = 0.47` |
| μ_eff (voids) | 0.149 | Screened effective | `cgc_mu: float = 0.149` |
| μ_Lyα (IGM) | 0.045 | Lyα constraint | `MU_LYALPHA = 0.045` |

**Consistency Check:**
```
μ_eff(void) = μ_bare × S(ρ_void) × g(z)

For voids: ρ_void ~ 0.1 ρ_crit
S(0.1) = 1/(1 + (0.1/200)^2) = 1/(1 + 2.5×10⁻⁷) ≈ 1.0

So: μ_eff ≈ μ_bare × 1.0 × g(z)

At z=0: g(0) = exp[-(0-1.64)²/(2×1.5²)] = exp(-0.60) ≈ 0.55
μ_eff(z=0) = 0.48 × 0.55 ≈ 0.26

This does NOT match 0.149!
```

**RESOLUTION NEEDED:**
- Either μ_bare or g(z) or screening formula needs adjustment
- Current values are internally INCONSISTENT

---

### z_trans (Transition Redshift)

| Source | Value | Status |
|--------|-------|--------|
| Derived (q(z)=0) | z_acc = 0.63 | ✓ Correct |
| + Scalar delay | Δz = 1.0-1.04 | Phenomenological |
| Final derived | z_trans = 1.64-1.67 | ✓ Physically motivated |
| Code | `cgc_z_trans: float = 1.64` | ✓ Consistent |
| Some code | `z_trans = 2.0` | ❌ INCONSISTENT |

**Issue:** Some old code uses z_trans = 2.0, need to unify to 1.64.

---

### α (Screening Exponent)

| Source | Value | Status |
|--------|-------|--------|
| Thesis text | α = 2 | Chameleon m_eff² ~ ρ |
| Code constant | `SCREENING_ALPHA = 2` | cgc_physics.py:40 |
| Code formula | `(rho/rho_thresh)**SCREENING_ALPHA` | ✓ Uses α=2 |

**Status:** ✓ CONSISTENT (α = 2 everywhere in current code)

---

### ρ_thresh (Screening Threshold)

| Source | Value | Status |
|--------|-------|--------|
| Virial theorem | 18π² ≈ 178 | ✓ Standard |
| Rounded | 200 ρ_crit | ✓ Reasonable |
| Code | `rho_thresh: float = 200.0` | ✓ Consistent |

**Status:** ✓ DERIVED and CONSISTENT

---

## PHASE 3: CODE-MATH SYNCHRONIZATION CHECK

### Issue 1: BAO/Growth Formula Discrepancy

**Reference Document (CGC_EQUATIONS_REFERENCE.txt):**
```
NEW BAO FORMULA (CORRECT): [1 + μ × exp(-z/z_trans)]
```

**Actual Code (cgc_physics.py:560):**
```python
return DV_rd_lcdm * (1 + cgc.mu * (1 + z)**(-cgc.n_g))
```

**STATUS: ❌ CODE USES OLD FORMULA**

The code still uses `(1+z)^(-n_g)` but the reference says it should be `exp(-z/z_trans)`.

---

### Issue 2: Growth Formula Same Problem

**Reference Document:**
```
NEW GROWTH FORMULA (CORRECT): [1 + 0.1μ × exp(-z/z_trans)]
```

**Actual Code (cgc_physics.py:500):**
```python
return fsigma8_lcdm * (1 + alpha * cgc.mu * (1 + z)**(-cgc.n_g))
```

**STATUS: ❌ CODE USES OLD FORMULA**

---

### Issue 3: Redshift Evolution Function Mismatch

**Master Equation uses:** `g(z) = exp[-(z-z_trans)²/(2σ_z²)]` (Gaussian)

**BAO/Growth use:** `(1+z)^(-n_g)` (Power law)

**These are DIFFERENT FUNCTIONS!**

```
g(z=0) with Gaussian:     exp[-(-1.64)²/(2×1.5²)] = 0.55
g(z=0) with power law:    (1+0)^(-0.138) = 1.0

These differ by 45%!
```

---

## PHASE 4: MASTER EQUATION DECOMPOSITION

### Component Analysis

| Component | Formula | Code Function | Match? |
|-----------|---------|---------------|--------|
| f(k) | (k/k_pivot)^n_g | `scale_dependence()` | ✓ |
| g(z) Gaussian | exp[-(z-z_trans)²/(2σ²)] | `redshift_evolution()` | ✓ |
| g(z) BAO | (1+z)^(-n_g) | `apply_cgc_to_bao()` | ❌ WRONG |
| S(ρ) | 1/(1+(ρ/ρ_thresh)^α) | `screening_function()` | ✓ α=2 |
| G_eff/G_N | 1 + μ×F | `Geff_over_G()` | ✓ |

---

## PHASE 5: PARAMETER VALUE CONSISTENCY MATRIX

| Parameter | Text Value | Equation | Code Value | Status |
|-----------|------------|----------|------------|--------|
| β₀ | 0.70 | √(m_t/v)² | 0.70 | ✓ (phenomenological) |
| n_g (EFT) | 0.014 | β₀²/4π² | 0.0124 | ⚠️ Sign error in derivation |
| n_g (fit) | 0.138 | MCMC | 0.138 | ✓ |
| μ_bare | 0.48 | QFT | 0.48 | ✓ |
| μ_eff | 0.149 | Screened | 0.149 | ⚠️ Inconsistent with μ_bare |
| z_trans | 1.64 | q(z)=0 + delay | 1.64 | ✓ |
| α | 2 | Klein-Gordon | 2 | ✓ |
| ρ_thresh | 200 | 18π² | 200 | ✓ |

---

## PHASE 6: CRITICAL ERRORS SUMMARY

### 🔴 CRITICAL (Must Fix)

1. **BAO/Growth Formula Mismatch**
   - Reference says: `exp(-z/z_trans)`
   - Code uses: `(1+z)^(-n_g)`
   - Impact: Different z-evolution, affects tension reduction

2. **Two-μ Problem Unresolved**
   - μ_bare = 0.48 does not reduce to μ_eff = 0.149 with stated screening
   - Need to clarify which μ is "the" SDCG coupling

3. **n_g Derivation Sign Error**
   - Small-x approximation: 1/(1+x) ≈ 1-x, not 1+x
   - May flip scale dependence direction

### 🟡 MODERATE (Should Fix)

4. **Inconsistent g(z) Functions**
   - Master equation uses Gaussian
   - Observables use power law
   - Need to unify or justify difference

5. **β₀ = 0.70 is Phenomenological**
   - Claims "derived from QFT" are overstated
   - Should honestly present as benchmark

### 🟢 MINOR (Cosmetic)

6. **Some old code has z_trans = 2.0**
   - Should unify to 1.64 everywhere

---

## PHASE 7: SYNCHRONIZATION PLAN

### Step 1: Fix BAO/Growth Formulas (Priority: HIGH)

**Option A: Use Reference Formula (Recommended)**
```python
# cgc_physics.py:560
def apply_cgc_to_bao(DV_rd_lcdm, z, cgc):
    z = np.asarray(z)
    # CORRECTED: Use z_trans-based exponential, not (1+z)^(-n_g)
    return DV_rd_lcdm * (1 + cgc.mu * np.exp(-z / cgc.z_trans))

# cgc_physics.py:500
def apply_cgc_to_growth(fsigma8_lcdm, z, cgc):
    z = np.asarray(z)
    alpha = CGC_COUPLINGS['growth']  # 0.1
    # CORRECTED: Use z_trans-based exponential
    return fsigma8_lcdm * (1 + alpha * cgc.mu * np.exp(-z / cgc.z_trans))
```

**Option B: Document Why (1+z)^(-n_g) is Used**
- If keeping power law, document that it's PHENOMENOLOGICAL
- Note it differs from Master Equation's Gaussian

### Step 2: Clarify μ Hierarchy (Priority: HIGH)

Add clear documentation:
```
μ VALUES IN SDCG:
1. μ_bare = 0.48 (QFT, unscreened)
2. μ_cosmic = 0.47 (MCMC unconstrained)  
3. μ_eff = 0.149 (MCMC with Lyα, used in code)
4. μ_Lyα = 0.045 (Lyα conservative bound)

CODE USES: μ_eff = 0.149 as the default cgc_mu parameter
This is the EFFECTIVE coupling in cosmological voids.
```

### Step 3: Fix n_g Sign Error Note (Priority: MEDIUM)

Already done in previous update - document the sign error.

### Step 4: Unify z_trans = 1.64 (Priority: LOW)

Search and replace any remaining `z_trans = 2.0` to 1.64.

---

## PHYSICAL PREDICTION VERIFICATION

### Prediction 1: H₀ Tension Reduction

**Formula:** H₀_CGC = H₀_Planck × (1 + α_h0 × μ)
```
α_h0 = 0.31
μ = 0.149
H₀_Planck = 67.4 km/s/Mpc

H₀_CGC = 67.4 × (1 + 0.31 × 0.149) = 67.4 × 1.046 = 70.5 km/s/Mpc
```

**Verification:** ✓ MATCHES claimed 70.5 km/s/Mpc

### Prediction 2: S₈ Tension Reduction

**Formula:** S₈_CGC = S₈_Planck × (1 + α_s8 × μ)
```
α_s8 = -0.40
μ = 0.149
S₈_Planck = 0.83

S₈_CGC = 0.83 × (1 - 0.40 × 0.149) = 0.83 × 0.940 = 0.78
```

**Verification:** ✓ MATCHES claimed S₈ = 0.78

### Prediction 3: Dwarf Galaxy Velocity

**Formula:** Δv/v ≈ ½μ(S_void - S_cluster)
```
μ = 0.149, S_void ≈ 1, S_cluster ≈ 0
Δv/v = 0.5 × 0.149 × 1 = 0.075

For v = 80 km/s: Δv = 6 km/s
```

**But thesis claims 12 km/s!**
```
To get 12 km/s: Δv/v = 12/80 = 0.15
Requires: μ = 0.30 (which is μ_eff × 2)
```

**Issue:** Dwarf prediction uses different μ than cosmological?

---

## FINAL RECOMMENDATIONS

### Immediate Fixes Required:

1. **Update BAO/Growth formulas** to use `exp(-z/z_trans)` OR document why power law is kept

2. **Create PARAMETER_DEFINITIONS.md** clearly stating which μ value is used where

3. **Run consistency test** to verify tension reduction still works after formula fix

4. **Audit thesis PDF** to ensure equations match code exactly

### Documentation Needed:

1. Honest statement that β₀, n_g derivations are phenomenological
2. Clear μ hierarchy explanation
3. Formula choice justification (Gaussian vs power law vs exponential)

### Before Publication:

1. All equations in thesis must match code exactly
2. All parameter values must be consistent
3. Sign error in n_g derivation must be noted

---

*Audit completed: February 4, 2026*
*Status: CRITICAL ISSUES REQUIRE ATTENTION*
