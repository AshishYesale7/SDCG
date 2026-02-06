# SDCG DERIVATIONS AUDIT

## Comprehensive Formula Verification and Honest Assessment

This document provides a step-by-step verification of all SDCG formulas,
identifying what is rigorously derived versus phenomenological.

---

## 1. β₀ (SCALAR-MATTER COUPLING)

### Status: **PHENOMENOLOGICAL ESTIMATE** (not rigorous QFT)

### Claimed Derivation (Page 7-8):
```
Step 1: β₀² = N_c·y_t²×(m_t/v)² = 3×0.99×0.494 = 1.47 → β₀ ≈ 1.21
Step 2: Add factor of 2... β₀² = (N_c/2)·y_t²·(m_t²/v²) = 0.74
Step 3: Actually, β₀² = m_t²/v² = 0.49 → β₀ = 0.70
```

### Issues:
1. **Factor of 2**: Appears without QFT justification
2. **Color factor N_c=3**: Dropped without explanation
3. **Missing loops**: Real anomaly needs full 1-loop diagrams

### Honest Status:
β₀ = 0.70 is a **dimensionless O(1) coupling benchmarked from m_t/v**.
This is physically motivated but NOT a rigorous derivation.

### Recommended Framing:
> "β₀ is a dimensionless scalar-matter coupling. Naturalness suggests β₀ ~ O(1),
> and we adopt β₀ = 0.70 as a benchmark motivated by the Standard Model ratio
> m_t/v ≈ 0.70. The exact QFT calculation involves complex renormalization
> group flow; here we treat β₀ as phenomenological."

---

## 2. n_g (SCALE EXPONENT)

### Status: **ROUGH ESTIMATE** (with sign error in original derivation)

### Claimed Derivation (Page 9-10):
```
From RG equation: μ d/dμ G_eff⁻¹ = β₀²/16π²
Integrating: G_eff⁻¹(k) - G_N⁻¹ = (β₀²/16π²) ln(k/k_*)
Inverting: G_eff(k)/G_N = 1/(1 + x)
Approximating: G_eff(k)/G_N ≈ 1 + x  ← SIGN ERROR!
Therefore: n_g = β₀²/4π² = 0.014
```

### Critical Error:
For small x: **1/(1+x) ≈ 1 - x**, NOT 1 + x!

### Numerical Value:
- n_g(EFT) = β₀²/4π² = 0.49/39.48 ≈ 0.0124 (for SCALE dependence k^n_g)
- n_g(MCMC) = 0.14 (for REDSHIFT evolution (1+z)^(-n_g))

### Honest Status:
The EFT value 0.014 is a rough estimate. The code uses n_g = 0.14 fitted
from MCMC, which is **phenomenological**.

---

## 3. μ (GRAVITATIONAL COUPLING)

### Status: **PLAUSIBLE ESTIMATE** (with arbitrary cutoff choices)

### Claimed Derivation (Page 11-13):
```
μ_bare = (β₀²/16π²) × ln(M_Pl/H₀)
       = (0.49/157.9) × 140
       = 0.00310 × 140 = 0.434 ≈ 0.43-0.48
```

### Verification:
- ln(M_Pl/H₀) = ln(2.4×10^60) ≈ 139 ✓
- 16π² = 157.91 ✓
- Arithmetic: 0.49 × 140 / 157.91 = 0.434 ✓

### Issue:
Parameter table says μ_bare = 0.48, derivation gives 0.43. **Inconsistent!**

### Honest Status:
The hierarchy logarithm ln(M_Pl/H₀) ≈ 140 is correct. The overall estimate
is plausible but choice of IR/UV cutoffs is somewhat arbitrary.

### What Code Uses:
- μ_eff (voids) = 0.149 (MCMC best-fit)
- This is μ_bare × screening × redshift-evolution

---

## 4. SCREENING MECHANISM

### Status: **STANDARD PHYSICS** (but exponent α is inconsistent)

### Screening Factor:
```
S(ρ) = 1 / [1 + (ρ/ρ_thresh)^α]
```

### Issue:
- Text sometimes says α = 1 (linear potential)
- Text sometimes says α = 2 (quadratic potential)
- Code uses α = 1

### Physical Basis:
- ρ_thresh = 200 ρ_crit from virial overdensity (18π² ≈ 178)
- This is well-motivated ✓

### Honest Status:
Screening concept is standard chameleon physics. Exponent α depends on
scalar potential, which is **model-dependent**.

---

## 5. z_trans (TRANSITION REDSHIFT)

### Status: **WELL-MOTIVATED** (conceptually sound)

### Derivation:
```
Step 1: Acceleration onset q(z)=0
        Ω_m(1+z_acc)³ = 2Ω_Λ
        z_acc = (2Ω_Λ/Ω_m)^(1/3) - 1 ≈ 0.63

Step 2: Scalar response delay
        Δz ≈ 1.0 (one Hubble time)

Step 3: Total
        z_trans = z_acc + Δz ≈ 1.64
```

### Verification:
- (2×0.685/0.315)^(1/3) = 1.629, so z_acc = 0.629 ✓
- z_trans = 0.63 + 1.0 = 1.63 ≈ 1.64 ✓

### Issue:
Code sometimes uses z_trans = 2.0. **Inconsistent!**

### Honest Status:
This is the most physically motivated derivation. The value 1.64 is
reasonably derived from cosmological evolution.

---

## 6. CROSSOVER DISTANCE (Casimir Experiment)

### Status: **MATHEMATICALLY CORRECT** ✓

### Derivation:
```
Casimir pressure: P_C = π²ℏc / (240 d⁴)
Gravity pressure: P_G = 2πGσ²
Setting P_C = P_G:
d_c = (πℏc / 480Gσ²)^(1/4)
```

### Verification (1mm gold plates, σ = 19.3 kg/m²):
- d_c = 9.54 μm ✓ (matches document's 9.55 μm)

### Honest Status:
This derivation is **rigorous and dimensionally consistent**.

---

## 7. MASTER EQUATION

### Status: **STRUCTURALLY SOUND** (with phenomenological components)

### Formula:
```
G_eff(k,z,ρ) = G_N × [1 + μ × f(k) × g(z) × S(ρ)]

where:
  f(k) = (k/k_*)^n_g           (scale dependence)
  g(z) = ½[1 - tanh((z-z_trans)/σ_z)]  (redshift evolution)
  S(ρ) = 1/(1 + (ρ/ρ_thresh)^α)        (screening)
```

### Dimensional Check:
- All factors are dimensionless ✓
- G_eff has same units as G_N ✓

### Honest Status:
The structure is motivated by scalar-tensor gravity. Parameters
μ, n_g, z_trans are phenomenological.

---

## SUMMARY TABLE

| Parameter | Status | Issue |
|-----------|--------|-------|
| β₀ = 0.70 | PHENOMENOLOGICAL | Not rigorous QFT (missing loops, factors) |
| n_g = 0.014 | ROUGH ESTIMATE | Sign error in derivation |
| n_g = 0.14 | MCMC FIT | Phenomenological |
| μ = 0.43-0.48 | PLAUSIBLE | Arbitrary cutoffs, inconsistent values |
| μ = 0.149 | MCMC FIT | Phenomenological |
| z_trans = 1.64 | WELL-MOTIVATED | Best derivation in the thesis |
| ρ_thresh = 200 | DERIVED | From virial overdensity ✓ |
| α = 1 or 2 | MODEL-DEPENDENT | Inconsistent usage |
| d_c = 9.55 μm | RIGOROUS | Math correct ✓ |

---

## RECOMMENDATIONS

### 1. Frame Parameters Honestly
Replace "EFT-derived" with "phenomenological estimate motivated by..."

### 2. Fix Sign Error
Note that 1/(1+x) ≈ 1-x for small x, not 1+x.

### 3. Unify Values
Use consistent values in text, tables, and code:
- μ = 0.149 (MCMC best-fit)
- n_g = 0.14 (for z-evolution)
- z_trans = 1.64
- α = 1 (pick one)

### 4. Focus on Strengths
The **cosmological predictions** (tension reduction, dwarf rotation, Casimir)
don't depend on whether β₀ comes from perfect QFT or reasonable phenomenology.

---

## CONCLUSION

The SDCG framework is **scientifically honest** when framed as:

> "A phenomenological scalar-tensor modification of gravity with parameters
> motivated by Standard Model physics and constrained by cosmological data.
> The framework makes testable predictions for H₀/S₈ tension reduction,
> void dwarf rotation curves, and laboratory experiments."

This is **more defensible** than claiming rigorous QFT derivations that
have known issues.

---

*Last updated: 2026-02-04*
*Based on comprehensive formula audit*
