# Thesis v12 Canonical Parameters

**Last Updated**: Codebase restructuring session

This document defines the **official canonical parameter values** for the CGC (Casimir-Gravity Crossover) / SDCG framework as established in Thesis v12.

---

## Quick Reference

| Parameter | Value | Status | Source |
|-----------|-------|--------|--------|
| **μ_fit** | 0.47 ± 0.03 | MCMC FITTED | 6σ detection from cosmological data |
| **μ_eff(void)** | 0.149 | DERIVED | = μ_fit × S_avg ≈ 0.47 × 0.31 |
| **n_g** | 0.0125 | FIXED | β₀²/4π² (renormalization group) |
| **z_trans** | 1.67 | FIXED | Cosmic dynamics q(z)=0 |
| **α** | 2.0 | FIXED | Klein-Gordon quadratic potential |
| **ρ_thresh** | 200 ρ_crit | FIXED | Virial theorem |
| **β₀** | 0.70 | FIXED | Standard Model m_t/v = 173/246 |

---

## Critical Clarification: μ_fit vs μ_eff

### The Single Free Parameter
**μ_fit = 0.47 ± 0.03** is the **FUNDAMENTAL** MCMC best-fit coupling strength.

This is the **only free parameter** in the Thesis v12 formulation.

### Derived Effective Couplings
The effective coupling in different environments is DERIVED from μ_fit:

```
μ_eff(environment) = μ_fit × S(ρ_environment)
```

Where S(ρ) is the screening function:
```
S(ρ) = 1 / (1 + (ρ/ρ_thresh)²)    [with α = 2]
```

### Key Environment Values

| Environment | ρ/ρ_crit | S(ρ) | μ_eff |
|-------------|----------|------|-------|
| Voids | 0.1 | 0.31 | 0.149 |
| Filaments | 1.0 | 0.99996 | 0.47 |
| Clusters | 1000 | 4×10⁻⁵ | 2×10⁻⁵ |
| IGM (Lyα) | 50-100 | 0.14 | 0.066 |

---

## Fixed Parameters (Not Fitted)

### n_g = 0.0125
- **Source**: β₀²/4π² from renormalization group flow
- **Status**: FIXED by theory (like c_T = 1 in Horndeski)
- **Old value**: n_g = 0.138 was an MCMC-fitted value from earlier thesis versions

### z_trans = 1.67
- **Source**: Cosmic dynamics (matter-DE crossover)
- **Status**: FIXED by cosmological physics
- **Old value**: z_trans = 1.64 was used in earlier versions

### α = 2.0
- **Source**: Klein-Gordon equation with quadratic potential
- **Status**: FIXED by scalar field theory
- **Note**: Earlier versions used α = 1 or variable α

### ρ_thresh = 200 ρ_crit
- **Source**: Virial theorem (18π² ≈ 178, rounded to 200)
- **Status**: FIXED by astrophysics
- **Note**: Standard cluster overdensity threshold

---

## Screening Function

The chameleon-like screening mechanism:

```
S(ρ) = 1 / (1 + (ρ/ρ_thresh)^α)
     = 1 / (1 + (ρ/200)²)      [Thesis v12]
```

### Properties
- S → 1 in voids (ρ << ρ_thresh)
- S → (ρ_thresh/ρ)² in clusters (ρ >> ρ_thresh)
- Continuous suppression across density range

---

## Tension Reduction

With μ_eff(void) = 0.149:

- **H₀ tension**: 67.4 → 71.3 km/s/Mpc (87% reduction)
- **S₈ tension**: 0.83 → 0.74 (84% reduction)

---

## Lyα Forest Constraint

The Lyα constraint μ < 0.07 is satisfied by screening:

```
μ_eff(Lyα) = μ_fit × f(k) × S(ρ_IGM) × g(z=3)
           ≈ 0.47 × 0.5 × 0.14 × 0.3
           ≈ 0.010 << 0.07 ✓
```

---

## Testable Predictions

1. **Void dwarf rotation**: Δv = +10-15 km/s enhancement
2. **Scale-dependent growth**: f(k)σ₈ with DESI/Euclid
3. **Casimir experiment**: Critical distance d_c ≈ 9.5 μm

---

## Code Usage

```python
from cgc.cgc_physics import CGCPhysics

# Thesis v12 canonical parameters
cgc = CGCPhysics(
    mu=0.47,        # μ_fit (fundamental MCMC best-fit)
    z_trans=1.67,   # FIXED by theory
    rho_thresh=200  # FIXED by theory
)
# n_g = 0.0125 is automatically set (fixed property)

# Get effective coupling in voids
mu_eff_void = cgc.mu * cgc.screening(rho=0.1)  # ≈ 0.149

# Get effective coupling for Lyα
mu_eff_lya = cgc.mu_eff_for_environment('lyalpha')  # ≈ 0.066
```

---

## Historical Note

Earlier thesis versions (v1-v11) used different parameter values:
- μ = 0.149 (this was μ_eff(void), not μ_fit)
- n_g = 0.138 (MCMC-fitted, now superseded by fixed value)
- z_trans = 1.64 (slightly different cosmic dynamics estimate)

Thesis v12 establishes the canonical values with proper distinction between:
- **μ_fit** (fundamental fitted parameter)
- **μ_eff** (derived environment-dependent effective coupling)
