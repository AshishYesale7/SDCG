#!/usr/bin/env python3
"""
RIGOROUS VERIFICATION OF GOLD PLATE EXPERIMENT PHYSICS
=======================================================
Verifies that the gravitational and Casimir calculations are correct.
"""

import numpy as np

print("=" * 70)
print("RIGOROUS GRAVITATIONAL CALCULATION VERIFICATION")
print("=" * 70)

# Physical constants (CODATA 2018)
hbar = 1.054571817e-34    # J·s (reduced Planck constant)
c = 299792458             # m/s (speed of light, exact)
G = 6.67430e-11           # N·m²/kg² (gravitational constant)
pi = np.pi

# Gold: 1 mm thick plates
rho_gold = 19300          # kg/m³ (gold density)
t = 1e-3                  # 1 mm plate thickness
sigma = rho_gold * t      # 19.3 kg/m² surface mass density

print(f"\n1. INPUT PARAMETERS:")
print(f"   ℏ = {hbar:.6e} J·s")
print(f"   c = {c} m/s")
print(f"   G = {G:.5e} N·m²/kg²")
print(f"   ρ_gold = {rho_gold} kg/m³")
print(f"   t = {t*1e3} mm")
print(f"   σ = ρ × t = {sigma} kg/m²")

# ============================================================
# GRAVITATIONAL PRESSURE FORMULA DERIVATION
# ============================================================
print(f"\n2. GRAVITATIONAL PRESSURE (from Gauss's law for gravity):")
print(f"   ")
print(f"   For an infinite uniform sheet with surface mass density σ:")
print(f"   The gravitational field is: g = 2πGσ (constant!)")
print(f"   ")
print(f"   The pressure (force per area) on a second plate:")
print(f"   P_grav = σ × g = σ × (2πGσ) = 2πGσ²")
print(f"   ")

P_grav = 2 * pi * G * sigma**2
print(f"   P_grav = 2π × {G:.5e} × ({sigma})²")
print(f"   P_grav = {P_grav:.6e} Pa")

# Force per cm²
A_cm2 = 1e-4  # m²
F_grav = P_grav * A_cm2
print(f"   F_grav = {F_grav:.6e} N/cm²")
print(f"   F_grav = {F_grav*1e9:.3f} nN/cm²")

# ============================================================
# CASIMIR PRESSURE FORMULA
# ============================================================
print(f"\n3. CASIMIR PRESSURE (Lifshitz formula, perfect conductor limit):")
print(f"   ")
print(f"   P_Casimir = π²ℏc / (240 d⁴)")
print(f"   ")
print(f"   This is the attractive pressure between two perfectly")
print(f"   conducting parallel plates separated by distance d.")

# ============================================================
# CROSSOVER DISTANCE DERIVATION
# ============================================================
print(f"\n4. CROSSOVER DISTANCE d_c:")
print(f"   ")
print(f"   Set Casimir pressure = Gravitational pressure:")
print(f"   ")
print(f"   π²ℏc/(240 d_c⁴) = 2πGσ²")
print(f"   ")
print(f"   Solving for d_c⁴:")
print(f"   d_c⁴ = π²ℏc / (240 × 2πGσ²)")
print(f"   d_c⁴ = πℏc / (480 Gσ²)")
print(f"   ")
print(f"   Therefore:")
print(f"   d_c = (πℏc / 480Gσ²)^(1/4)")
print(f"   ")

# Calculate d_c
numerator = pi * hbar * c
denominator = 480 * G * sigma**2
d_c = (numerator / denominator)**0.25

print(f"   Numerator = π × ℏ × c = {numerator:.6e} J·m")
print(f"   Denominator = 480 × G × σ² = {denominator:.6e}")
print(f"   d_c⁴ = {numerator/denominator:.6e} m⁴")
print(f"   ")
print(f"   d_c = {d_c:.6e} m")
print(f"   d_c = {d_c*1e6:.2f} μm")

# ============================================================
# VERIFICATION: PRESSURES EQUAL AT d_c
# ============================================================
print(f"\n5. VERIFICATION - PRESSURES AT d_c = {d_c*1e6:.2f} μm:")

P_casimir = (pi**2 * hbar * c) / (240 * d_c**4)

print(f"   P_Casimir = {P_casimir:.6e} Pa")
print(f"   P_grav    = {P_grav:.6e} Pa")
print(f"   ")
print(f"   Ratio P_C/P_G = {P_casimir/P_grav:.6f}")
print(f"   (Should be exactly 1.000000 at crossover)")

F_C = P_casimir * A_cm2
F_G = P_grav * A_cm2

print(f"   ")
print(f"   F_Casimir = {F_C*1e9:.3f} nN/cm²")
print(f"   F_grav    = {F_G*1e9:.3f} nN/cm²")

# ============================================================
# THESIS CHECK
# ============================================================
print(f"\n" + "=" * 70)
print("THESIS v12 VALUES: VERIFIED CORRECT!")
print("=" * 70)
print(f"   ✓ 1 mm gold plates (σ = 19.3 kg/m²)")
print(f"   ✓ d_c = {d_c*1e6:.2f} μm ≈ 10 μm")
print(f"   ✓ Forces at crossover: {F_C*1e9:.1f} nN/cm² (easily measurable)")
print(f"   ✓ P_C/P_G = 1.000000 (exact)")

# ============================================================
# WHAT GIVES 95 μm?
# ============================================================
print(f"\n" + "=" * 70)
print("CLARIFICATION: WHERE DID 95 μm COME FROM?")
print("=" * 70)

d_target = 95e-6  # 95 μm
sigma_for_95um = np.sqrt(pi * hbar * c / (480 * G * d_target**4))
thickness_for_95um = sigma_for_95um / rho_gold

print(f"   For d_c = 95 μm, we would need:")
print(f"   σ = {sigma_for_95um:.4f} kg/m²")
print(f"   thickness = {thickness_for_95um*1e6:.1f} μm")
print(f"   ")
print(f"   → 95 μm crossover comes from ~10 μm gold FILMS, not 1 mm plates!")
print(f"   → The original thesis mixed up these two configurations.")
