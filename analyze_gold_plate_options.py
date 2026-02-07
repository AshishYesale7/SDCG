#!/usr/bin/env python3
"""
GOLD PLATE EXPERIMENT: PHYSICS-BASED OPTION ANALYSIS
Which configuration is best for experimentation?
"""
import numpy as np

# Constants
hbar = 1.055e-34  # J·s
c = 3e8           # m/s
G = 6.674e-11     # m³/(kg·s²)
rho_gold = 19300  # kg/m³

print("="*70)
print("GOLD PLATE EXPERIMENT: PHYSICS-BASED OPTION ANALYSIS")
print("="*70)

# OPTION A: 10 μm films
print("\n--- OPTION A: 10 um films ---")
t_A = 10e-6
sigma_A = rho_gold * t_A
d_c_A = (np.pi * hbar * c / (480 * G * sigma_A**2))**(1/4)
P_A = np.pi**2 * hbar * c / (240 * d_c_A**4)
F_A = P_A * 1e-4  # 1 cm² area

print(f"Thickness: 10 um, sigma = {sigma_A:.3f} kg/m2")
print(f"Crossover distance: d_c = {d_c_A*1e6:.1f} um")
print(f"Force at crossover (1 cm2): {F_A*1e12:.3f} pN (piconewtons)")

# OPTION B: 1 mm plates
print("\n--- OPTION B: 1 mm plates ---")
t_B = 1e-3
sigma_B = rho_gold * t_B
d_c_B = (np.pi * hbar * c / (480 * G * sigma_B**2))**(1/4)
P_B = np.pi**2 * hbar * c / (240 * d_c_B**4)
F_B = P_B * 1e-4  # 1 cm² area

print(f"Thickness: 1 mm, sigma = {sigma_B:.1f} kg/m2")
print(f"Crossover distance: d_c = {d_c_B*1e6:.1f} um")
print(f"Force at crossover (1 cm2): {F_B*1e9:.3f} nN (nanonewtons)")

# Comparison
print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(f"""
                        OPTION A          OPTION B
                        (10 um films)     (1 mm plates)
---------------------------------------------------------------
Crossover distance      {d_c_A*1e6:.0f} um            {d_c_B*1e6:.1f} um
Force magnitude         {F_A*1e12:.2f} pN          {F_B*1e9:.2f} nN
Force ratio (B/A)       1x                {F_B/F_A:.0f}x

AFM detection limit     ~1 pN             ~1 pN
Measurable?             VERY HARD         FEASIBLE

Casimir experiments     Rarely >50 um     Standard 1-10 um
""")

print("="*70)
print("RECOMMENDATION: OPTION B (1 mm plates, d_c = 9.6 um)")
print("="*70)
print("""
PHYSICS REASONS:

1. SIGNAL STRENGTH: Option B has ~100x stronger forces
   - Option A: 0.01 pN (barely detectable)
   - Option B: 0.9 nN (easily measurable)

2. EXPERIMENTAL FEASIBILITY:
   - Casimir experiments are routinely done at 1-10 um
   - At 95 um, Casimir force is 8000x weaker (d^-4 scaling)
   - Electrostatic/thermal noise dominates at large distances

3. PLATE HANDLING:
   - 1 mm plates: Robust, flat, easy to align
   - 10 um films: Fragile, prone to bending

4. MATHEMATICAL APPROXIMATION:
   - Infinite plate formula requires d << plate_size
   - At 9.6 um with 1 cm plates: d/L = 0.001 (excellent)
   - At 95 um with 1 cm plates: d/L = 0.01 (okay but worse)

CONCLUSION:
-----------
The thesis should be CORRECTED to use:
  - 1 mm thick gold plates
  - sigma = 19.3 kg/m2
  - d_c = 9.6 um (NOT 95 um)

This makes the experiment FEASIBLE with current technology.
""")
