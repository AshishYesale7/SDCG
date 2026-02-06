#!/usr/bin/env python3
"""
Check force magnitudes for gold plate experiment
"""
import numpy as np

hbar = 1.054571817e-34
c = 299792458
G = 6.67430e-11
pi = np.pi

sigma = 19.3  # kg/m2
d_c = 9.55e-6  # m

# Pressure at crossover
P = 2 * pi * G * sigma**2
print(f"Pressure at crossover: {P:.3e} Pa")

print("\nForce per different plate areas:")
print(f"  Per 1 cm2 (1e-4 m2):   {P * 1e-4 * 1e12:.1f} pN")
print(f"  Per 10 cm2 (1e-3 m2):  {P * 1e-3 * 1e12:.1f} pN = {P * 1e-3 * 1e9:.3f} nN")
print(f"  Per 100 cm2 (1e-2 m2): {P * 1e-2 * 1e12:.1f} pN = {P * 1e-2 * 1e9:.2f} nN")

print("\nTypical Casimir experiments:")
print("  - Use sphere-plate geometry (proximity force approx)")
print("  - Measure forces in pN to nN range")
print("  - Our 16 pN/cm2 is measurable with modern AFM")
