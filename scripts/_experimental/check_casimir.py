#!/usr/bin/env python3
"""
Check Casimir force values in thesis
"""
import numpy as np

hbar = 1.054571817e-34
c = 299792458
pi = np.pi

d = 10e-6  # 10 um

# Casimir pressure (perfect conductor limit)
P_C = pi**2 * hbar * c / (240 * d**4)
print(f"Casimir pressure at d = 10 um: {P_C:.3e} Pa")

# Force for 1 cm^2 = 1e-4 m^2
A = 1e-4  # m^2
F = P_C * A
print(f"Force for 1 cm^2: {F:.3e} N")
print(f"                = {F*1e9:.4f} nN")
print(f"                = {F*1e12:.1f} pN")

print()
print("Thesis claims: F_Casimir ~ 1.3 nN at 10 um")
print("Actual value:  F_Casimir ~ 0.013 nN = 13 pN at 10 um")
print()
print("This is a factor of 100 error!")

# What distance gives 1.3 nN?
F_target = 1.3e-9
d_correct = (pi**2 * hbar * c * A / (240 * F_target))**0.25
print(f"\nFor F = 1.3 nN, we need d = {d_correct*1e6:.2f} um")
print("So 1.3 nN is correct for ~1.8 um, not 10 um.")

# Or what area gives 1.3 nN at 10 um?
A_needed = F_target / P_C
print(f"\nOr for d = 10 um and F = 1.3 nN, we need area = {A_needed*1e4:.0f} cm^2")
print(f"                                           = {np.sqrt(A_needed)*100:.1f} cm x {np.sqrt(A_needed)*100:.1f} cm plate")
