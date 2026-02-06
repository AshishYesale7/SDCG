#!/usr/bin/env python3
"""How z_trans affects the CGC model"""
import numpy as np

print("=" * 65)
print("HOW z_trans AFFECTS THE CGC MODEL")
print("=" * 65)

def f_z(z, z_trans):
    """Redshift suppression factor"""
    return 1 / (1 + (z / z_trans)**2)

z_values = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

print("\nRedshift evolution f(z) = 1/(1 + (z/z_trans)^2)")
print("-" * 65)
print(f"{'z':>6} | z_trans=1.34 | z_trans=1.67 | z_trans=2.00")
print("-" * 65)

for z in z_values:
    f_134 = f_z(z, 1.34)
    f_167 = f_z(z, 1.67)
    f_200 = f_z(z, 2.00)
    print(f"{z:>6.1f} | {f_134:>11.3f} | {f_167:>11.3f} | {f_200:>11.3f}")

print("\n" + "=" * 65)
print("WHAT z_trans CONTROLS:")
print("=" * 65)
print("""
z_trans = 1.67 means:
  - At z < 1.67: CGC modification is ACTIVE (f(z) > 0.5)
  - At z > 1.67: CGC modification is SUPPRESSED
  - At z = 3 (Lya): f(3) = 1/(1 + (3/1.67)^2) = 0.24 (76% suppressed)
  - At z = 1100 (CMB): f(1100) ~ 0 (completely suppressed)

Physical meaning:
  - CGC turns on when universe transitions from deceleration to acceleration
  - q(z) = 0 occurs at z ~ 0.63
  - Scalar field takes ~1 Hubble time to respond -> z_trans = 0.63 + 1.04 = 1.67
""")

print("=" * 65)
print("EFFECT ON Lya CONSTRAINT:")
print("=" * 65)

z_lya = 3.0
for z_trans in [1.34, 1.67, 2.00]:
    g_z = 1 / np.sqrt(1 + (z_lya / z_trans)**2)
    mu_lya = 0.148 * 0.7 * g_z * 0.22
    print(f"z_trans = {z_trans}: g(z=3) = {g_z:.3f} -> mu_eff(Lya) = {mu_lya:.4f}")

print("""
CONCLUSION:
-----------
- z_trans = 1.67 is THEORETICALLY CORRECT (from q=0 + delay)
- Changing it would shift when CGC activates
- Lower z_trans -> CGC turns on later -> LESS tension reduction
- Higher z_trans -> CGC turns on earlier -> MORE tension reduction
                                          (but may violate high-z constraints)

z_trans = 1.67 is the SWEET SPOT from theory - don't change it!
""")
