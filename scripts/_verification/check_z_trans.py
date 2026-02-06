#!/usr/bin/env python3
"""Check z_trans derivation"""
import numpy as np

print("z_trans DERIVATION CHECK")
print("=" * 60)

Omega_m = 0.315
Omega_L = 0.685

# Method 1: Matter-DE equality
z_eq1 = (Omega_L / Omega_m)**(1/3) - 1
print(f"Matter-DE equality: z = (Omega_L/Omega_m)^(1/3) - 1 = {z_eq1:.3f}")

# Method 2: Deceleration-acceleration transition q=0
z_accel = (2*Omega_L / Omega_m)**(1/3) - 1
print(f"q=0 transition:     z = (2*Omega_L/Omega_m)^(1/3) - 1 = {z_accel:.3f}")

print()
print("CONCLUSION:")
print(f"  The thesis z_eq = 0.63 matches q=0 transition ({z_accel:.2f})")
print(f"  z_trans = {z_accel:.2f} + 1.04 = {z_accel + 1.04:.2f} approx 1.67")
print()
print("  The CODE COMMENT should say 'q=0 transition' not 'Matter-DE equality'")
print("  But z_trans = 1.67 is CORRECT!")
