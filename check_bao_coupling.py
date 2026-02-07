#!/usr/bin/env python3
"""Check BAO coupling inconsistency — why μ settles at 0.04 instead of 0.15"""
import numpy as np

mu_current = 0.04
n_g = 0.0125
z_bao = np.array([0.38, 0.51, 0.61])

print("=== CURRENT: BAO coupling = 1.0 (no suppression) ===")
print(f"mu = {mu_current}")
boost = mu_current * (1 + z_bao)**(-n_g)
for z, b in zip(z_bao, boost):
    print(f"  z={z}: BAO boost = {b*100:.2f}% (data ~1.5% precise)")
H0 = 67.36 * (1 + 0.31 * mu_current)
print(f"  H0_eff = {H0:.2f} km/s/Mpc (barely touches H0 tension)")

print("\n=== IF BAO COUPLING = 0.31 (same as H0): ===")
mu_est = 0.15
boost2 = 0.31 * mu_est * (1 + z_bao)**(-n_g)
for z, b in zip(z_bao, boost2):
    print(f"  z={z}: BAO boost = {b*100:.2f}% (reasonable for data!)")
H0_2 = 67.36 * (1 + 0.31 * mu_est)
print(f"  H0_eff = {H0_2:.2f} km/s/Mpc (resolves H0 tension!)")

print("\n=== WHAT mu IS NEEDED FOR H0 = 70.5? ===")
mu_needed = (70.5 / 67.36 - 1) / 0.31
print(f"mu_needed = {mu_needed:.3f}")
boost_10 = mu_needed * (1 + 0.38)**(-n_g) * 100
boost_031 = 0.31 * mu_needed * (1 + 0.38)**(-n_g) * 100
print(f"BAO boost with coupling=1.0: {boost_10:.1f}% -- TOO BIG for 1.5% data!")
print(f"BAO boost with coupling=0.31: {boost_031:.1f}% -- fits within data precision!")

print("\n=== SUMMARY ===")
print("The BAO formula uses mu directly (coupling=1.0)")
print("But H0 uses 0.31*mu, growth uses 0.1*mu, SNe uses 0.5*mu")
print("BAO is the TIGHTEST constraint, forcing mu down to ~0.04")
print("If BAO had a physically consistent coupling (~0.31),")
print("mu would be free to reach ~0.15 and fully resolve H0 tension")
