#!/usr/bin/env python3
"""Quick BAO data check."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from simulations.cgc.config import CONSTANTS

z = np.array([0.38, 0.51, 0.61])
DV_rd_data = np.array([8.467, 8.857, 9.445])

h, omega_b, omega_cdm = 0.6736, 0.02237, 0.1200
H0 = h * 100
Omega_m = (omega_b + omega_cdm) / h**2
c = CONSTANTS['c']
r_d = CONSTANTS['r_d_fid']

print(f'Omega_m = {Omega_m:.4f}, r_d_fid = {r_d}, c = {c}')

for zi in z:
    z_int = np.linspace(0, zi, 1000)
    H_int = H0 * np.sqrt(Omega_m * (1+z_int)**3 + (1 - Omega_m))
    D_M = c * np.trapz(1.0/H_int, z_int)
    Hz = H0 * np.sqrt(Omega_m * (1+zi)**3 + (1-Omega_m))
    D_V = (c * zi * D_M**2 / Hz)**(1/3)
    DV_rd = D_V / r_d
    print(f'z={zi}: model DV/rd={DV_rd:.3f}  vs  data DV/rd={DV_rd_data[z==zi][0]:.3f}  (delta = {DV_rd - DV_rd_data[z==zi][0]:.3f})')

print()
print('Published BOSS DR12 (Alam+2017): DV/rd = 10.27, 13.38, 15.29')
print('Data file has: 8.467, 8.857, 9.445')
print('Model predicts:', end=' ')
