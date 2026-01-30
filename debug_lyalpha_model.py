#!/usr/bin/env python3
"""Debug script to calibrate Lyman-alpha model."""
import numpy as np

# Load actual data
data = np.loadtxt('data/lyalpha/eboss_lyalpha_REAL.dat', comments='#')
z_arr, k_arr, P_F_arr = data[:,0], data[:,1], data[:,2]
err_stat, err_sys = data[:,3], data[:,4]
err_total = np.sqrt(err_stat**2 + err_sys**2)

print("=" * 60)
print("LYMAN-ALPHA DATA ANALYSIS")
print("=" * 60)

print("\nDATA at k=0.01 s/km:")
for z_val in np.unique(z_arr):
    mask = (z_arr == z_val) & (k_arr == 0.01)
    if np.any(mask):
        print(f"  z={z_val:.1f}: P_F={P_F_arr[mask][0]:.4f}")

# Model calibration - use z=3 as pivot
z_pivot = 3.0
mask_pivot = (z_arr == 3.0) & (k_arr == 0.01)
P_pivot = P_F_arr[mask_pivot][0]
print(f"\nUsing P_pivot(z=3, k=0.01) = {P_pivot:.4f}")

# Test different z evolution exponents
print("\n" + "=" * 60)
print("Finding best z-evolution exponent")
print("=" * 60)

best_alpha = 2.0
best_chi2 = 1e10

for alpha in np.arange(1.0, 4.0, 0.1):
    total_chi2 = 0
    for z_val in np.unique(z_arr):
        mask = (z_arr == z_val) & (k_arr == 0.01)
        if np.any(mask):
            P_data = P_F_arr[mask][0]
            err = err_total[mask][0]
            z_evo = ((1 + z_pivot) / (1 + z_val))**alpha
            P_model = P_pivot * z_evo
            total_chi2 += ((P_model - P_data) / err)**2
    
    if total_chi2 < best_chi2:
        best_chi2 = total_chi2
        best_alpha = alpha

print(f"Best alpha: {best_alpha:.1f} (chi2 = {best_chi2:.2f})")

# Now test k-dependence
print("\n" + "=" * 60)
print("Data at z=3.0 (all k values):")
print("=" * 60)
mask_z3 = z_arr == 3.0
k_z3 = k_arr[mask_z3]
P_z3 = P_F_arr[mask_z3]

print("k [s/km]  P_F        log(k)    log(P_F)  n_eff")
k_pivot_val = 0.01
P_pivot_val = P_z3[k_z3 == k_pivot_val][0]

for i, (k, P) in enumerate(zip(k_z3, P_z3)):
    if k > 0.001:  # Skip first point for log ratio
        n_eff = np.log(P / P_pivot_val) / np.log(k / k_pivot_val)
        print(f"{k:.4f}    {P:.4f}     {np.log10(k):.2f}     {np.log10(P):.2f}     {n_eff:.2f}")

# Fit spectral index using log-log slope over broader range
mask_fit = z_arr == 3.0
log_k = np.log10(k_arr[mask_fit])
log_P = np.log10(P_F_arr[mask_fit])
coeffs = np.polyfit(log_k, log_P, 1)  # Linear fit in log-log space
n_F_fit = coeffs[0]
P_norm = 10**coeffs[1]  # Normalization at k=1
print(f"\nFitted spectral index n_F = {n_F_fit:.3f}")
print(f"Intercept (P at k=1): {P_norm:.6f}")

# Better approach: fit P_F = A * k^n for k < 0.05
# At k=0.01: P_F = A * 0.01^n
# So A = P_F(k=0.01) / 0.01^n

# Full model test with best parameters
print("\n" + "=" * 60)
print("Full model comparison")
print("=" * 60)

alpha = best_alpha
n_F = n_F_fit
k_cutoff = 0.08  # s/km - thermal cutoff

print(f"Model: P_F(k,z) = {P_pivot:.4f} * ((1+{z_pivot})/(1+z))^{alpha:.1f} * (k/0.01)^{n_F:.2f} * exp(-(k/{k_cutoff})^1.5)")
print()

total_chi2 = 0
n_pts = 0
print("z     k        P_model   P_data    (P_m-P_d)/err")
for z_val in [2.2, 3.0, 4.2]:
    mask = z_arr == z_val
    for k, P_data, err in zip(k_arr[mask], P_F_arr[mask], err_total[mask]):
        z_evo = ((1 + z_pivot) / (1 + z_val))**alpha
        k_term = (k / 0.01)**n_F
        cutoff = np.exp(-(k / k_cutoff)**1.5)
        P_model = P_pivot * z_evo * k_term * cutoff
        resid = (P_model - P_data) / err
        total_chi2 += resid**2
        n_pts += 1
        print(f"{z_val:.1f}   {k:.4f}   {P_model:.4f}    {P_data:.4f}    {resid:+.2f}")
    print()

print(f"Total chi2 = {total_chi2:.1f} for {n_pts} points")
print(f"chi2/dof = {total_chi2/n_pts:.2f}")
