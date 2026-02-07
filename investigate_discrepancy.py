#!/usr/bin/env python3
"""
INVESTIGATE DATA INCONSISTENCY
==============================
There's a discrepancy between stored 'results' dict and raw velocity arrays.
This script investigates what went wrong.
"""

import numpy as np
from pathlib import Path

print("=" * 80)
print("INVESTIGATING DATA INCONSISTENCY")
print("=" * 80)

results_file = Path("results/cgc_dwarf_analysis.npz")
data = np.load(results_file, allow_pickle=True)

v_void = data['v_void']
v_cluster = data['v_cluster']
results = data['results'].item()  # Extract dict from 0-d array

print("\nStored 'results' dict:")
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n" + "-" * 60)
print("RECALCULATING FROM RAW ARRAYS:")
print("-" * 60)

# Recalculate
mean_void_calc = np.mean(v_void)
mean_cluster_calc = np.mean(v_cluster)
delta_v_calc = mean_void_calc - mean_cluster_calc

sem_void_calc = np.std(v_void) / np.sqrt(len(v_void))
sem_cluster_calc = np.std(v_cluster) / np.sqrt(len(v_cluster))

print(f"\nRecalculated values:")
print(f"  mean_void:    {mean_void_calc:.6f} km/s")
print(f"  mean_cluster: {mean_cluster_calc:.6f} km/s")
print(f"  delta_v:      {delta_v_calc:.6f} km/s")
print(f"  sem_void:     {sem_void_calc:.6f} km/s")
print(f"  sem_cluster:  {sem_cluster_calc:.6f} km/s")

print(f"\nStored values:")
print(f"  mean_void:    {results['mean_void']:.6f} km/s")
print(f"  mean_cluster: {results['mean_cluster']:.6f} km/s")
print(f"  delta_v:      {results['delta_v']:.6f} km/s")
print(f"  err_void:     {results['err_void']:.6f} km/s")
print(f"  err_cluster:  {results['err_cluster']:.6f} km/s")

print("\n" + "-" * 60)
print("COMPARISON:")
print("-" * 60)

print(f"\nmean_void difference:    {abs(mean_void_calc - results['mean_void']):.6f} km/s")
print(f"mean_cluster difference: {abs(mean_cluster_calc - results['mean_cluster']):.6f} km/s")
print(f"delta_v difference:      {abs(delta_v_calc - results['delta_v']):.6f} km/s")

# The stored delta_v is POSITIVE but our calc is NEGATIVE!
print("\n" + "=" * 60)
print("KEY FINDING:")
print("=" * 60)

if np.sign(delta_v_calc) != np.sign(results['delta_v']):
    print("⚠ SIGN MISMATCH!")
    print(f"  Stored delta_v:     {results['delta_v']:+.2f} km/s (POSITIVE)")
    print(f"  Calculated delta_v: {delta_v_calc:+.2f} km/s (NEGATIVE)")
    print("\nThis means the stored 'mean_void' and 'mean_cluster' were SWAPPED!")
    print("OR: The delta_v was calculated as cluster - void instead of void - cluster")
else:
    print("Signs match - investigating other issues...")

# Check if stored means are swapped
if abs(results['mean_void'] - mean_cluster_calc) < 0.01:
    print("\n✓ CONFIRMED: Stored 'mean_void' actually contains cluster mean!")
    print("✓ CONFIRMED: Stored 'mean_cluster' actually contains void mean!")
    print("\nThe labels were SWAPPED in the results dict!")

print("\n" + "=" * 60)
print("CORRECT VALUES:")
print("=" * 60)
print(f"TRUE void mean:    {mean_void_calc:.2f} km/s (N={len(v_void)})")
print(f"TRUE cluster mean: {mean_cluster_calc:.2f} km/s (N={len(v_cluster)})")
print(f"TRUE Δv = void - cluster = {delta_v_calc:+.2f} km/s")
