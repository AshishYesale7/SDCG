#!/usr/bin/env python3
"""Verify tight EFT priors overcome likelihood preference."""

import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from simulations.cgc.likelihoods import log_probability
from simulations.cgc.data_loader import load_real_data

print("="*70)
print("VERIFYING TIGHT EFT PRIORS WITH REAL DATA")
print("="*70)

# Load real data
print("\nLoading real cosmological data...")
data = load_real_data(verbose=False)

# Test parameters (7 params: omega_b, omega_cdm, h, ln10As, n_s, tau, mu)
# n_g, z_trans, rho_thresh are fixed by theory
theta_v12 = np.array([0.02242, 0.120, 0.6736, 3.044, 0.9649, 0.054, 0.47])
theta_mcmc = np.array([0.02242, 0.120, 0.6736, 3.044, 0.9649, 0.054, 0.36])

kwargs = {'include_sne': False, 'include_lyalpha': False}

print("\n" + "-"*70)
print("POSTERIOR COMPARISON (REAL DATA)")
print("-"*70)

# No EFT priors (flat only)
logP_v12_flat = log_probability(theta_v12, data, use_eft_priors=False, 
                                 tight_eft_priors=False, **kwargs)
logP_mcmc_flat = log_probability(theta_mcmc, data, use_eft_priors=False, 
                                  tight_eft_priors=False, **kwargs)

print(f"\n1. FLAT PRIORS (no EFT):")
print(f"   v12 values:  logP = {logP_v12_flat:.2f}")
print(f"   MCMC values: logP = {logP_mcmc_flat:.2f}")
print(f"   Δ(v12 - mcmc) = {logP_v12_flat - logP_mcmc_flat:.2f}")
if logP_v12_flat > logP_mcmc_flat:
    print(f"   → v12 preferred ✓")
else:
    print(f"   → MCMC preferred (likelihood pulls wrong!)")

# Moderate EFT priors
logP_v12_mod = log_probability(theta_v12, data, use_eft_priors=True, 
                                tight_eft_priors=False, **kwargs)
logP_mcmc_mod = log_probability(theta_mcmc, data, use_eft_priors=True, 
                                 tight_eft_priors=False, **kwargs)

print(f"\n2. MODERATE EFT PRIORS (n_g ± 0.003, z_trans ± 0.20):")
print(f"   v12 values:  logP = {logP_v12_mod:.2f}")
print(f"   MCMC values: logP = {logP_mcmc_mod:.2f}")
print(f"   Δ(v12 - mcmc) = {logP_v12_mod - logP_mcmc_mod:.2f}")
if logP_v12_mod > logP_mcmc_mod:
    print(f"   → v12 preferred ✓")
else:
    print(f"   → MCMC preferred ❌ (priors too weak)")

# Tight EFT priors
logP_v12_tight = log_probability(theta_v12, data, use_eft_priors=False, 
                                  tight_eft_priors=True, **kwargs)
logP_mcmc_tight = log_probability(theta_mcmc, data, use_eft_priors=False, 
                                   tight_eft_priors=True, **kwargs)

print(f"\n3. TIGHT EFT PRIORS (μ ± 0.03, n_g ± 0.001, z_trans ± 0.05):")
print(f"   v12 values:  logP = {logP_v12_tight:.2f}")
print(f"   MCMC values: logP = {logP_mcmc_tight:.2f}")
print(f"   Δ(v12 - mcmc) = {logP_v12_tight - logP_mcmc_tight:.2f}")
if logP_v12_tight > logP_mcmc_tight:
    print(f"   → v12 preferred ✓")
else:
    print(f"   → MCMC preferred ❌ (priors still too weak!)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if logP_v12_tight > logP_mcmc_tight:
    delta = logP_v12_tight - logP_mcmc_tight
    print(f"\n✓ With tight EFT priors, v12 values are preferred by Δ = {delta:.1f}")
    print(f"  MCMC should now converge to v12 thesis values when using --tight-eft flag.")
else:
    delta = logP_mcmc_tight - logP_v12_tight
    print(f"\n❌ Even with tight priors, MCMC values are preferred by Δ = {delta:.1f}")
    print(f"   Need to fix the likelihood model itself!")

print("="*70)
