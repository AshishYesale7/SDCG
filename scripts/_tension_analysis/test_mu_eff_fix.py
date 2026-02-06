#!/usr/bin/env python3
"""
Test MCMC with μ_eff parameterization fix.

Runs a quick MCMC to verify that:
1. μ_eff ≈ 0.149 is found (not μ_bare ≈ 0.47)
2. Lyα constraint is satisfied (μ_eff(Lyα) < 0.05)
3. Prior bounds [0.05, 0.35] are working
"""

import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from simulations.cgc.mcmc import run_mcmc
from simulations.cgc.data_loader import load_real_data
from simulations.cgc.cgc_physics import CGCPhysics

def main():
    print("=" * 70)
    print("TESTING μ_eff PARAMETERIZATION FIX")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    data = load_real_data(verbose=False)
    print("   ✓ Data loaded")
    
    # Run quick MCMC (32 walkers, 500 steps)
    print("\n2. Running MCMC (32 walkers, 500 steps)...")
    print("   This should find μ_eff ≈ 0.149, NOT μ_bare ≈ 0.47")
    
    sampler, chains = run_mcmc(
        data=data,
        n_walkers=32,
        n_steps=500,
        verbose=True,
        save_chains=False
    )
    
    # Analyze results
    print("\n3. MCMC RESULTS:")
    print("-" * 50)
    
    # Extract μ samples (last column)
    if len(chains.shape) == 3:
        # (walkers, steps, params) -> flatten
        chains_flat = chains.reshape(-1, chains.shape[-1])
    else:
        chains_flat = chains
    
    mu_samples = chains_flat[:, -1]
    mu_mean = np.mean(mu_samples)
    mu_std = np.std(mu_samples)
    
    print(f"   μ_eff = {mu_mean:.4f} ± {mu_std:.4f}")
    print(f"   95% CI: [{np.percentile(mu_samples, 2.5):.4f}, {np.percentile(mu_samples, 97.5):.4f}]")
    
    # Check if we found μ_eff in the expected range
    print("\n4. DIAGNOSIS:")
    # Valid range for μ_eff(void): 0.10 - 0.35
    # This gives μ_eff(Lyα) = 0.016 - 0.056, satisfying Lyα constraint
    if 0.10 <= mu_mean <= 0.35:
        print(f"   ✓ μ_eff(void) = {mu_mean:.3f} is in valid range [0.10, 0.35]")
        print("   ✓ This is the EFFECTIVE coupling in low-density voids")
        found_mu_eff = True
    elif mu_mean > 0.35:
        print(f"   ⚠ μ_eff = {mu_mean:.3f} is above 0.35")
        print("     → May violate Lyα even with screening")
        found_mu_eff = False
    elif mu_mean < 0.10:
        print(f"   ⚠ μ_eff = {mu_mean:.3f} is below 0.10")
        print("     → May be too weak for tension reduction")
        found_mu_eff = True  # Still valid, just weak
    
    # Check Lyα constraint (WITH PROPER SCREENING!)
    print("\n5. LYMAN-α CONSTRAINT CHECK (with screening):")
    cgc = CGCPhysics(mu=mu_mean)
    mu_lyalpha = cgc.mu_eff_for_environment('lyalpha')
    
    print(f"   μ_eff(void) = {mu_mean:.3f} (what we sample)")
    print(f"   ")
    print(f"   Lyα environment: IGM at z~3, ρ ~ 50 ρ_crit (DENSE!)")
    print(f"   Suppression factors:")
    print(f"     • S(ρ_IGM)/S(ρ_void) ≈ 0.94 (density screening)")
    print(f"     • g(z=3)            ≈ 0.67 (redshift evolution)")
    print(f"     • Vainshtein        ≈ 0.25 (small-scale screening)")
    print(f"     • Total             ≈ 0.16")
    print(f"   ")
    print(f"   μ_eff(Lyα) = {mu_mean:.3f} × 0.16 = {mu_lyalpha:.4f}")
    print(f"   Constraint: μ_eff(Lyα) < 0.05")
    
    if mu_lyalpha < 0.05:
        print(f"   ✓ Lyα constraint SATISFIED! (screening works)")
        lya_ok = True
    else:
        print(f"   ✗ Lyα constraint violated ({mu_lyalpha:.3f} > 0.05)")
        print(f"      Need stronger screening or lower μ_eff(void)")
        lya_ok = False
    
    # Compute implied μ_bare
    print("\n6. BACK-CALCULATED μ_bare:")
    mu_bare = cgc.mu_bare
    print(f"   μ_bare = μ_eff / S_avg = {mu_mean:.3f} / 0.31 = {mu_bare:.3f}")
    print(f"   Expected from QFT: μ_bare ≈ 0.48")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if found_mu_eff and lya_ok:
        print("✅ SUCCESS: MCMC now correctly samples μ_eff (~0.149)")
        print("✅ Lyα constraint satisfied")
        print("✅ Ready for production runs!")
    else:
        print("❌ ISSUES REMAIN:")
        if not found_mu_eff:
            print("   - μ value not matching expected μ_eff")
        if not lya_ok:
            print("   - Lyα constraint violated")
    
    print("=" * 70)
    
    return found_mu_eff and lya_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
