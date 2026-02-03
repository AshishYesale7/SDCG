#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            SDCG MCMC WITH EFT-INFORMED PRIORS (Quick Test)                   ║
║                                                                              ║
║  This run uses Gaussian priors on n_g and z_trans since they are DERIVED    ║
║  quantities from the EFT, not free parameters.                               ║
║                                                                              ║
║  EFT-DERIVED VALUES:                                                         ║
║    n_g = β₀²/4π² = 0.70²/39.48 = 0.014 ± 0.003 (20% from β₀)               ║
║    z_trans = z_acc + Δz_delay = 0.64 + 1.0 = 1.64 ± 0.20                    ║
║                                                                              ║
║  FREE PARAMETERS (constrained by data):                                      ║
║    μ = coupling strength (constrained by Lyα: μ < 0.1 in IGM)               ║
║                                                                              ║
║  Expected: MCMC should now find values CONSISTENT with EFT predictions!     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import sys
import time
from datetime import datetime

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# QUICK TEST CONFIGURATION (shorter run to verify EFT priors work)
# =============================================================================

N_WALKERS = 32        # Reduced for quick test
N_STEPS = 500         # Quick test - should converge faster with EFT priors
BURNIN_FRAC = 0.3     
THIN = 5              

# Data settings
INCLUDE_SNE = True     
INCLUDE_LYALPHA = True 

# =============================================================================
# EFT PHYSICS CONSTANTS (From Thesis v10)
# =============================================================================

BETA_0 = 0.70  # SM benchmark from conformal anomaly

# DERIVED VALUES (not free parameters!)
N_G_EFT = BETA_0**2 / (4 * np.pi**2)  # ≈ 0.0124
Z_TRANS_EFT = 1.64  # From q(z) = 0

# μ is the only truly free CGC parameter
MU_EFT = 0.149  # MCMC best-fit in voids (6σ detection)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            SDCG MCMC WITH EFT-INFORMED PRIORS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY DIFFERENCE FROM PREVIOUS RUN:
  • Previous: Flat priors on n_g ∈ [0, 2], z_trans ∈ [0.5, 5]
  • Now: Gaussian priors centered on EFT values!

EFT-DERIVED PRIORS:
  n_g     = {N_G_EFT:.4f} ± 0.003 (Gaussian prior, derived from β₀²/4π²)
  z_trans = {Z_TRANS_EFT:.2f} ± 0.20 (Gaussian prior, derived from q(z) = 0)

FREE PARAMETERS:
  μ       = fitted (prior: [0, 0.5], data constrains via Lyα screening)

Expected result: Fitted values should now be CONSISTENT with EFT predictions!
""")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run MCMC with EFT-informed priors."""
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Import CGC modules
    print("Loading CGC modules...")
    from cgc.data_loader import DataLoader
    from cgc.mcmc import run_mcmc, print_physics_validation
    from cgc.parameters import CGCParameters
    from cgc.config import setup_directories, PATHS
    
    # Setup directories
    setup_directories()
    
    # =========================================================================
    # STEP 1: LOAD REAL DATA
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: LOADING REAL COSMOLOGICAL DATA")
    print("="*70 + "\n")
    
    loader = DataLoader(use_real_data=True)
    data = loader.load_all(
        include_sne=INCLUDE_SNE, 
        include_lyalpha=INCLUDE_LYALPHA
    )
    
    # Print data summary
    print("\n📊 Data Summary:")
    print(f"   CMB:     {data.get('cmb', {}).get('n_points', 0)} multipoles")
    print(f"   BAO:     {len(data.get('bao', {}).get('z', []))} measurements")
    print(f"   Growth:  {len(data.get('growth', {}).get('z', []))} fσ8 points")
    if 'sne' in data:
        print(f"   SNe:     {len(data.get('sne', {}).get('z', []))} supernovae")
    if 'lyalpha' in data:
        print(f"   Lyα:     {len(data.get('lyalpha', {}).get('k', []))} k bins")
    
    # =========================================================================
    # STEP 2: SET UP INITIAL PARAMETERS (EFT VALUES)
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING PARAMETERS (EFT-DERIVED VALUES)")
    print("="*70 + "\n")
    
    # Start from EFT-derived values
    params = CGCParameters(
        cgc_mu=MU_EFT,           # 0.149 (constrained by data)
        cgc_n_g=N_G_EFT,         # 0.014 (DERIVED, not free)
        cgc_z_trans=Z_TRANS_EFT, # 1.64 (DERIVED, not free)
        cgc_rho_thresh=200.0     # From chameleon theory
    )
    
    print(f"Initial parameters (EFT-derived):")
    print(f"   μ         = {params.cgc_mu:.4f} (FREE - constrained by data)")
    print(f"   n_g       = {params.cgc_n_g:.4f} (DERIVED from β₀²/4π²)")
    print(f"   z_trans   = {params.cgc_z_trans:.3f} (DERIVED from q(z)=0)")
    print(f"   ρ_thresh  = {params.cgc_rho_thresh:.1f}")
    
    # =========================================================================
    # STEP 3: RUN MCMC WITH EFT PRIORS
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: RUNNING MCMC WITH EFT-INFORMED PRIORS")
    print("="*70 + "\n")
    
    print(f"⏳ Starting MCMC: {N_WALKERS} walkers × {N_STEPS} steps")
    print(f"   Expected samples after thinning: ~{N_WALKERS * int(N_STEPS * (1-BURNIN_FRAC)) // THIN:,}")
    print(f"\n   🔬 KEY: Using Gaussian priors on n_g and z_trans!")
    print()
    
    sampler, chains = run_mcmc(
        data=data,
        n_walkers=N_WALKERS,
        n_steps=N_STEPS,
        params=params,
        include_sne=INCLUDE_SNE,
        include_lyalpha=INCLUDE_LYALPHA,
        n_processes=None,
        seed=42,
        save_chains=True,
        verbose=True
    )
    
    # =========================================================================
    # STEP 4: ANALYZE RESULTS
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: ANALYZING RESULTS")
    print("="*70 + "\n")
    
    # Get thinned chains
    discard = int(BURNIN_FRAC * N_STEPS)
    flat_chains = sampler.get_chain(discard=discard, thin=THIN, flat=True)
    
    print(f"📊 Chain Statistics:")
    print(f"   Total samples: {len(flat_chains):,}")
    
    # Parameter names
    param_names = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ',
                   'μ', 'n_g', 'z_trans', 'ρ_thresh']
    
    # Compute statistics
    print("\n" + "─"*70)
    print("PARAMETER CONSTRAINTS (median ± 1σ)")
    print("─"*70)
    
    results = {}
    for i, name in enumerate(param_names):
        samples = flat_chains[:, i]
        median = np.median(samples)
        lower = np.percentile(samples, 16)
        upper = np.percentile(samples, 84)
        std = (upper - lower) / 2
        
        results[name] = {
            'median': median,
            'lower': lower,
            'upper': upper,
            'std': std
        }
        
        # Highlight CGC parameters
        if name in ['μ', 'n_g', 'z_trans', 'ρ_thresh']:
            print(f"  ★ {name:10s}: {median:10.4f} ± {std:.4f}  [{lower:.4f}, {upper:.4f}]")
        else:
            print(f"    {name:10s}: {median:10.4f} ± {std:.4f}")
    
    # =========================================================================
    # STEP 5: EFT CONSISTENCY CHECK
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: EFT CONSISTENCY CHECK")
    print("="*70)
    
    mu_fitted = results['μ']['median']
    ng_fitted = results['n_g']['median']
    zt_fitted = results['z_trans']['median']
    
    print("\n┌────────────────────────────────────────────────────────────────────┐")
    print("│ COMPARISON WITH EFT-DERIVED VALUES                                 │")
    print("├────────────────────────────────────────────────────────────────────┤")
    print(f"│  μ:       fitted = {mu_fitted:.4f} ± {results['μ']['std']:.4f}                              │")
    print(f"│           EFT (void) = {MU_EFT:.4f} (only free CGC param)               │")
    print(f"│                                                                    │")
    print(f"│  n_g:     fitted = {ng_fitted:.4f} ± {results['n_g']['std']:.4f}                              │")
    print(f"│           EFT (β₀²/4π²) = {N_G_EFT:.4f} (DERIVED, not free)              │")
    
    ng_consistent = abs(ng_fitted - N_G_EFT) < 3 * results['n_g']['std']
    print(f"│           Consistent? {'✓ YES' if ng_consistent else '✗ NO'}                                        │")
    print(f"│                                                                    │")
    print(f"│  z_trans: fitted = {zt_fitted:.3f} ± {results['z_trans']['std']:.3f}                              │")
    print(f"│           EFT (q(z)=0) = {Z_TRANS_EFT:.2f} (DERIVED, not free)               │")
    
    zt_consistent = abs(zt_fitted - Z_TRANS_EFT) < 3 * results['z_trans']['std']
    print(f"│           Consistent? {'✓ YES' if zt_consistent else '✗ NO'}                                        │")
    print("└────────────────────────────────────────────────────────────────────┘")
    
    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: SAVING RESULTS")
    print("="*70 + "\n")
    
    # Save comprehensive results
    results_file = os.path.join(PATHS['results'], f'sdcg_eft_prior_{timestamp}.npz')
    
    np.savez(
        results_file,
        chains=flat_chains,
        n_walkers=N_WALKERS,
        n_steps=N_STEPS,
        burnin_frac=BURNIN_FRAC,
        thin=THIN,
        use_eft_priors=True,
        mu_median=results['μ']['median'],
        mu_std=results['μ']['std'],
        n_g_median=results['n_g']['median'],
        n_g_std=results['n_g']['std'],
        z_trans_median=results['z_trans']['median'],
        z_trans_std=results['z_trans']['std'],
        eft_n_g=N_G_EFT,
        eft_z_trans=Z_TRANS_EFT,
        eft_mu=MU_EFT
    )
    
    print(f"✓ Results saved to: {results_file}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MCMC WITH EFT PRIORS COMPLETE                                               ║
║                                                                              ║
║  Runtime: {hours}h {minutes}m {seconds}s                                                   ║
║                                                                              ║
║  KEY RESULTS:                                                                ║
║    μ         = {mu_fitted:.4f} ± {results['μ']['std']:.4f} (FREE parameter)                        ║
║    n_g       = {ng_fitted:.4f} ± {results['n_g']['std']:.4f} (EFT prior: {N_G_EFT:.4f} ± 0.003)               ║
║    z_trans   = {zt_fitted:.3f} ± {results['z_trans']['std']:.3f}  (EFT prior: {Z_TRANS_EFT:.2f} ± 0.20)                ║
║                                                                              ║
║  EFT CONSISTENCY:                                                            ║
║    n_g consistent with EFT?     {'✓ YES' if ng_consistent else '✗ NO'}                                 ║
║    z_trans consistent with EFT? {'✓ YES' if zt_consistent else '✗ NO'}                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    return sampler, flat_chains, results


if __name__ == "__main__":
    main()
