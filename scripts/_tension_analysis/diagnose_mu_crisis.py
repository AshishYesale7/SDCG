#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SDCG μ Crisis Diagnostic Script                           ║
║                                                                              ║
║  Diagnoses whether MCMC is fitting μ_bare (0.48) or μ_eff (0.149)           ║
║  and provides recommendations for fixing the likelihood function.            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_diagnosis():
    """Run the μ crisis diagnostic."""
    
    # Load chains
    chain_path = 'results/chains/mcmc_chains.npz'
    if not os.path.exists(chain_path):
        print(f"Note: Chain file not found at {chain_path}")
        print("Running with synthetic example data for demonstration...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        chains = np.random.normal(0.47, 0.03, (1000, 7))  # 7 params, μ is last
        mu_samples = chains[:, -1]
    else:
        loaded = np.load(chain_path)
        chains = loaded['chains']
        # Flatten if 3D (n_walkers, n_steps, n_params) -> (n_samples, n_params)
        if chains.ndim == 3:
            chains = chains.reshape(-1, chains.shape[-1])
        mu_samples = chains[:, -1]
    
    print('=' * 70)
    print('SDCG μ CRISIS DIAGNOSIS')
    print('=' * 70)
    
    # 1. Basic statistics
    mu_mean = np.mean(mu_samples)
    mu_std = np.std(mu_samples)
    
    print(f'\n1. MCMC RESULTS:')
    print(f'   μ = {mu_mean:.4f} ± {mu_std:.4f}')
    print(f'   95% CI: [{np.percentile(mu_samples, 2.5):.4f}, {np.percentile(mu_samples, 97.5):.4f}]')
    
    # 2. Compare with theory
    print(f'\n2. THEORY COMPARISON:')
    print(f'   Expected μ_eff (voids) = 0.149')
    print(f'   Expected μ_bare (QFT)  = 0.48')
    print(f'   MCMC found             = {mu_mean:.4f}')
    
    if abs(mu_mean - 0.48) < 0.05:
        print(f'   → Matches μ_bare (0.48), not μ_eff (0.149)!')
        print(f'   → CODE IS FITTING THE BARE COUPLING, NOT EFFECTIVE!')
        fitting_bare = True
    elif abs(mu_mean - 0.149) < 0.05:
        print(f'   → Matches μ_eff (0.149) as expected!')
        fitting_bare = False
    else:
        print(f'   → Matches neither expected value - investigate!')
        fitting_bare = mu_mean > 0.3  # Assume bare if high
    
    # 3. Physical implications
    print(f'\n3. PHYSICAL IMPLICATIONS of μ = {mu_mean:.3f}:')
    
    # H₀ prediction
    def calculate_H0(mu_value):
        """H₀ = 67.4 × (1 + 0.31×μ)"""
        return 67.4 * (1 + 0.31 * mu_value)
    
    H0_sdcg = calculate_H0(mu_mean)
    print(f'   H₀(Planck) = 67.4 km/s/Mpc')
    print(f'   H₀(SDCG, μ={mu_mean:.3f}) = {H0_sdcg:.1f} km/s/Mpc')
    print(f'   H₀(SH0ES) = 73.0 km/s/Mpc')
    
    if H0_sdcg > 73.0:
        print(f'   ⚠️ WARNING: H₀ prediction EXCEEDS SH0ES value!')
    
    # Lyman-α constraint
    def mu_lyalpha(mu_bare):
        """μ_eff(Lyα) = μ_bare × S(ρ_IGM) × f(z=3)"""
        S_IGM = 0.95  # Screening in IGM
        f_z3 = 0.24   # Redshift suppression at z=3
        return mu_bare * S_IGM * f_z3
    
    mu_lya = mu_lyalpha(mu_mean)
    print(f'\n   Lyman-α constraint:')
    print(f'   μ_eff(Lyα) = μ × S_IGM × f(z=3)')
    print(f'               = {mu_mean:.3f} × 0.95 × 0.24')
    print(f'               = {mu_lya:.3f}')
    print(f'   Constraint: μ_eff(Lyα) < 0.05')
    status = 'VIOLATED! (2× too high)' if mu_lya > 0.05 else 'OK'
    print(f'   Status: {status}')
    
    # 4. Screening check
    print(f'\n4. SCREENING FACTOR CALCULATION:')
    print(f'   If MCMC μ = μ_bare = {mu_mean:.3f}, then to get μ_eff = 0.149:')
    screening_needed = 0.149 / mu_mean
    print(f'   Required screening: S = μ_eff / μ_bare = 0.149 / {mu_mean:.3f} = {screening_needed:.3f}')
    print(f'   This means average universe density should give S ≈ {screening_needed:.3f}')
    
    # 5. Parameter correlations
    print(f'\n5. PARAMETER CORRELATIONS:')
    param_names = ['omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau', 'mu']
    corr_matrix = np.corrcoef(chains.T)
    
    print('   μ correlations:')
    for i, name in enumerate(param_names):
        corr = corr_matrix[i, -1]
        strength = '(STRONG)' if abs(corr) > 0.5 else ''
        if abs(corr) > 0.3:
            print(f'   μ ↔ {name}: {corr:.3f} {strength}')
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # μ posterior
    axes[0, 0].hist(mu_samples, bins=50, density=True, alpha=0.7, color='blue')
    axes[0, 0].axvline(0.149, color='red', linewidth=3, label=r'Expected $\mu_{eff}$ = 0.149')
    axes[0, 0].axvline(0.48, color='green', linewidth=2, linestyle='--', label=r'$\mu_{bare}$ = 0.48')
    axes[0, 0].axvline(mu_mean, color='black', linewidth=2, label=f'MCMC μ = {mu_mean:.3f}')
    axes[0, 0].set_xlabel(r'$\mu$', fontsize=12)
    axes[0, 0].set_ylabel('Probability Density', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].set_title(r'μ Crisis: MCMC finds $\mu_{bare}$, not $\mu_{eff}$', fontsize=14)
    
    # H₀ vs μ
    h_samples = chains[:, 2] * 100
    axes[0, 1].scatter(mu_samples[::10], h_samples[::10], alpha=0.3, s=1, c='blue')
    axes[0, 1].axhline(67.4, color='blue', linestyle='--', linewidth=2, label=r'Planck $H_0$')
    axes[0, 1].axhline(73.0, color='orange', linestyle='--', linewidth=2, label=r'SH0ES $H_0$')
    axes[0, 1].axvline(0.149, color='red', linestyle=':', linewidth=2, label=r'Expected $\mu_{eff}$')
    axes[0, 1].set_xlabel(r'$\mu$', fontsize=12)
    axes[0, 1].set_ylabel(r'$H_0$ (km/s/Mpc)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].set_title(r'$H_0$ vs $\mu$ Correlation', fontsize=14)
    
    # Screening factor distribution
    rho_values = np.logspace(-1, 3, 100)
    S_values = 1 / (1 + (rho_values / 200)**2)
    axes[1, 0].semilogx(rho_values, S_values, linewidth=2, color='blue')
    axes[1, 0].axhline(screening_needed, color='red', linestyle='--', linewidth=2,
                       label=f'S needed = {screening_needed:.2f}')
    axes[1, 0].axvline(1.0, color='gray', linestyle=':', label=r'$\rho = \rho_{crit}$')
    axes[1, 0].set_xlabel(r'$\rho/\rho_{crit}$ (log scale)', fontsize=12)
    axes[1, 0].set_ylabel(r'Screening Factor S($\rho$)', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].set_title('Screening Function', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0.1, 1000])
    axes[1, 0].set_ylim([0, 1.1])
    
    # Chain trace for μ (subsample)
    n_samples = len(mu_samples)
    step_indices = np.arange(n_samples)
    axes[1, 1].plot(step_indices[::20], mu_samples[::20], alpha=0.5, linewidth=0.5)
    axes[1, 1].axhline(0.149, color='red', linewidth=2, label=r'$\mu_{eff}$ target = 0.149')
    axes[1, 1].axhline(mu_mean, color='black', linewidth=1, linestyle='--',
                       label=f'MCMC mean = {mu_mean:.3f}')
    axes[1, 1].set_xlabel('Sample Index', fontsize=12)
    axes[1, 1].set_ylabel(r'$\mu$', fontsize=12)
    axes[1, 1].set_title('Chain Trace (subsampled)', fontsize=14)
    axes[1, 1].legend()
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/sdcg_mu_crisis_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f'\n📊 Diagnostic plot saved to plots/sdcg_mu_crisis_diagnostic.png')
    
    # Conclusion and recommendations
    print(f'\n' + '=' * 70)
    print('DIAGNOSIS SUMMARY')
    print('=' * 70)
    
    if fitting_bare:
        print('\n⚠️ PROBLEM: Your code is fitting μ_bare (~0.47), not μ_eff (~0.149)!')
        print('\n🔧 RECOMMENDED FIXES:')
        print('\nOption 1: Reparameterize to sample μ_eff (RECOMMENDED)')
        print('   - Sample μ_eff in MCMC (range: 0.1-0.3)')
        print('   - Compute μ_bare = μ_eff / S_avg inside likelihood')
        print('   - Use different μ_eff for different datasets (CMB, Lyα, etc.)')
        print('\nOption 2: Add strong Gaussian prior on μ')
        print('   - Add prior: N(μ=0.149, σ=0.02)')
        print('   - This forces μ to expected μ_eff value')
        print('\nOption 3: Apply screening in likelihood (complex)')
        print('   - For each dataset, compute environment-specific μ_eff')
        print('   - μ_eff(CMB) = μ_bare × S(ρ_avg)')
        print('   - μ_eff(Lyα) = μ_bare × S(ρ_IGM)')
    else:
        print('\n✅ Your code appears to be correctly fitting μ_eff!')
    
    print('\n' + '=' * 70)
    
    return fitting_bare, mu_mean, mu_std


if __name__ == '__main__':
    run_diagnosis()
