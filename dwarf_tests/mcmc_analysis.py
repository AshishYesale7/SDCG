#!/usr/bin/env python3
"""
CGC Dwarf Test - MCMC Analysis Module
======================================
Uses MCMC to properly sample the posterior distribution of the
velocity difference Δv between void and cluster dwarf galaxies.

This provides:
- Full posterior distribution of Δv
- Bayesian model comparison (CGC vs ΛCDM)
- Publication-quality corner plots and posterior plots

CGC Prediction: Δv = +12 ± 3 km/s (void dwarfs rotate faster)
ΛCDM Prediction: Δv = 0 km/s (no environment dependence)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# THEORY PREDICTIONS
# =============================================================================

CGC_PREDICTED_DELTA_V = 12.0   # km/s
CGC_PREDICTED_ERROR = 3.0      # km/s
LCDM_PREDICTED_DELTA_V = 0.0   # km/s


# =============================================================================
# LIKELIHOOD FUNCTIONS
# =============================================================================

def log_likelihood_velocity(theta, v_void, v_void_err, v_cluster, v_cluster_err):
    """
    Log-likelihood for velocity difference model.
    
    Model: Each sample has a true mean velocity with intrinsic scatter.
    
    Parameters
    ----------
    theta : array
        [mu_void, mu_cluster, sigma_int] - mean velocities and intrinsic scatter
    v_void : array
        Observed void dwarf velocities
    v_void_err : array
        Velocity errors for void dwarfs
    v_cluster : array
        Observed cluster dwarf velocities
    v_cluster_err : array
        Velocity errors for cluster dwarfs
    
    Returns
    -------
    float
        Log-likelihood value
    """
    mu_void, mu_cluster, sigma_int = theta
    
    if sigma_int < 0:
        return -np.inf
    
    # Total variance = measurement error² + intrinsic scatter²
    var_void = v_void_err**2 + sigma_int**2
    var_cluster = v_cluster_err**2 + sigma_int**2
    
    # Log-likelihood for void dwarfs
    ll_void = -0.5 * np.sum(
        (v_void - mu_void)**2 / var_void + np.log(2 * np.pi * var_void)
    )
    
    # Log-likelihood for cluster dwarfs
    ll_cluster = -0.5 * np.sum(
        (v_cluster - mu_cluster)**2 / var_cluster + np.log(2 * np.pi * var_cluster)
    )
    
    return ll_void + ll_cluster


def log_prior(theta):
    """
    Flat prior on velocity parameters.
    
    Parameters
    ----------
    theta : array
        [mu_void, mu_cluster, sigma_int]
    
    Returns
    -------
    float
        Log-prior (0 if in bounds, -inf otherwise)
    """
    mu_void, mu_cluster, sigma_int = theta
    
    # Reasonable bounds for dwarf galaxy velocities
    if not (10 < mu_void < 200):
        return -np.inf
    if not (10 < mu_cluster < 200):
        return -np.inf
    if not (0 < sigma_int < 100):
        return -np.inf
    
    return 0.0


def log_posterior(theta, v_void, v_void_err, v_cluster, v_cluster_err):
    """
    Log-posterior = log-prior + log-likelihood
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood_velocity(theta, v_void, v_void_err, v_cluster, v_cluster_err)
    
    return lp + ll


# =============================================================================
# MCMC SAMPLER
# =============================================================================

def run_dwarf_mcmc(v_void, v_void_err, v_cluster, v_cluster_err,
                   n_walkers=32, n_steps=2000, n_burn=500,
                   seed=42, verbose=True):
    """
    Run MCMC to sample the posterior of velocity parameters.
    
    Parameters
    ----------
    v_void, v_cluster : arrays
        Velocity measurements
    v_void_err, v_cluster_err : arrays
        Velocity errors
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Burn-in steps to discard
    seed : int
        Random seed
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        MCMC results including chains, samples, and derived quantities
    """
    try:
        import emcee
    except ImportError:
        print("Installing emcee...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "emcee"])
        import emcee
    
    np.random.seed(seed)
    
    # Dimensionality: [mu_void, mu_cluster, sigma_int]
    n_dim = 3
    
    # Initial positions - start near the data means
    mu_void_init = np.mean(v_void)
    mu_cluster_init = np.mean(v_cluster)
    sigma_init = np.std(np.concatenate([v_void, v_cluster])) / 2
    
    theta0 = np.array([mu_void_init, mu_cluster_init, sigma_init])
    
    # Initialize walkers in a ball around the initial guess
    pos = theta0 + 1e-2 * np.random.randn(n_walkers, n_dim)
    
    # Ensure sigma_int is positive for all walkers
    pos[:, 2] = np.abs(pos[:, 2])
    
    if verbose:
        print("\n" + "=" * 60)
        print("CGC DWARF TEST - MCMC ANALYSIS")
        print("=" * 60)
        print(f"Void dwarfs:    N = {len(v_void)}")
        print(f"Cluster dwarfs: N = {len(v_cluster)}")
        print(f"MCMC: {n_walkers} walkers × {n_steps} steps")
        print("-" * 60)
    
    # Set up sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior,
        args=(v_void, v_void_err, v_cluster, v_cluster_err)
    )
    
    # Run MCMC
    if verbose:
        print("Running MCMC...")
    
    sampler.run_mcmc(pos, n_steps, progress=verbose)
    
    # Get chains after burn-in
    chains = sampler.get_chain(discard=n_burn, thin=10, flat=True)
    
    # Extract samples
    mu_void_samples = chains[:, 0]
    mu_cluster_samples = chains[:, 1]
    sigma_int_samples = chains[:, 2]
    
    # Compute Δv = mu_void - mu_cluster
    delta_v_samples = mu_void_samples - mu_cluster_samples
    
    # Summary statistics
    delta_v_mean = np.mean(delta_v_samples)
    delta_v_std = np.std(delta_v_samples)
    delta_v_median = np.median(delta_v_samples)
    delta_v_16 = np.percentile(delta_v_samples, 16)
    delta_v_84 = np.percentile(delta_v_samples, 84)
    
    # Compute Bayes factors
    # P(Δv | CGC) vs P(Δv | ΛCDM)
    # Using Savage-Dickey density ratio approximation
    
    # Prior is flat, so density at any point is ~ 1/prior_width
    prior_width = 200 - 10  # From our prior bounds
    
    # Posterior density at CGC prediction (Δv = 12)
    kde = stats.gaussian_kde(delta_v_samples)
    posterior_at_cgc = kde(CGC_PREDICTED_DELTA_V)[0]
    posterior_at_lcdm = kde(LCDM_PREDICTED_DELTA_V)[0]
    
    # Bayes factor (CGC vs ΛCDM)
    if posterior_at_lcdm > 0:
        bayes_factor = posterior_at_cgc / posterior_at_lcdm
    else:
        bayes_factor = np.inf
    
    # Credible intervals
    ci_68 = (delta_v_16, delta_v_84)
    ci_95 = (np.percentile(delta_v_samples, 2.5), np.percentile(delta_v_samples, 97.5))
    
    # P(Δv > 0) - probability that void dwarfs rotate faster
    p_positive = np.mean(delta_v_samples > 0)
    
    # P(Δv consistent with CGC) - within 2σ of prediction
    p_cgc = np.mean(np.abs(delta_v_samples - CGC_PREDICTED_DELTA_V) < 2 * CGC_PREDICTED_ERROR)
    
    # P(Δv consistent with ΛCDM) - within 3 km/s of zero
    p_lcdm = np.mean(np.abs(delta_v_samples) < 3)
    
    results = {
        # Chains
        'chains': chains,
        'delta_v_samples': delta_v_samples,
        'mu_void_samples': mu_void_samples,
        'mu_cluster_samples': mu_cluster_samples,
        'sigma_int_samples': sigma_int_samples,
        
        # Summary statistics
        'delta_v_mean': delta_v_mean,
        'delta_v_std': delta_v_std,
        'delta_v_median': delta_v_median,
        'ci_68': ci_68,
        'ci_95': ci_95,
        
        # Mean velocities
        'mu_void': np.mean(mu_void_samples),
        'mu_void_err': np.std(mu_void_samples),
        'mu_cluster': np.mean(mu_cluster_samples),
        'mu_cluster_err': np.std(mu_cluster_samples),
        'sigma_int': np.mean(sigma_int_samples),
        
        # Model comparison
        'bayes_factor_cgc_lcdm': bayes_factor,
        'p_positive': p_positive,
        'p_cgc': p_cgc,
        'p_lcdm': p_lcdm,
        
        # MCMC diagnostics
        'acceptance_fraction': np.mean(sampler.acceptance_fraction),
        'n_samples': len(delta_v_samples),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("MCMC RESULTS")
        print("=" * 60)
        print(f"\nVelocity Difference (Void - Cluster):")
        print(f"  Δv = {delta_v_mean:.2f} ± {delta_v_std:.2f} km/s")
        print(f"  Median: {delta_v_median:.2f} km/s")
        print(f"  68% CI: [{ci_68[0]:.2f}, {ci_68[1]:.2f}] km/s")
        print(f"  95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}] km/s")
        print(f"\nModel Comparison:")
        print(f"  P(Δv > 0) = {p_positive:.3f}")
        print(f"  P(CGC-consistent) = {p_cgc:.3f}")
        print(f"  P(ΛCDM-consistent) = {p_lcdm:.3f}")
        print(f"  Bayes Factor (CGC/ΛCDM) = {bayes_factor:.2f}")
        print(f"\nMCMC Diagnostics:")
        print(f"  Acceptance fraction: {results['acceptance_fraction']:.3f}")
        print(f"  Effective samples: {results['n_samples']}")
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_delta_v_posterior(results, save_path='plots'):
    """
    Plot the posterior distribution of Δv with theory predictions.
    
    Parameters
    ----------
    results : dict
        MCMC results from run_dwarf_mcmc()
    save_path : str
        Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delta_v = results['delta_v_samples']
    
    # Histogram of posterior
    ax.hist(delta_v, bins=50, density=True, alpha=0.7, 
            color='steelblue', edgecolor='darkblue', label='Posterior P(Δv|data)')
    
    # KDE for smoother curve
    kde = stats.gaussian_kde(delta_v)
    x_plot = np.linspace(delta_v.min() - 5, delta_v.max() + 5, 200)
    ax.plot(x_plot, kde(x_plot), 'b-', linewidth=2)
    
    # Mark predictions
    # ΛCDM
    ax.axvline(LCDM_PREDICTED_DELTA_V, color='gray', linestyle='--', 
               linewidth=2.5, label='ΛCDM: Δv = 0')
    ax.axvspan(-3, 3, color='gray', alpha=0.2)
    
    # CGC
    ax.axvline(CGC_PREDICTED_DELTA_V, color='green', linestyle='--', 
               linewidth=2.5, label=f'CGC: Δv = +{CGC_PREDICTED_DELTA_V} km/s')
    ax.axvspan(CGC_PREDICTED_DELTA_V - CGC_PREDICTED_ERROR, 
               CGC_PREDICTED_DELTA_V + CGC_PREDICTED_ERROR, 
               color='green', alpha=0.2)
    
    # Mark measured value
    ax.axvline(results['delta_v_mean'], color='red', linestyle='-', 
               linewidth=2.5, label=f"Measured: Δv = {results['delta_v_mean']:.1f} ± {results['delta_v_std']:.1f} km/s")
    
    # Credible interval
    ci = results['ci_95']
    ax.axvspan(ci[0], ci[1], color='red', alpha=0.15, label='95% CI')
    
    ax.set_xlabel('Δv = V$_{void}$ - V$_{cluster}$ [km/s]', fontsize=14)
    ax.set_ylabel('Posterior Probability Density', fontsize=14)
    ax.set_title('Posterior Distribution of Velocity Difference (MCMC)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add text box with statistics
    textstr = '\n'.join([
        f"N$_{{void}}$ = {len(results['mu_void_samples'])} samples",
        f"Δv = {results['delta_v_mean']:.1f} ± {results['delta_v_std']:.1f} km/s",
        f"P(Δv > 0) = {results['p_positive']:.2f}",
        f"Bayes Factor = {results['bayes_factor_cgc_lcdm']:.1f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'delta_v_posterior_mcmc.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_file}")
    
    plt.show()
    
    return fig


def plot_corner(results, save_path='plots'):
    """
    Corner plot showing parameter correlations.
    
    Parameters
    ----------
    results : dict
        MCMC results
    save_path : str
        Directory to save plots
    """
    try:
        import corner
    except ImportError:
        print("Installing corner...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "corner"])
        import corner
    
    os.makedirs(save_path, exist_ok=True)
    
    chains = results['chains']
    
    # Add Δv as a derived parameter
    delta_v = chains[:, 0] - chains[:, 1]
    samples = np.column_stack([chains, delta_v])
    
    labels = [
        r'$\mu_{void}$ [km/s]',
        r'$\mu_{cluster}$ [km/s]',
        r'$\sigma_{int}$ [km/s]',
        r'$\Delta v$ [km/s]'
    ]
    
    # Create corner plot
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        truths=[None, None, None, CGC_PREDICTED_DELTA_V],
        truth_color='green'
    )
    
    fig.suptitle('CGC Dwarf Test - Parameter Posterior', fontsize=14, y=1.02)
    
    save_file = os.path.join(save_path, 'dwarf_corner_plot.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_file}")
    
    plt.show()
    
    return fig


def plot_model_comparison(results, save_path='plots'):
    """
    Visual comparison of CGC vs ΛCDM predictions with data.
    
    Parameters
    ----------
    results : dict
        MCMC results
    save_path : str
        Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Left panel: Violin plot of posteriors ===
    ax1 = axes[0]
    
    data_violin = [
        results['mu_void_samples'],
        results['mu_cluster_samples']
    ]
    
    parts = ax1.violinplot(data_violin, positions=[0, 1], showmeans=True, showmedians=True)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    
    parts['bodies'][0].set_facecolor('blue')
    parts['bodies'][1].set_facecolor('red')
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Void Dwarfs', 'Cluster Dwarfs'])
    ax1.set_ylabel('Rotation Velocity [km/s]')
    ax1.set_title('Posterior Distributions of Mean Velocities')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add labels
    ax1.text(0, results['mu_void'] + 10, 
             f"μ = {results['mu_void']:.1f} ± {results['mu_void_err']:.1f}",
             ha='center', fontsize=10)
    ax1.text(1, results['mu_cluster'] + 10,
             f"μ = {results['mu_cluster']:.1f} ± {results['mu_cluster_err']:.1f}",
             ha='center', fontsize=10)
    
    # === Right panel: Δv comparison ===
    ax2 = axes[1]
    
    # Predictions
    models = ['ΛCDM', 'CGC', 'Observed']
    values = [LCDM_PREDICTED_DELTA_V, CGC_PREDICTED_DELTA_V, results['delta_v_mean']]
    errors = [0.5, CGC_PREDICTED_ERROR, results['delta_v_std']]
    colors = ['gray', 'green', 'blue']
    
    y_pos = np.arange(len(models))
    
    for i, (model, val, err, color) in enumerate(zip(models, values, errors, colors)):
        ax2.errorbar(val, i, xerr=err, fmt='o', markersize=12, 
                     color=color, capsize=8, capthick=2, linewidth=2,
                     label=f'{model}: {val:.1f} ± {err:.1f} km/s')
    
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Δv = V$_{void}$ - V$_{cluster}$ [km/s]')
    ax2.set_title('Velocity Difference: Predictions vs Observation')
    ax2.set_xlim(-10, 25)
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=9)
    
    # Add verdict
    if results['p_cgc'] > 0.5:
        verdict = "CGC FAVORED"
        verdict_color = 'green'
    elif results['p_lcdm'] > 0.5:
        verdict = "ΛCDM FAVORED"
        verdict_color = 'gray'
    else:
        verdict = "INCONCLUSIVE"
        verdict_color = 'orange'
    
    ax2.text(0.95, 0.05, verdict, transform=ax2.transAxes, 
             fontsize=14, fontweight='bold', color=verdict_color,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'model_comparison_mcmc.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_file}")
    
    plt.show()
    
    return fig


def plot_trace(results, save_path='plots'):
    """
    Trace plots to check MCMC convergence.
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    chains = results['chains']
    delta_v = chains[:, 0] - chains[:, 1]
    
    params = [chains[:, 0], chains[:, 1], chains[:, 2], delta_v]
    names = [r'$\mu_{void}$', r'$\mu_{cluster}$', r'$\sigma_{int}$', r'$\Delta v$']
    
    for ax, param, name in zip(axes, params, names):
        ax.plot(param, alpha=0.7, linewidth=0.5)
        ax.axhline(np.mean(param), color='red', linestyle='--', label='Mean')
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    
    axes[-1].set_xlabel('Sample')
    axes[0].set_title('MCMC Trace Plots - Convergence Check')
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'mcmc_trace_dwarf.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_file}")
    
    plt.show()
    
    return fig


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_mcmc_analysis():
    """
    Complete MCMC analysis pipeline for CGC dwarf test.
    """
    from filter_samples import prepare_samples
    
    print("=" * 70)
    print("CGC VOID vs CLUSTER DWARF TEST - MCMC ANALYSIS")
    print("=" * 70)
    
    # Load and filter data
    print("\n[1/4] Loading and filtering data...")
    samples = prepare_samples()
    
    df_void = samples['void']
    df_cluster = samples['cluster']
    
    # Extract velocities and errors
    v_void = df_void['V_rot'].values
    v_void_err = df_void['V_rot_err'].values
    v_cluster = df_cluster['V_rot'].values
    v_cluster_err = df_cluster['V_rot_err'].values
    
    print(f"  Void dwarfs:    N = {len(v_void)}")
    print(f"  Cluster dwarfs: N = {len(v_cluster)}")
    
    # Run MCMC
    print("\n[2/4] Running MCMC sampling...")
    results = run_dwarf_mcmc(
        v_void, v_void_err,
        v_cluster, v_cluster_err,
        n_walkers=32,
        n_steps=3000,
        n_burn=500,
        verbose=True
    )
    
    # Generate plots
    print("\n[3/4] Generating plots...")
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    plot_delta_v_posterior(results, plots_dir)
    plot_corner(results, plots_dir)
    plot_model_comparison(results, plots_dir)
    plot_trace(results, plots_dir)
    
    # Save results
    print("\n[4/4] Saving results...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    np.savez(
        os.path.join(results_dir, 'cgc_dwarf_mcmc_results.npz'),
        chains=results['chains'],
        delta_v_samples=results['delta_v_samples'],
        delta_v_mean=results['delta_v_mean'],
        delta_v_std=results['delta_v_std'],
        ci_68=results['ci_68'],
        ci_95=results['ci_95'],
        bayes_factor=results['bayes_factor_cgc_lcdm'],
        p_cgc=results['p_cgc'],
        p_lcdm=results['p_lcdm']
    )
    print(f"✓ Saved: {results_dir}/cgc_dwarf_mcmc_results.npz")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
    Velocity Difference (Void - Cluster):
    ═══════════════════════════════════════════════════════════════════
      Δv = {results['delta_v_mean']:.2f} ± {results['delta_v_std']:.2f} km/s
      
      95% Credible Interval: [{results['ci_95'][0]:.2f}, {results['ci_95'][1]:.2f}] km/s
    
    Theory Comparison:
    ═══════════════════════════════════════════════════════════════════
      CGC prediction:  +12 ± 3 km/s
      ΛCDM prediction:   0 ± 0.5 km/s
      
      P(CGC-consistent):  {results['p_cgc']:.3f}
      P(ΛCDM-consistent): {results['p_lcdm']:.3f}
      Bayes Factor (CGC/ΛCDM): {results['bayes_factor_cgc_lcdm']:.2f}
    
    Verdict:
    ═══════════════════════════════════════════════════════════════════
    """)
    
    if results['bayes_factor_cgc_lcdm'] > 3:
        print("      ✓✓✓  STRONG EVIDENCE FOR CGC  ✓✓✓")
    elif results['bayes_factor_cgc_lcdm'] > 1:
        print("      ✓  MODERATE EVIDENCE FOR CGC")
    elif results['bayes_factor_cgc_lcdm'] > 0.33:
        print("      ???  INCONCLUSIVE - NEED MORE DATA")
    else:
        print("      ✗  EVIDENCE FAVORS ΛCDM")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == '__main__':
    results = run_full_mcmc_analysis()
