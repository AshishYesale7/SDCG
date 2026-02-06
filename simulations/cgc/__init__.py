"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ██████╗  ██████╗    ████████╗██╗  ██╗███████╗ ██████╗ ██████╗ ██╗   ██╗ ║
║    ██╔════╝██╔════╝ ██╔════╝    ╚══██╔══╝██║  ██║██╔════╝██╔═══██╗██╔══██╗╚██╗ ██╔╝ ║
║    ██║     ██║  ███╗██║            ██║   ███████║█████╗  ██║   ██║██████╔╝ ╚████╔╝  ║
║    ██║     ██║   ██║██║            ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗  ╚██╔╝   ║
║    ╚██████╗╚██████╔╝╚██████╗       ██║   ██║  ██║███████╗╚██████╔╝██║  ██║   ██║    ║
║     ╚═════╝ ╚═════╝  ╚═════╝       ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ║
║                                                                              ║
║                  Casimir-Gravity Crossover (CGC) Theory                      ║
║                      Cosmological Analysis Framework                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CGC Theory Analysis Package
===========================

A comprehensive framework for testing the Casimir-Gravity Crossover (CGC) theory
against real cosmological observations including:

    • Planck 2018 CMB power spectrum (TT, TE, EE)
    • BOSS DR12/16 BAO measurements
    • Pantheon+ Type Ia supernovae
    • Lyman-α forest flux power spectrum
    • RSD growth rate measurements (fσ8)
    • SH0ES Cepheid-calibrated H0

The CGC theory proposes modifications to gravity at large scales arising from
quantum vacuum fluctuations (Casimir-like effects), potentially resolving the
Hubble tension and S8 discrepancies.

Modules
-------
config : Configuration, paths, and constants
parameters : CGC and cosmological parameter definitions
data_loader : Load real and mock cosmological datasets
likelihoods : Likelihood functions for all probes
mcmc : Markov Chain Monte Carlo sampling
nested_sampling : Nested sampling for Bayesian evidence
analysis : Statistical analysis and model comparison
plotting : Publication-quality visualization
theory : CGC theoretical predictions

Usage
-----
>>> from cgc import CGCParameters, run_analysis
>>> params = CGCParameters()
>>> results = run_analysis(use_real_data=True, n_steps=10000)

Author: CGC Collaboration
Version: 2.0.0
Date: January 2026
"""

__version__ = "2.0.0"
__author__ = "CGC Collaboration"

# =============================================================================
# PACKAGE IMPORTS
# =============================================================================

# Configuration
from .config import (
    PATHS,
    CONSTANTS,
    PLANCK_BASELINE,
    MCMC_DEFAULTS,
    setup_directories
)

# Parameters
from .parameters import (
    CGCParameters,
    get_parameter_names,
    get_parameter_bounds,
    get_latex_labels
)

# Data Loading
from .data_loader import (
    load_real_data,
    load_mock_data,
    load_pantheon_sne,
    load_lyalpha_data,
    DataLoader
)

# Likelihoods
from .likelihoods import (
    log_likelihood,
    log_likelihood_cmb,
    log_likelihood_bao,
    log_likelihood_sne,
    log_likelihood_lyalpha,
    log_likelihood_growth,
    log_likelihood_h0,
    log_prior
)

# MCMC
from .mcmc import (
    run_mcmc,
    MCMCSampler
)

# Nested Sampling
from .nested_sampling import (
    run_nested_sampling,
    compute_bayesian_evidence
)

# Analysis
from .analysis import (
    analyze_chains,
    compute_tension_metrics,
    compute_model_comparison,
    compute_derived_parameters,
    generate_summary_report
)

# Plotting
from .plotting import (
    plot_all,
    PlotGenerator
)

# Theory
from .theory import (
    CGCTheory,
    create_theory_from_params,
    compare_lcdm_cgc
)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_analysis(use_real_data=True, n_steps=1000, n_walkers=32, 
                 use_sne=False, use_lyalpha=False, generate_plots=True,
                 verbose=True):
    """
    Run complete CGC analysis pipeline.
    
    This is the main entry point for running a full CGC theory analysis.
    It loads data, runs MCMC, computes statistics, and generates plots.
    
    Parameters
    ----------
    use_real_data : bool, default=True
        If True, use real cosmological observations.
        If False, use synthetic mock data for testing.
    
    n_steps : int, default=1000
        Number of MCMC steps per walker.
        For publication: use 10000+ steps.
    
    n_walkers : int, default=32
        Number of MCMC walkers (should be ≥ 2 × n_params).
    
    use_sne : bool, default=False
        Include Pantheon+ supernovae in the likelihood.
    
    use_lyalpha : bool, default=False
        Include Lyman-α forest data (requires special treatment).
    
    generate_plots : bool, default=True
        Generate all analysis plots.
    
    verbose : bool, default=True
        Print progress messages.
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'chains': MCMC chains (n_samples × n_params)
        - 'flat_chains': Flattened chains
        - 'means': Parameter means
        - 'stds': Parameter standard deviations
        - 'H0_mean': Hubble constant estimate
        - 'S8_mean': S8 estimate
        - 'tension_metrics': Tension reduction statistics
    
    Examples
    --------
    Quick test run:
    >>> results = run_analysis(use_real_data=False, n_steps=500)
    
    Publication-quality run:
    >>> results = run_analysis(use_real_data=True, n_steps=10000, 
    ...                        use_sne=True, use_lyalpha=True)
    
    See Also
    --------
    run_mcmc : Lower-level MCMC function
    run_nested_sampling : For Bayesian evidence computation
    """
    # Setup directories
    setup_directories()
    
    # Load data
    if use_real_data:
        data = load_real_data(verbose=verbose)
    else:
        data = load_mock_data(verbose=verbose)
    
    # Run MCMC
    results = run_mcmc(
        data, 
        n_walkers=n_walkers, 
        n_steps=n_steps,
        include_sne=use_sne,
        include_lyalpha=use_lyalpha,
        verbose=verbose
    )
    
    # Analyze results
    chains = results['flat_chains']
    param_stats = analyze_chains(chains)
    tensions = compute_tension_metrics(chains, data)
    
    # Add to results
    results['param_stats'] = param_stats
    results['tensions'] = tensions
    results['data'] = data
    
    # Generate plots
    if generate_plots:
        plot_all(results, data)
    
    return results


# Module-level docstrings for IDE support
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Configuration
    'PATHS',
    'CONSTANTS',
    'PLANCK_BASELINE',
    'MCMC_DEFAULTS',
    'setup_directories',
    
    # Parameters
    'CGCParameters',
    'get_parameter_names',
    'get_parameter_bounds',
    'get_latex_labels',
    
    # Data
    'load_real_data',
    'load_mock_data',
    'load_pantheon_sne',
    'load_lyalpha_data',
    'DataLoader',
    
    # Likelihoods
    'log_likelihood',
    'log_likelihood_cmb',
    'log_likelihood_bao',
    'log_likelihood_sne',
    'log_likelihood_lyalpha',
    'log_likelihood_growth',
    'log_likelihood_h0',
    'log_prior',
    
    # MCMC
    'run_mcmc',
    'MCMCSampler',
    
    # Nested Sampling
    'run_nested_sampling',
    'compute_bayesian_evidence',
    
    # Analysis
    'analyze_chains',
    'compute_tension_metrics',
    'compute_model_comparison',
    'compute_derived_parameters',
    'generate_summary_report',
    
    # Plotting
    'plot_all',
    'PlotGenerator',
    
    # Theory
    'CGCTheory',
    'create_theory_from_params',
    'compare_lcdm_cgc',
    
    # Convenience
    'run_analysis',
]
