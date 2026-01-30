#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CGC Cosmological Analysis - Main Entry Point                    ║
║                                                                              ║
║  Casimir-Gravity Crossover (CGC) Theory Parameter Estimation                 ║
║  Using Real Cosmological Data + MCMC/Nested Sampling                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This script performs:                                                       ║
║    1. Load real cosmological data (Planck, BOSS, RSD, SNe)                  ║
║    2. Run MCMC or nested sampling for CGC parameters                        ║
║    3. Compute model comparison (BIC/AIC/Bayes factors)                      ║
║    4. Analyze cosmological tensions (H0, S8)                                ║
║    5. Generate publication-quality plots                                     ║
║    6. Save comprehensive results                                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Usage:                                                                      ║
║    python run_cgc_analysis.py --real --steps 5000                           ║
║    python run_cgc_analysis.py --mock --quick                                ║
║    python run_cgc_analysis.py --nested --nlive 500                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import os
import numpy as np
from datetime import datetime
import warnings

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import CGC modules
from cgc.config import PATHS, MCMC_DEFAULTS, setup_directories
from cgc.parameters import CGCParameters
from cgc.data_loader import DataLoader, load_real_data, load_mock_data
from cgc.likelihoods import log_likelihood, log_prior
from cgc.mcmc import run_mcmc, MCMCSampler
from cgc.analysis import (
    analyze_chains, 
    compute_tension_metrics,
    compute_model_comparison,
    generate_summary_report,
    run_full_analysis
)
from cgc.plotting import plot_all, PlotGenerator


# =============================================================================
# CLI ARGUMENT PARSER
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for CGC analysis.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='CGC Theory MCMC Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with mock data
  python run_cgc_analysis.py --mock --quick

  # Full analysis with real data
  python run_cgc_analysis.py --real --steps 10000

  # Extended run for publication
  python run_cgc_analysis.py --real --steps 50000 --walkers 64

  # Nested sampling for model comparison
  python run_cgc_analysis.py --real --nested --nlive 500

  # Include all data (SNe + Lyman-α)
  python run_cgc_analysis.py --real --include-sne --include-lyalpha
        """
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--real', action='store_true',
        help='Use real cosmological data (Planck, BOSS, RSD, etc.)'
    )
    data_group.add_argument(
        '--mock', action='store_true',
        help='Use synthetic mock data for testing'
    )
    
    # Sampling method
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        '--mcmc', action='store_true', default=True,
        help='Use MCMC sampling (default)'
    )
    method_group.add_argument(
        '--nested', action='store_true',
        help='Use nested sampling for Bayesian evidence'
    )
    
    # MCMC settings
    parser.add_argument(
        '--steps', type=int, default=MCMC_DEFAULTS.get('n_steps_standard', 1000),
        help=f'Number of MCMC steps (default: {MCMC_DEFAULTS.get("n_steps_standard", 1000)})'
    )
    parser.add_argument(
        '--walkers', type=int, default=MCMC_DEFAULTS.get('n_walkers', 32),
        help=f'Number of MCMC walkers (default: {MCMC_DEFAULTS.get("n_walkers", 32)})'
    )
    parser.add_argument(
        '--burn', type=int, default=200,
        help='Burn-in steps to discard (default: 200)'
    )
    
    # Nested sampling settings
    parser.add_argument(
        '--nlive', type=int, default=500,
        help='Number of live points for nested sampling (default: 500)'
    )
    parser.add_argument(
        '--dlogz', type=float, default=0.1,
        help='dlogz stopping criterion for nested sampling (default: 0.1)'
    )
    
    # Data options
    parser.add_argument(
        '--include-sne', action='store_true',
        help='Include Pantheon+ supernovae in likelihood'
    )
    parser.add_argument(
        '--include-lyalpha', action='store_true',
        help='Include Lyman-α forest data in likelihood'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test run (500 steps, 16 walkers)'
    )
    
    # Output options
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save results to disk'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Custom output directory'
    )
    
    args = parser.parse_args()
    
    # Apply quick mode settings
    if args.quick:
        args.steps = 500
        args.walkers = 24  # Must be > 2 * n_params (10 params, so > 20)
        args.burn = 100
        args.nlive = 50
    
    return args


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(args: argparse.Namespace) -> dict:
    """
    Main analysis function.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    
    Returns
    -------
    dict
        Analysis results dictionary.
    """
    # Setup directories
    setup_directories()
    
    # =========================================================================
    # HEADER
    # =========================================================================
    
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  CGC THEORY: COSMOLOGICAL PARAMETER ESTIMATION".center(68) + "║")
    print("║" + "  Casimir-Gravity Crossover Analysis".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╠" + "═"*68 + "╣")
    print(f"║  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<55} ║")
    print(f"║  Data source: {'REAL' if args.real else 'MOCK':<52} ║")
    print(f"║  Method: {'Nested Sampling' if args.nested else 'MCMC':<56} ║")
    print("╚" + "═"*68 + "╝")
    print("")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    if args.real:
        data = load_real_data(
            verbose=args.verbose,
            include_sne=args.include_sne,
            include_lyalpha=args.include_lyalpha
        )
        print(f"\n✓ Loaded real cosmological data")
    else:
        data = load_mock_data(
            verbose=args.verbose,
            include_sne=args.include_sne,
            include_lyalpha=args.include_lyalpha
        )
        print(f"\n✓ Generated mock data")
    
    # Summary of data
    if 'cmb' in data:
        print(f"  • CMB: {len(data['cmb']['ell'])} multipoles")
    if 'bao' in data:
        print(f"  • BAO: {len(data['bao']['z'])} redshift bins")
    if 'growth' in data:
        print(f"  • Growth: {len(data['growth']['z'])} RSD measurements")
    if 'H0' in data:
        print(f"  • H0: Planck + SH0ES")
    if 'sne' in data:
        print(f"  • SNe: {len(data['sne']['z'])} Pantheon+ supernovae")
    if 'lyalpha' in data:
        print(f"  • Lyman-α: {len(data['lyalpha']['z'])} flux power spectrum points")
    
    # =========================================================================
    # RUN SAMPLING
    # =========================================================================
    
    if args.nested:
        # Nested sampling
        print("\n" + "="*70)
        print("NESTED SAMPLING")
        print("="*70)
        print(f"Live points: {args.nlive}")
        print(f"dlogz threshold: {args.dlogz}")
        
        from cgc.nested_sampling import run_nested_sampling
        
        results = run_nested_sampling(
            data,
            nlive=args.nlive,
            dlogz=args.dlogz,
            compare_lcdm=True,
            include_sne=args.include_sne,
            include_lyalpha=args.include_lyalpha,
            verbose=args.verbose
        )
        
        # Extract posterior samples for analysis
        chains = results['posterior_samples']
        
    else:
        # MCMC sampling
        print("\n" + "="*70)
        print("MCMC SAMPLING")
        print("="*70)
        print(f"Walkers: {args.walkers}")
        print(f"Steps: {args.steps}")
        print(f"Burn-in: {args.burn}")
        
        sampler, chains = run_mcmc(
            data,
            n_walkers=args.walkers,
            n_steps=args.steps,
            include_sne=args.include_sne,
            include_lyalpha=args.include_lyalpha,
            verbose=args.verbose
        )
        
        results = {'sampler': sampler, 'flat_chains': chains, 'chains': chains}
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    print("\n" + "="*70)
    print("ANALYZING RESULTS")
    print("="*70)
    
    # Parameter statistics
    param_stats = analyze_chains(chains)
    
    print("\nParameter constraints (mean ± std):")
    print("-"*50)
    
    param_labels = [
        ('omega_b', 'ω_b'),
        ('omega_cdm', 'ω_cdm'),
        ('h', 'h'),
        ('mu', 'μ (CGC)'),
        ('n_g', 'n_g (CGC)'),
        ('z_trans', 'z_trans (CGC)'),
    ]
    
    for key, label in param_labels:
        if key in param_stats:
            s = param_stats[key]
            print(f"  {label:<15}: {s.mean:.6f} ± {s.std:.6f}")
    
    # Tension metrics
    tensions = compute_tension_metrics(chains, data)
    
    print("\n" + "-"*50)
    print("Derived Parameters:")
    print("-"*50)
    print(f"  H0 = {tensions['H0_mean']:.2f} ± {tensions['H0_std']:.2f} km/s/Mpc")
    print(f"  S8 = {tensions['S8_mean']:.4f} ± {tensions['S8_std']:.4f}")
    
    print("\n" + "-"*50)
    print("Tension Analysis:")
    print("-"*50)
    print(f"  H0 tension (ΛCDM): {tensions['H0_tension_lcdm']:.1f}σ")
    print(f"  H0 tension (CGC):  {tensions['H0_tension_cgc']:.1f}σ")
    print(f"  → Reduction: {tensions['H0_reduction']:.0f}%")
    print(f"")
    print(f"  S8 tension (ΛCDM): {tensions['S8_tension_lcdm']:.1f}σ")
    print(f"  S8 tension (CGC):  {tensions['S8_tension_cgc']:.1f}σ")
    print(f"  → Reduction: {tensions['S8_reduction']:.0f}%")
    
    # Model comparison
    if args.nested and 'bayes_factor' in results:
        print("\n" + "-"*50)
        print("Bayesian Model Comparison:")
        print("-"*50)
        bf = results['bayes_factor']
        print(f"  log Bayes factor: {bf['log_bayes_factor']:.2f}")
        print(f"  Interpretation: {bf['interpretation']}")
    else:
        comparison = compute_model_comparison(results, data)
        print("\n" + "-"*50)
        print("Information Criteria:")
        print("-"*50)
        print(f"  ΔBIC = {comparison['delta_BIC']:.1f}")
        print(f"  ΔAIC = {comparison['delta_AIC']:.1f}")
        print(f"  χ²/dof = {comparison['chi2_reduced']:.2f}")
        print(f"  Interpretation: {comparison['interpretation']}")
    
    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    
    if not args.no_plots:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        plots = plot_all(results, data)
        print(f"\n✓ Generated {len(plots)} plots")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    if not args.no_save:
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        output_dir = args.output_dir or PATHS['results']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save chains
        chains_path = os.path.join(output_dir, f'cgc_chains_{timestamp}.npy')
        np.save(chains_path, chains)
        print(f"  ✓ Chains: {chains_path}")
        
        # Save full results
        results_path = os.path.join(output_dir, f'cgc_results_{timestamp}.npz')
        np.savez(results_path, 
                 chains=chains,
                 means=np.mean(chains, axis=0),
                 stds=np.std(chains, axis=0),
                 H0_mean=tensions['H0_mean'],
                 H0_std=tensions['H0_std'],
                 S8_mean=tensions['S8_mean'],
                 S8_std=tensions['S8_std'])
        print(f"  ✓ Results: {results_path}")
        
        # Generate and save summary
        summary_path = os.path.join(output_dir, f'cgc_summary_{timestamp}.txt')
        generate_summary_report(chains, data, results, summary_path)
        print(f"  ✓ Summary: {summary_path}")
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + "  ANALYSIS COMPLETE".center(68) + "║")
    print("╠" + "═"*68 + "╣")
    print(f"║  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<55} ║")
    print(f"║  Samples: {len(chains):<56} ║")
    print(f"║  H0 = {tensions['H0_mean']:.2f} ± {tensions['H0_std']:.2f} km/s/Mpc" + " "*38 + "║")
    print(f"║  μ (CGC) = {param_stats['mu'].mean:.4f} ± {param_stats['mu'].std:.4f}" + " "*41 + "║")
    print("╚" + "═"*68 + "╝")
    print("")
    
    return {
        'chains': chains,
        'results': results,
        'param_stats': param_stats,
        'tensions': tensions,
        'data': data
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        results = run_analysis(args)
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
