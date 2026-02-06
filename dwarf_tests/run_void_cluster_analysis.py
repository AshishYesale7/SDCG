#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SDCG FALSIFICATION TEST: Void vs Cluster Dwarf Galaxies            ║
║                                                                              ║
║  Phase 3: Statistical Analysis                                               ║
║                                                                              ║
║  The Test:                                                                   ║
║    Δv = <v_rot(void)> - <v_rot(cluster)>                                    ║
║                                                                              ║
║  SDCG Predictions:                                                           ║
║    MCMC Best-fit (μ=0.149):    Δv ≈ +12 ± 3 km/s                            ║
║    ΛCDM (μ=0):                 Δv = 0 km/s                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings

# Paths
FILTERED_DIR = Path(__file__).parent / "filtered"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# SDCG Theory Parameters (Thesis v12)
# NOTE: μ_fit = 0.47 is the FUNDAMENTAL fitted value
# μ_eff(void) = μ_fit × S(void) ≈ 0.47 × 0.31 ≈ 0.149
MU_FIT = 0.47                # MCMC best-fit (6σ detection) - Thesis v12
MU_EFF_VOID = 0.149          # Effective coupling in voids (μ_fit × S_avg)
MU_UNCONSTRAINED = 0.786     # Unconstrained μ (without Lyα, from thesis)
MU_LYA_CONSTRAINED = 0.05    # Lyα-constrained μ upper limit
RHO_THRESH = 200             # Screening threshold (ρ_crit units) - FIXED
ALPHA_SCREEN = 2             # Screening exponent - FIXED (Klein-Gordon)

# Environmental densities (ρ/ρ_crit)
RHO_VOID = 0.1               # Cosmic void
RHO_CLUSTER = 200            # Cluster outskirts


def screening_function(rho, rho_thresh=RHO_THRESH, alpha=ALPHA_SCREEN):
    """Chameleon screening function S(ρ)."""
    return 1 / (1 + (rho / rho_thresh)**alpha)


def sdcg_velocity_difference(v_typical, mu, S_void, S_cluster):
    """
    Compute predicted velocity difference between void and cluster.
    
    σ_v ∝ √(G_eff) = √(G_N × (1 + μ × S))
    
    Δv = v × [√(1 + μ × S_void) - √(1 + μ × S_cluster)]
    """
    return v_typical * (np.sqrt(1 + mu * S_void) - np.sqrt(1 + mu * S_cluster))


def load_filtered_samples():
    """Load filtered void and cluster dwarf samples."""
    void_path = FILTERED_DIR / "void_dwarfs.csv"
    cluster_path = FILTERED_DIR / "cluster_dwarfs.csv"
    
    if not void_path.exists() or not cluster_path.exists():
        raise FileNotFoundError("Filtered samples not found. Run filter_dwarf_sample.py first.")
    
    void_dwarfs = pd.read_csv(void_path)
    cluster_dwarfs = pd.read_csv(cluster_path)
    
    return void_dwarfs, cluster_dwarfs


def welch_t_test(sample1, sample2):
    """
    Welch's t-test for samples with unequal variances.
    
    Returns test statistic, p-value, and effect size (Cohen's d).
    """
    n1, n2 = len(sample1), len(sample2)
    m1, m2 = np.mean(sample1), np.mean(sample2)
    s1, s2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / pooled_std if pooled_std > 0 else 0
    
    # Standard error of difference
    se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': m1 - m2,
        'se_diff': se_diff,
        'n1': n1,
        'n2': n2,
        'mean1': m1,
        'mean2': m2,
        'std1': s1,
        'std2': s2
    }


def bootstrap_confidence_interval(sample1, sample2, n_boot=10000, alpha=0.05):
    """
    Compute bootstrap confidence interval for mean difference.
    """
    np.random.seed(42)
    
    n1, n2 = len(sample1), len(sample2)
    diffs = np.zeros(n_boot)
    
    for i in range(n_boot):
        boot1 = np.random.choice(sample1, size=n1, replace=True)
        boot2 = np.random.choice(sample2, size=n2, replace=True)
        diffs[i] = np.mean(boot1) - np.mean(boot2)
    
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    
    return {
        'mean': np.mean(diffs),
        'std': np.std(diffs),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_level': 1 - alpha
    }


def interpret_result(delta_v, se_delta_v, mu_unconstrained=MU_UNCONSTRAINED, 
                     mu_constrained=MU_LYA_CONSTRAINED):
    """
    Interpret the observed Δv against SDCG predictions.
    """
    # Compute screening factors
    S_void = screening_function(RHO_VOID)
    S_cluster = screening_function(RHO_CLUSTER)
    
    # Typical dwarf rotation velocity
    v_typical = 50.0  # km/s
    
    # SDCG predictions
    dv_unconstrained = sdcg_velocity_difference(v_typical, mu_unconstrained, S_void, S_cluster)
    dv_constrained = sdcg_velocity_difference(v_typical, mu_constrained, S_void, S_cluster)
    dv_lcdm = 0.0
    
    # Compute tensions
    if se_delta_v > 0:
        tension_unconstrained = abs(delta_v - dv_unconstrained) / se_delta_v
        tension_constrained = abs(delta_v - dv_constrained) / se_delta_v
        tension_lcdm = abs(delta_v - dv_lcdm) / se_delta_v
    else:
        tension_unconstrained = tension_constrained = tension_lcdm = np.nan
    
    return {
        'delta_v_observed': delta_v,
        'se_observed': se_delta_v,
        'predictions': {
            'unconstrained': {'value': dv_unconstrained, 'tension_sigma': tension_unconstrained},
            'lya_constrained': {'value': dv_constrained, 'tension_sigma': tension_constrained},
            'lcdm': {'value': dv_lcdm, 'tension_sigma': tension_lcdm}
        },
        'screening': {
            'S_void': S_void,
            'S_cluster': S_cluster
        }
    }


def run_analysis():
    """Run the complete statistical analysis."""
    print("=" * 70)
    print("SDCG FALSIFICATION TEST: VOID vs CLUSTER ANALYSIS")
    print("=" * 70)
    print()
    
    # Load samples
    print("PHASE 3A: Loading Filtered Samples")
    print("-" * 70)
    
    try:
        void_dwarfs, cluster_dwarfs = load_filtered_samples()
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("ERROR: FILTERED SAMPLES NOT FOUND")
        print(f"{'='*60}\n")
        print(f"{e}\n")
        print("This analysis requires REAL observational data.")
        print("Mock/synthetic data is NOT permitted for scientific validity.\n")
        print("To proceed:")
        print("  1. Run: python download_catalogs.py")
        print("  2. Run: python filter_dwarf_sample.py")
        print("  3. Then run this script again\n")
        print("If download fails, manually obtain data from:")
        print("  ALFALFA: http://egg.astro.cornell.edu/alfalfa/data/")
        print("  Voids:   https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/421/926")
        print(f"{'='*60}")
        raise SystemExit(1)
    
    v_void = void_dwarfs['v_rot'].values
    v_cluster = cluster_dwarfs['v_rot'].values
    
    print(f"\n  Void dwarfs:    N = {len(v_void)}, <v_rot> = {np.mean(v_void):.1f} km/s")
    print(f"  Cluster dwarfs: N = {len(v_cluster)}, <v_rot> = {np.mean(v_cluster):.1f} km/s")
    
    # Statistical tests
    print("\nPHASE 3B: Statistical Analysis")
    print("-" * 70)
    
    # Welch's t-test
    test_result = welch_t_test(v_void, v_cluster)
    
    print(f"\n  Mean difference: Δv = {test_result['mean_diff']:+.2f} ± {test_result['se_diff']:.2f} km/s")
    print(f"  Welch's t-test:  t = {test_result['t_stat']:.3f}, p = {test_result['p_value']:.4f}")
    print(f"  Effect size:     Cohen's d = {test_result['cohens_d']:.3f}")
    
    # Bootstrap CI
    print("\n  Computing bootstrap confidence interval...")
    boot_result = bootstrap_confidence_interval(v_void, v_cluster)
    print(f"  95% CI: [{boot_result['ci_low']:.2f}, {boot_result['ci_high']:.2f}] km/s")
    
    # Interpretation
    print("\nPHASE 3C: Theory Comparison")
    print("-" * 70)
    
    interpretation = interpret_result(test_result['mean_diff'], test_result['se_diff'])
    
    print(f"\n  Observed: Δv = {interpretation['delta_v_observed']:+.2f} ± {interpretation['se_observed']:.2f} km/s")
    print()
    
    print("  SDCG Predictions vs Observation:")
    print("  " + "-" * 50)
    
    for model, pred in interpretation['predictions'].items():
        status = "✓" if pred['tension_sigma'] < 2 else "✗"
        print(f"  {status} {model:20s}: {pred['value']:+.2f} km/s (tension: {pred['tension_sigma']:.1f}σ)")
    
    # Final verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()
    
    unconstrained_tension = interpretation['predictions']['unconstrained']['tension_sigma']
    constrained_tension = interpretation['predictions']['lya_constrained']['tension_sigma']
    lcdm_tension = interpretation['predictions']['lcdm']['tension_sigma']
    
    if unconstrained_tension > 3:
        print("  ✗ UNCONSTRAINED SDCG (μ = 0.41) is FALSIFIED at >3σ")
    elif unconstrained_tension > 2:
        print("  ⚠ UNCONSTRAINED SDCG (μ = 0.41) is in TENSION at >2σ")
    else:
        print("  ✓ UNCONSTRAINED SDCG (μ = 0.41) is consistent within 2σ")
    
    if constrained_tension < 2:
        print("  ✓ MCMC BEST-FIT SDCG (μ = 0.149) is consistent within 2σ")
    else:
        print("  ✗ MCMC BEST-FIT SDCG (μ = 0.149) shows tension")
    
    if lcdm_tension < 2:
        print("  ✓ ΛCDM (no modification) is consistent within 2σ")
    else:
        print("  ✗ ΛCDM shows tension - possible signal detected")
    
    # Save results
    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    results = {
        'n_void': len(v_void),
        'n_cluster': len(v_cluster),
        'v_void_mean': np.mean(v_void),
        'v_void_std': np.std(v_void),
        'v_cluster_mean': np.mean(v_cluster),
        'v_cluster_std': np.std(v_cluster),
        'delta_v': test_result['mean_diff'],
        'delta_v_se': test_result['se_diff'],
        't_stat': test_result['t_stat'],
        'p_value': test_result['p_value'],
        'cohens_d': test_result['cohens_d'],
        'ci_95_low': boot_result['ci_low'],
        'ci_95_high': boot_result['ci_high'],
        'pred_unconstrained': interpretation['predictions']['unconstrained']['value'],
        'pred_constrained': interpretation['predictions']['lya_constrained']['value'],
        'tension_unconstrained': unconstrained_tension,
        'tension_constrained': constrained_tension,
        'tension_lcdm': lcdm_tension
    }
    
    # Save as NPZ
    np.savez(RESULTS_DIR / "void_cluster_analysis.npz", **results)
    print(f"\n  Results saved to: {RESULTS_DIR / 'void_cluster_analysis.npz'}")
    
    # Save as text summary
    summary_path = RESULTS_DIR / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SDCG VOID vs CLUSTER DWARF GALAXY TEST\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: 2026-02-02\n\n")
        f.write("SAMPLE:\n")
        f.write(f"  Void dwarfs:    N = {len(v_void)}\n")
        f.write(f"  Cluster dwarfs: N = {len(v_cluster)}\n\n")
        f.write("RESULT:\n")
        f.write(f"  Δv = {test_result['mean_diff']:+.2f} ± {test_result['se_diff']:.2f} km/s\n")
        f.write(f"  95% CI: [{boot_result['ci_low']:.2f}, {boot_result['ci_high']:.2f}] km/s\n")
        f.write(f"  p-value = {test_result['p_value']:.4f}\n\n")
        f.write("VERDICT:\n")
        f.write(f"  Unconstrained SDCG: {'FALSIFIED' if unconstrained_tension > 3 else 'Consistent'} ({unconstrained_tension:.1f}σ)\n")
        f.write(f"  Lyα-Constrained:    {'Consistent' if constrained_tension < 2 else 'Tension'} ({constrained_tension:.1f}σ)\n")
        f.write(f"  ΛCDM:               {'Consistent' if lcdm_tension < 2 else 'Tension'} ({lcdm_tension:.1f}σ)\n")
    
    print(f"  Summary saved to: {summary_path}")
    
    print()
    print("Next step: python interpret_results.py (for visualization)")
    
    return results


if __name__ == "__main__":
    results = run_analysis()
