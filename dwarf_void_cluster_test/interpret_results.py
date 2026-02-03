#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SDCG FALSIFICATION TEST: Void vs Cluster Dwarf Galaxies            ║
║                                                                              ║
║  Phase 4: Result Interpretation & Visualization                             ║
║                                                                              ║
║  Generates:                                                                  ║
║    1. Publication-ready figure comparing Δv to predictions                  ║
║    2. Detailed interpretation with physical context                         ║
║    3. LaTeX-formatted summary for thesis inclusion                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_results():
    """Load analysis results."""
    results_path = RESULTS_DIR / "void_cluster_analysis.npz"
    if not results_path.exists():
        raise FileNotFoundError("Results not found. Run run_void_cluster_analysis.py first.")
    
    data = np.load(results_path, allow_pickle=True)
    return {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data.files}


def create_figure(results):
    """Create publication-ready figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    delta_v = results['delta_v']
    se = results['delta_v_se']
    ci_low = results['ci_95_low']
    ci_high = results['ci_95_high']
    
    # Predictions
    pred_unc = results['pred_unconstrained']
    pred_con = results['pred_constrained']
    pred_lcdm = 0.0
    
    # Color scheme
    colors = {
        'observed': '#2196F3',      # Blue
        'unconstrained': '#F44336', # Red
        'constrained': '#4CAF50',   # Green
        'lcdm': '#9E9E9E'           # Gray
    }
    
    # Plot predictions as vertical bands
    band_width = 0.3
    
    # ΛCDM band (no effect)
    ax.axhspan(-2, 2, alpha=0.2, color=colors['lcdm'], label='ΛCDM (±2 km/s)')
    
    # Lyα-constrained prediction
    ax.axhline(pred_con, color=colors['constrained'], linestyle='--', linewidth=2,
               label=f'Lyα-constrained: +{pred_con:.1f} km/s')
    
    # Unconstrained prediction
    ax.axhline(pred_unc, color=colors['unconstrained'], linestyle=':', linewidth=2,
               label=f'Unconstrained: +{pred_unc:.1f} km/s')
    
    # Observed value with error bar
    ax.errorbar(0.5, delta_v, yerr=[[delta_v - ci_low], [ci_high - delta_v]],
                fmt='o', markersize=12, color=colors['observed'],
                capsize=8, capthick=2, elinewidth=2,
                label=f'Observed: {delta_v:+.1f} km/s')
    
    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(-20, 20)
    ax.set_xticks([])
    ax.set_ylabel('Δv = ⟨v_rot(void)⟩ - ⟨v_rot(cluster)⟩  [km/s]', fontsize=12)
    ax.set_title('SDCG Falsification Test: Void vs Cluster Dwarf Galaxies', fontsize=14, fontweight='bold')
    
    # Add interpretation zones
    ax.text(0.95, 15, '✗ Falsifies ΛCDM', ha='right', va='center', fontsize=10, color='gray')
    ax.text(0.95, -15, '✗ Falsifies ΛCDM', ha='right', va='center', fontsize=10, color='gray')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Add sample size annotation
    ax.text(0.02, 0.02, f"N_void = {results['n_void']:.0f}, N_cluster = {results['n_cluster']:.0f}",
            transform=ax.transAxes, fontsize=9, va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = PLOTS_DIR / "void_cluster_test.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    fig.savefig(PLOTS_DIR / "void_cluster_test.png", dpi=150, bbox_inches='tight')
    
    print(f"  Figure saved: {fig_path}")
    
    return fig


def generate_latex_table(results):
    """Generate LaTeX table for thesis inclusion."""
    latex = r"""
\begin{table}[h]
\centering
\caption{SDCG Void vs Cluster Dwarf Galaxy Test Results}
\label{tab:void_cluster_test}
\begin{tabular}{lcc}
\toprule
\textbf{Quantity} & \textbf{Value} & \textbf{Unit} \\
\midrule
\multicolumn{3}{l}{\textit{Sample}} \\
$N_{\rm void}$ & """ + f"{results['n_void']:.0f}" + r""" & -- \\
$N_{\rm cluster}$ & """ + f"{results['n_cluster']:.0f}" + r""" & -- \\
\midrule
\multicolumn{3}{l}{\textit{Observed}} \\
$\langle v_{\rm rot} \rangle_{\rm void}$ & """ + f"{results['v_void_mean']:.1f} \\pm {results['v_void_std']:.1f}" + r""" & km/s \\
$\langle v_{\rm rot} \rangle_{\rm cluster}$ & """ + f"{results['v_cluster_mean']:.1f} \\pm {results['v_cluster_std']:.1f}" + r""" & km/s \\
$\Delta v$ & """ + f"{results['delta_v']:+.2f} \\pm {results['delta_v_se']:.2f}" + r""" & km/s \\
\midrule
\multicolumn{3}{l}{\textit{Predictions}} \\
Unconstrained ($\mu = 0.41$) & """ + f"{results['pred_unconstrained']:+.1f}" + r""" & km/s \\
MCMC best-fit ($\mu = 0.149$) & """ + f"{results['pred_constrained']:+.1f}" + r""" & km/s \\
$\Lambda$CDM ($\mu = 0$) & $0$ & km/s \\
\midrule
\multicolumn{3}{l}{\textit{Tension}} \\
vs Unconstrained & """ + f"{results['tension_unconstrained']:.1f}$\\sigma$" + r""" & -- \\
vs MCMC best-fit & """ + f"{results['tension_constrained']:.1f}$\\sigma$" + r""" & -- \\
vs $\Lambda$CDM & """ + f"{results['tension_lcdm']:.1f}$\\sigma$" + r""" & -- \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_path = PLOTS_DIR / "void_cluster_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    
    print(f"  LaTeX table saved: {latex_path}")
    
    return latex


def interpret_physics(results):
    """Generate detailed physical interpretation."""
    
    delta_v = results['delta_v']
    se = results['delta_v_se']
    tension_unc = results['tension_unconstrained']
    tension_con = results['tension_constrained']
    tension_lcdm = results['tension_lcdm']
    
    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │  SDCG SCREENING PHYSICS                                             │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  S(ρ) = 1 / (1 + (ρ/ρ_thresh)²)                                    │
    │                                                                     │
    │  • Void environment:    ρ ~ 0.1 ρ_crit    →  S ≈ 1.00 (unscreened) │
    │  • Cluster environment: ρ ~ 200 ρ_crit    →  S ≈ 0.50 (screened)   │
    │                                                                     │
    │  G_eff = G_N × (1 + μ × S(ρ))                                      │
    │                                                                     │
    │  Δv ∝ √(G_void) - √(G_cluster)                                     │
    │     ∝ √(1 + μ) - √(1 + μ × 0.5)                                    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n  OBSERVED RESULT:")
    print(f"    Δv = {delta_v:+.2f} ± {se:.2f} km/s")
    print()
    
    # Unconstrained interpretation
    print("  1. UNCONSTRAINED SDCG (μ = 0.41):")
    print(f"     Prediction: +{results['pred_unconstrained']:.1f} km/s")
    if tension_unc > 3:
        print(f"     Tension:    {tension_unc:.1f}σ → FALSIFIED (>3σ)")
        print("     → This rules out SDCG without Lyα constraints")
    elif tension_unc > 2:
        print(f"     Tension:    {tension_unc:.1f}σ → STRONG TENSION")
    else:
        print(f"     Tension:    {tension_unc:.1f}σ → Consistent")
    print()
    
    # MCMC best-fit interpretation  
    print("  2. MCMC BEST-FIT SDCG (μ = 0.149):")
    print(f"     Prediction: +{results['pred_constrained']:.1f} km/s")
    if tension_con < 1:
        print(f"     Tension:    {tension_con:.1f}σ → EXCELLENT AGREEMENT")
    elif tension_con < 2:
        print(f"     Tension:    {tension_con:.1f}σ → Consistent (within 2σ)")
    else:
        print(f"     Tension:    {tension_con:.1f}σ → IN TENSION")
    print()
    
    # ΛCDM interpretation
    print("  3. ΛCDM (no modification):")
    print("     Prediction: 0 km/s")
    if tension_lcdm < 1:
        print(f"     Tension:    {tension_lcdm:.1f}σ → CONSISTENT (no signal detected)")
    elif tension_lcdm < 2:
        print(f"     Tension:    {tension_lcdm:.1f}σ → Consistent (within 2σ)")
    else:
        print(f"     Tension:    {tension_lcdm:.1f}σ → POSSIBLE SIGNAL DETECTED")
    print()
    
    # Overall verdict
    print("=" * 70)
    print("  FINAL VERDICT:")
    print("-" * 70)
    
    if tension_unc > 3 and tension_con < 2:
        verdict = """
    The data FALSIFY unconstrained SDCG but are CONSISTENT with the
    Lyα-constrained version. This demonstrates that:
    
    1. SDCG modifications must be small (μ < 0.1)
    2. Lyα forest constraints are essential
    3. The framework survives this falsification test
    
    STATUS: SDCG (Lyα-constrained) ✓ PASSES
"""
    elif tension_lcdm > 2 and tension_con < 2:
        verdict = """
    The data show a POSSIBLE SIGNAL inconsistent with ΛCDM but consistent
    with Lyα-constrained SDCG. This is preliminary evidence for modified
    gravity, but requires confirmation.
    
    STATUS: SDCG ✓ TENTATIVELY SUPPORTED
"""
    elif tension_con > 2 and tension_lcdm < 2:
        verdict = """
    The data are CONSISTENT with ΛCDM and show TENSION with SDCG.
    No evidence for gravity modification is detected.
    
    STATUS: SDCG ✗ NO SUPPORT
"""
    else:
        verdict = """
    The data are INCONCLUSIVE. Both ΛCDM and Lyα-constrained SDCG
    are consistent within 2σ. Larger samples are needed for discrimination.
    
    STATUS: INCONCLUSIVE - need more data
"""
    
    print(verdict)
    
    return verdict


def main():
    """Main interpretation pipeline."""
    print("=" * 70)
    print("SDCG FALSIFICATION TEST: INTERPRETATION")
    print("=" * 70)
    print()
    
    # Load results
    print("Loading analysis results...")
    try:
        results = load_results()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run run_void_cluster_analysis.py first.")
        sys.exit(1)
    
    print("  ✓ Results loaded")
    print()
    
    # Generate figure
    print("Generating publication figure...")
    try:
        create_figure(results)
        print("  ✓ Figure created")
    except Exception as e:
        print(f"  ⚠ Figure creation failed: {e}")
        print("    (matplotlib may not be available)")
    print()
    
    # Generate LaTeX table
    print("Generating LaTeX table...")
    generate_latex_table(results)
    print("  ✓ LaTeX table created")
    print()
    
    # Physical interpretation
    verdict = interpret_physics(results)
    
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()
    print("Outputs:")
    print(f"  • Figure:      {PLOTS_DIR / 'void_cluster_test.pdf'}")
    print(f"  • LaTeX table: {PLOTS_DIR / 'void_cluster_table.tex'}")
    print(f"  • Results:     {RESULTS_DIR / 'analysis_summary.txt'}")
    
    return results, verdict


if __name__ == "__main__":
    results, verdict = main()
