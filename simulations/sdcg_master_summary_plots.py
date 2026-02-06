#!/usr/bin/env python3
"""
SDCG Master Summary Plot
=========================

Creates a single comprehensive figure combining:
- MCMC results (cosmological constraints)
- LaCE Lyman-α results (μ constraints)
- All simulation stripping estimates
- Observed dwarf galaxy data
- SDCG theoretical predictions

For thesis presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'simulations' else SCRIPT_DIR
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
PLOTS_DIR = PROJECT_ROOT / 'plots' / 'sdcg_comparison'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (16, 12),
    'figure.dpi': 150,
})

# =============================================================================
# v12 THESIS VALUES (Official)
# =============================================================================

V12 = {
    # MCMC Results (without Lyman-α)
    'mu_mcmc': 0.47,
    'mu_mcmc_err': 0.03,
    'n_g_mcmc': 0.92,
    'z_trans_mcmc': 2.22,
    
    # LaCE/Lyman-α constrained
    'mu_lya': 0.045,
    'mu_lya_err': 0.019,
    'mu_upper_95': 0.012,  # 95% CL upper limit
    
    # Tension reduction
    'H0_tension_before': 4.8,  # sigma
    'H0_tension_after': 1.8,
    'S8_tension_before': 2.6,
    'S8_tension_after': 0.8,
    
    # Dwarf galaxy velocities
    'dv_observed': 15.6,
    'dv_observed_err': 1.3,
    'dv_stripping': 8.4,
    'dv_stripping_err': 1.2,
    'dv_residual': 7.2,
    'dv_residual_err': 1.4,
    'dv_sdcg': 12.0,
    'dv_sdcg_err': 3.0,
    'signal_sigma': 5.3,
    
    # Screening parameters
    'mu_bare': 0.48,
    'mu_void': 0.47,
    'mu_cluster': 0.17,
    'mu_igm': 0.05,
    'mu_solar': 0.0,
    'rho_thresh': 200,
    'beta0': 0.70,
    
    # Simulations
    'dv_eagle': 7.8,
    'dv_tng': 8.2,
    'dv_fire': 10.6,
    'dv_simba': 7.5,
}


def create_master_summary_plot():
    """Create the master 6-panel summary figure"""
    
    fig = plt.figure(figsize=(18, 14))
    
    # Create grid with 3x2 layout
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    # =========================================================================
    # Panel A: μ Parameter Comparison (MCMC vs LaCE vs Theory)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    sources = ['QFT Bare\n(μ̄)', 'MCMC\n(no Lyα)', 'Lyα\nConstrained', 
               'μ_eff\n(Voids)', 'μ_eff\n(Clusters)', 'μ_eff\n(IGM z~3)']
    mu_vals = [V12['mu_bare'], V12['mu_mcmc'], V12['mu_lya'],
               V12['mu_void'], V12['mu_cluster'], V12['mu_igm']]
    mu_errs = [0.05, V12['mu_mcmc_err'], V12['mu_lya_err'], 
               0.03, 0.02, 0.02]
    colors = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#e67e22']
    
    bars1 = ax1.bar(sources, mu_vals, yerr=mu_errs, capsize=4,
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax1.axhline(y=V12['mu_bare'], color='purple', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axhline(y=V12['mu_upper_95'], color='red', linestyle=':', linewidth=2, 
               label=f'95% CL: μ < {V12["mu_upper_95"]}')
    
    for bar, val, err in zip(bars1, mu_vals, mu_errs):
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val + err + 0.01),
                    ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('μ (Gravitational Coupling)', fontsize=12)
    ax1.set_title('A) SDCG μ Parameter: Theory vs MCMC vs Environment',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.6)
    
    # =========================================================================
    # Panel B: Tension Reduction
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    x = np.arange(2)
    width = 0.35
    
    before = [V12['H0_tension_before'], V12['S8_tension_before']]
    after = [V12['H0_tension_after'], V12['S8_tension_after']]
    
    bars_before = ax2.bar(x - width/2, before, width, label='Before SDCG (ΛCDM)',
                         color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.85)
    bars_after = ax2.bar(x + width/2, after, width, label='After SDCG (μ=0.47)',
                        color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax2.axhline(y=2, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(1.5, 2.1, '2σ threshold', fontsize=10, color='gray')
    
    ax2.set_ylabel('Tension (σ)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['H₀ Tension', 'S₈ Tension'], fontsize=12)
    ax2.set_title('B) Cosmological Tension Reduction',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 6)
    
    # Add reduction percentages
    for i, (b, a) in enumerate(zip(before, after)):
        reduction = (b - a) / b * 100
        ax2.annotate(f'-{reduction:.0f}%', xy=(i, max(b, a) + 0.3),
                    ha='center', fontsize=11, fontweight='bold', color='green')
    
    # =========================================================================
    # Panel C: Simulation Stripping Comparison
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    sims = ['EAGLE', 'IllustrisTNG', 'FIRE-2', 'SIMBA', 'Combined\n(Weighted)']
    dv_sims = [V12['dv_eagle'], V12['dv_tng'], V12['dv_fire'], V12['dv_simba'],
               V12['dv_stripping']]
    dv_errs = [1.2, 1.5, 5.2, 3.3, V12['dv_stripping_err']]
    sim_colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2c3e50']
    
    bars3 = ax3.bar(sims, dv_sims, yerr=dv_errs, capsize=4,
                   color=sim_colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax3.axhline(y=V12['dv_stripping'], color='black', linestyle='--', linewidth=2,
               label=f'Combined: {V12["dv_stripping"]}±{V12["dv_stripping_err"]} km/s')
    
    for bar, val in zip(bars3, dv_sims):
        ax3.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val + 0.5),
                    ha='center', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Tidal Stripping Δv (km/s)', fontsize=12)
    ax3.set_title('C) Cosmological Simulation Predictions',
                 fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 18)
    
    # =========================================================================
    # Panel D: Dwarf Galaxy Signal Decomposition
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    components = ['Observed\n(Void-Cluster)', 'Tidal\nStripping', 
                  'SDCG\nResidual', 'SDCG\nPrediction', 'ΛCDM\nPrediction']
    comp_vals = [V12['dv_observed'], V12['dv_stripping'], V12['dv_residual'],
                V12['dv_sdcg'], 0]
    comp_errs = [V12['dv_observed_err'], V12['dv_stripping_err'], 
                V12['dv_residual_err'], V12['dv_sdcg_err'], 0]
    comp_colors = ['#3498db', '#f39c12', '#2ecc71', '#e74c3c', '#95a5a6']
    
    bars4 = ax4.bar(components, comp_vals, yerr=comp_errs, capsize=4,
                   color=comp_colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for bar, val, err in zip(bars4, comp_vals, comp_errs):
        if val > 0:
            ax4.annotate(f'{val:.1f}±{err:.1f}', 
                        xy=(bar.get_x() + bar.get_width()/2, val + 0.5),
                        ha='center', fontsize=9, fontweight='bold')
    
    ax4.annotate(f'{V12["signal_sigma"]}σ signal', xy=(2, V12['dv_residual'] + 2),
                fontsize=12, ha='center', color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax4.set_ylabel('Velocity Difference Δv (km/s)', fontsize=12)
    ax4.set_title('D) Dwarf Galaxy Signal Decomposition',
                 fontsize=14, fontweight='bold')
    ax4.set_ylim(-2, 22)
    
    # =========================================================================
    # Panel E: Screening Mechanism
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    rho = np.logspace(-2, 4, 200)
    S = np.exp(-rho / V12['rho_thresh'])
    mu_eff = V12['mu_bare'] * S
    
    ax5.loglog(rho, mu_eff, 'b-', linewidth=3, label='μ_eff(ρ) = μ̄ × exp(-ρ/ρ_thresh)')
    
    # Mark environments
    env_data = [
        ('Voids', 0.1, V12['mu_void'], 'green'),
        ('Field', 1, 0.35, 'blue'),
        ('IGM', 10, V12['mu_igm'], 'orange'),
        ('Clusters', 100, V12['mu_cluster'], 'red'),
        ('Solar', 1e4, 1e-5, 'gray'),
    ]
    
    for name, rho_val, mu_val, color in env_data:
        ax5.scatter([rho_val], [max(mu_val, 1e-5)], s=150, c=color, marker='*',
                   edgecolors='black', linewidth=1.5, zorder=5)
        ax5.annotate(name, xy=(rho_val*1.5, max(mu_val, 1e-5)*1.5),
                    fontsize=9, color=color, fontweight='bold')
    
    ax5.axvline(x=V12['rho_thresh'], color='gray', linestyle=':', linewidth=2)
    ax5.axhline(y=V12['mu_bare'], color='purple', linestyle='--', linewidth=2, alpha=0.5)
    
    ax5.set_xlabel('ρ/ρ_crit (Environmental Density)', fontsize=12)
    ax5.set_ylabel('μ_eff (Effective Coupling)', fontsize=12)
    ax5.set_title('E) SDCG Chameleon Screening Mechanism',
                 fontsize=14, fontweight='bold')
    ax5.legend(loc='lower left')
    ax5.set_xlim(0.01, 1e5)
    ax5.set_ylim(1e-6, 1)
    
    # =========================================================================
    # Panel F: Summary Table
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    table_data = [
        ['Parameter', 'Value', 'Source'],
        ['μ (MCMC)', f'{V12["mu_mcmc"]} ± {V12["mu_mcmc_err"]}', 'CMB+BAO+SNe'],
        ['μ (Lyα)', f'{V12["mu_lya"]} ± {V12["mu_lya_err"]}', 'DESI/eBOSS'],
        ['μ_bare (QFT)', f'{V12["mu_bare"]}', 'One-loop'],
        ['β₀ (SM)', f'{V12["beta0"]}', 'Conformal anomaly'],
        ['', '', ''],
        ['Δv observed', f'{V12["dv_observed"]} ± {V12["dv_observed_err"]} km/s', '86 galaxies'],
        ['Δv stripping', f'{V12["dv_stripping"]} ± {V12["dv_stripping_err"]} km/s', 'Simulations'],
        ['Δv residual', f'{V12["dv_residual"]} ± {V12["dv_residual_err"]} km/s', f'{V12["signal_sigma"]}σ signal'],
        ['Δv SDCG pred.', f'{V12["dv_sdcg"]} ± {V12["dv_sdcg_err"]} km/s', 'Theory'],
        ['', '', ''],
        ['H₀ reduction', '62%', '4.8σ → 1.8σ'],
        ['S₈ reduction', '69%', '2.6σ → 0.8σ'],
    ]
    
    table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.35, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header row
    for j in range(3):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax6.set_title('F) SDCG v12 Key Results Summary',
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Main title
    fig.suptitle('Scale-Dependent Crossover Gravity (SDCG): Comprehensive Analysis Summary\n'
                'Comparing MCMC, Lyman-α, Simulations, and Dwarf Galaxy Observations',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(PLOTS_DIR / 'sdcg_master_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'sdcg_master_summary.pdf', bbox_inches='tight')
    print(f"✓ Saved: {PLOTS_DIR / 'sdcg_master_summary.png'}")
    plt.show()
    
    return fig


def create_thesis_figure_velocity_comparison():
    """Create a clean figure for thesis: Observed vs Theory vs Simulations"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Y positions for categories
    y_obs = [5, 4.5, 4, 3.5]
    y_sim = [2.5, 2, 1.5, 1, 0.5]
    y_theory = [-0.5, -1]
    
    # Observations
    obs_labels = ['SPARC+ALFALFA+LG (86 gal)', 'Corrected (minus stripping)', 
                  'LITTLE THINGS', 'Void Dwarfs (Pustilnik+)']
    obs_vals = [V12['dv_observed'], V12['dv_residual'], 11.0, 18.0]
    obs_errs = [V12['dv_observed_err'], V12['dv_residual_err'], 2.8, 4.0]
    
    for y, label, val, err in zip(y_obs, obs_labels, obs_vals, obs_errs):
        ax.barh(y, val, xerr=err, height=0.4, color='#3498db', 
               edgecolor='black', linewidth=1.5, capsize=4, alpha=0.85)
        ax.text(val + err + 0.5, y, f'{val:.1f}±{err:.1f}', va='center', fontsize=10)
        ax.text(-1, y, label, va='center', ha='right', fontsize=10)
    
    # Simulations
    sim_labels = ['EAGLE', 'IllustrisTNG', 'FIRE-2', 'SIMBA', 'Combined Sims']
    sim_vals = [V12['dv_eagle'], V12['dv_tng'], V12['dv_fire'], V12['dv_simba'],
                V12['dv_stripping']]
    sim_errs = [1.2, 1.5, 5.2, 3.3, V12['dv_stripping_err']]
    
    for y, label, val, err in zip(y_sim, sim_labels, sim_vals, sim_errs):
        ax.barh(y, val, xerr=err, height=0.4, color='#f39c12', 
               edgecolor='black', linewidth=1.5, capsize=4, alpha=0.85)
        ax.text(val + err + 0.5, y, f'{val:.1f}±{err:.1f}', va='center', fontsize=10)
        ax.text(-1, y, label, va='center', ha='right', fontsize=10)
    
    # Theory
    theory_labels = ['SDCG Prediction (μ=0.47)', 'ΛCDM Prediction (μ=0)']
    theory_vals = [V12['dv_sdcg'], 0]
    theory_errs = [V12['dv_sdcg_err'], 0]
    theory_colors = ['#2ecc71', '#95a5a6']
    
    for y, label, val, err, color in zip(y_theory, theory_labels, theory_vals, 
                                         theory_errs, theory_colors):
        ax.barh(y, val, xerr=err if err > 0 else None, height=0.4, color=color, 
               edgecolor='black', linewidth=1.5, capsize=4 if err > 0 else 0, alpha=0.85)
        ax.text(max(val, 0.5) + (err if err > 0 else 0) + 0.5, y, 
               f'{val:.1f}±{err:.1f}' if err > 0 else '0', va='center', fontsize=10)
        ax.text(-1, y, label, va='center', ha='right', fontsize=10)
    
    # Reference lines
    ax.axvline(x=V12['dv_observed'], color='blue', linestyle='--', linewidth=2, alpha=0.4)
    ax.axvline(x=V12['dv_stripping'], color='orange', linestyle=':', linewidth=2, alpha=0.4)
    ax.axvline(x=V12['dv_sdcg'], color='green', linestyle='-', linewidth=2, alpha=0.4)
    
    # Category labels
    ax.text(-8, 4.5, 'OBSERVATIONS', fontsize=12, fontweight='bold', color='#3498db',
           rotation=90, va='center')
    ax.text(-8, 1.5, 'SIMULATIONS', fontsize=12, fontweight='bold', color='#f39c12',
           rotation=90, va='center')
    ax.text(-8, -0.75, 'THEORY', fontsize=12, fontweight='bold', color='#2ecc71',
           rotation=90, va='center')
    
    # Horizontal dividers
    ax.axhline(y=3, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlim(-1, 25)
    ax.set_ylim(-1.5, 5.8)
    ax.set_xlabel('Velocity Difference Δv (Void - Cluster) [km/s]', fontsize=14)
    ax.set_title('SDCG Dwarf Galaxy Velocity Test:\nObservations, Simulations, and Theory',
                fontsize=16, fontweight='bold')
    ax.set_yticks([])
    
    # Legend for reference lines
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, 
               label=f'Observed ({V12["dv_observed"]} km/s)'),
        Line2D([0], [0], color='orange', linestyle=':', linewidth=2, 
               label=f'Stripping ({V12["dv_stripping"]} km/s)'),
        Line2D([0], [0], color='green', linestyle='-', linewidth=2, 
               label=f'SDCG ({V12["dv_sdcg"]} km/s)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'thesis_velocity_comparison.png', dpi=300)
    plt.savefig(PLOTS_DIR / 'thesis_velocity_comparison.pdf')
    print(f"✓ Saved: {PLOTS_DIR / 'thesis_velocity_comparison.png'}")
    plt.show()
    
    return fig


if __name__ == "__main__":
    print("="*70)
    print("GENERATING SDCG MASTER SUMMARY PLOTS")
    print("="*70)
    
    print("\n1. Creating master 6-panel summary...")
    create_master_summary_plot()
    
    print("\n2. Creating thesis velocity comparison figure...")
    create_thesis_figure_velocity_comparison()
    
    print("\n" + "="*70)
    print("PLOTS COMPLETE")
    print("="*70)
