#!/usr/bin/env python3
"""
plot_results.py - Visualization for Void vs Cluster Dwarf Rotation Test

Generates publication-quality figures comparing velocity distributions
for void and cluster dwarf galaxies to test CGC prediction.

CGC Prediction: Δv = +12 ± 3 km/s (void dwarfs rotate faster)
ΛCDM Prediction: Δv = 0 km/s (no environment dependence)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Set publication-quality plot style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def plot_velocity_histograms(v_void, v_cluster, results, save_path='plots'):
    """
    Create side-by-side histograms comparing velocity distributions.
    
    Parameters
    ----------
    v_void : array-like
        Rotation velocities of void dwarf galaxies [km/s]
    v_cluster : array-like
        Rotation velocities of cluster dwarf galaxies [km/s]
    results : dict
        Analysis results from analyze_velocity.py
    save_path : str
        Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ===== Panel 1: Overlapping histograms =====
    ax1 = axes[0, 0]
    bins = np.linspace(15, 120, 22)
    
    ax1.hist(v_void, bins=bins, alpha=0.6, color='blue', 
             label=f'Void Dwarfs (N={len(v_void)})', density=True, edgecolor='darkblue')
    ax1.hist(v_cluster, bins=bins, alpha=0.6, color='red', 
             label=f'Cluster Dwarfs (N={len(v_cluster)})', density=True, edgecolor='darkred')
    
    # Mark means
    mean_void = results['mean_void']
    mean_cluster = results['mean_cluster']
    ax1.axvline(mean_void, color='blue', linestyle='--', linewidth=2, label=f'Mean Void: {mean_void:.1f} km/s')
    ax1.axvline(mean_cluster, color='red', linestyle='--', linewidth=2, label=f'Mean Cluster: {mean_cluster:.1f} km/s')
    
    ax1.set_xlabel('Rotation Velocity V$_{rot}$ [km/s]')
    ax1.set_ylabel('Normalized Density')
    ax1.set_title('Velocity Distributions by Environment')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # ===== Panel 2: Box plot comparison =====
    ax2 = axes[0, 1]
    box_data = [v_void, v_cluster]
    bp = ax2.boxplot(box_data, labels=['Void\nDwarfs', 'Cluster\nDwarfs'], 
                      patch_artist=True, widths=0.6)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Rotation Velocity V$_{rot}$ [km/s]')
    ax2.set_title('Velocity Distribution Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    ax2.scatter([1, 2], [mean_void, mean_cluster], color=['blue', 'red'], 
                s=100, marker='D', zorder=5, label='Mean')
    ax2.legend()
    
    # ===== Panel 3: Delta-v result with predictions =====
    ax3 = axes[1, 0]
    
    delta_v = results['delta_v']
    delta_v_err = results['delta_v_err']
    
    # Plot predictions
    predictions = {
        'ΛCDM': (0, 0.5, 'gray'),
        'CGC': (12, 3, 'green'),
        'Observed': (delta_v, delta_v_err, 'blue')
    }
    
    y_pos = [0, 1, 2]
    colors_pred = ['gray', 'green', 'blue']
    
    for i, (name, (val, err, color)) in enumerate(predictions.items()):
        ax3.errorbar(val, y_pos[i], xerr=err, fmt='o', markersize=12, 
                     color=color, capsize=6, capthick=2, linewidth=2)
    
    ax3.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(['ΛCDM\nPrediction', 'CGC\nPrediction', 'Observed\nΔv'])
    ax3.set_xlabel('Δv = V$_{void}$ - V$_{cluster}$ [km/s]')
    ax3.set_title('Velocity Difference: Theory vs Observation')
    ax3.set_xlim(-15, 30)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add significance annotation
    t_stat = results.get('t_statistic', 0)
    p_value = results.get('p_value', 1)
    significance = 'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'
    ax3.text(0.95, 0.05, f't = {t_stat:.2f}, p = {p_value:.4f}\n{significance}',
             transform=ax3.transAxes, ha='right', va='bottom',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== Panel 4: Summary statistics =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          VOID vs CLUSTER DWARF ROTATION TEST                 ║
    ║                  CGC Theory Validation                       ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  VOID DWARFS                                                 ║
    ║    Sample Size:      N = {len(v_void):>6}                           ║
    ║    Mean V_rot:       {mean_void:>6.2f} ± {results.get('err_void', 0):.2f} km/s                 ║
    ║                                                              ║
    ║  CLUSTER DWARFS                                              ║
    ║    Sample Size:      N = {len(v_cluster):>6}                           ║
    ║    Mean V_rot:       {mean_cluster:>6.2f} ± {results.get('err_cluster', 0):.2f} km/s                 ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  RESULT                                                      ║
    ║    Δv (observed):    {delta_v:>+6.2f} ± {delta_v_err:.2f} km/s                    ║
    ║    CGC prediction:   +12.00 ± 3.00 km/s                      ║
    ║    ΛCDM prediction:   0.00 ± 0.50 km/s                       ║
    ║                                                              ║
    ║    t-statistic:      {t_stat:>6.2f}                                  ║
    ║    p-value:          {p_value:>6.4f}                                ║
    ║                                                              ║
    ║  CONCLUSION: {'CGC SUPPORTED' if abs(delta_v - 12) < 2*delta_v_err else 'INCONCLUSIVE':^40}       ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=9, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    save_file = os.path.join(save_path, 'cgc_dwarf_test_results.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_file}")
    
    plt.show()
    
    return fig


def plot_mass_velocity_relation(df_void, df_cluster, save_path='plots'):
    """
    Plot stellar mass vs rotation velocity for both samples.
    
    Parameters
    ----------
    df_void : pd.DataFrame
        Void dwarf galaxy data
    df_cluster : pd.DataFrame  
        Cluster dwarf galaxy data
    save_path : str
        Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot void dwarfs
    if 'log_Mstar' in df_void.columns and 'V_rot' in df_void.columns:
        ax.scatter(df_void['log_Mstar'], df_void['V_rot'], 
                   c='blue', alpha=0.5, s=30, label='Void Dwarfs')
    
    # Plot cluster dwarfs
    if 'log_Mstar' in df_cluster.columns and 'V_rot' in df_cluster.columns:
        ax.scatter(df_cluster['log_Mstar'], df_cluster['V_rot'],
                   c='red', alpha=0.5, s=30, label='Cluster Dwarfs')
    
    ax.set_xlabel('log$_{10}$(M$_*$ / M$_\\odot$)')
    ax.set_ylabel('Rotation Velocity V$_{rot}$ [km/s]')
    ax.set_title('Stellar Mass - Rotation Velocity Relation')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add CGC prediction annotation
    ax.text(0.05, 0.95, 'CGC: Void dwarfs should show\n+12 km/s offset at fixed mass',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_file = os.path.join(save_path, 'mass_velocity_relation.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_file}")
    
    plt.show()
    
    return fig


def plot_bootstrap_distribution(bootstrap_diffs, observed_diff, save_path='plots'):
    """
    Plot bootstrap distribution of Δv to show uncertainty.
    
    Parameters
    ----------
    bootstrap_diffs : array-like
        Bootstrap samples of Δv
    observed_diff : float
        Observed velocity difference
    save_path : str
        Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7, 
            color='steelblue', edgecolor='darkblue')
    
    # Mark confidence intervals
    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)
    
    ax.axvline(observed_diff, color='red', linewidth=2, linestyle='-',
               label=f'Observed Δv = {observed_diff:.2f} km/s')
    ax.axvline(ci_low, color='orange', linewidth=1.5, linestyle='--',
               label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')
    ax.axvline(ci_high, color='orange', linewidth=1.5, linestyle='--')
    
    # Mark predictions
    ax.axvline(0, color='gray', linewidth=2, linestyle=':',
               label='ΛCDM: Δv = 0')
    ax.axvline(12, color='green', linewidth=2, linestyle=':',
               label='CGC: Δv = +12 km/s')
    
    ax.set_xlabel('Δv = V$_{void}$ - V$_{cluster}$ [km/s]')
    ax.set_ylabel('Bootstrap Density')
    ax.set_title('Bootstrap Distribution of Velocity Difference')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_file = os.path.join(save_path, 'bootstrap_distribution.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_file}")
    
    plt.show()
    
    return fig


def generate_all_plots(results_file='results/cgc_dwarf_analysis.npz', save_path='plots'):
    """
    Load analysis results and generate all plots.
    
    Parameters
    ----------
    results_file : str
        Path to saved analysis results
    save_path : str
        Directory to save plots
    """
    import pandas as pd
    
    print("=" * 60)
    print("CGC DWARF TEST - VISUALIZATION MODULE")
    print("=" * 60)
    
    # Load results
    if os.path.exists(results_file):
        data = np.load(results_file, allow_pickle=True)
        results = data['results'].item()
        v_void = data['v_void']
        v_cluster = data['v_cluster']
        
        print(f"✓ Loaded results from {results_file}")
    else:
        print(f"✗ Results file not found: {results_file}")
        print("  Please run analyze_velocity.py first.")
        return
    
    # Generate main comparison plot
    print("\nGenerating velocity comparison plots...")
    plot_velocity_histograms(v_void, v_cluster, results, save_path)
    
    # Generate bootstrap distribution if available
    if 'bootstrap_diffs' in results:
        print("\nGenerating bootstrap distribution plot...")
        plot_bootstrap_distribution(results['bootstrap_diffs'], 
                                    results['delta_v'], save_path)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    # Run visualization
    generate_all_plots(
        results_file='results/cgc_dwarf_analysis.npz',
        save_path='plots'
    )
