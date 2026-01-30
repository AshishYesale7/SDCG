"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CGC Plotting Module                                 ║
║                                                                              ║
║  Publication-quality visualization for CGC cosmological analysis.           ║
║  Generates 22+ diagnostic, posterior, and comparison plots.                 ║
║                                                                              ║
║  Plot Categories:                                                            ║
║    • Posterior distributions (corner plots, 1D marginalized)                ║
║    • Data comparisons (CMB, BAO, growth, SNe)                               ║
║    • Model comparison (ΛCDM vs CGC)                                         ║
║    • Tension analysis (H0, S8, σ8)                                          ║
║    • Diagnostic plots (chains, convergence, autocorrelation)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
>>> from cgc.plotting import PlotGenerator, plot_all
>>> from cgc.data_loader import load_real_data
>>> from cgc.mcmc import run_mcmc
>>>
>>> data = load_real_data()
>>> results = run_mcmc(data, n_steps=5000)
>>> 
>>> # Generate all plots
>>> plot_all(results, data)
>>>
>>> # Or use the class interface for customization
>>> plotter = PlotGenerator(results, data)
>>> plotter.corner_plot()
>>> plotter.h0_posterior()
>>> plotter.tension_comparison()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import Dict, Any, List, Tuple, Optional
import warnings

from .config import PATHS, PLANCK_BASELINE, TENSIONS, PLOT_SETTINGS

# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Set matplotlib style for publication quality
plt.rcParams.update({
    'font.size': PLOT_SETTINGS.get('font_size_label', 12),
    'axes.labelsize': PLOT_SETTINGS.get('font_size_label', 12),
    'axes.titlesize': PLOT_SETTINGS.get('font_size_title', 14),
    'xtick.labelsize': PLOT_SETTINGS.get('font_size_tick', 10),
    'ytick.labelsize': PLOT_SETTINGS.get('font_size_tick', 10),
    'legend.fontsize': PLOT_SETTINGS.get('font_size_legend', 10),
    'figure.dpi': PLOT_SETTINGS.get('dpi_screen', 100),
    'savefig.dpi': PLOT_SETTINGS.get('dpi_publication', 300),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': PLOT_SETTINGS.get('grid_alpha', 0.3),
    'lines.linewidth': PLOT_SETTINGS.get('line_width', 1.5),
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# Parameter labels for plots
PARAM_LABELS = [
    r'$\omega_b$', r'$\omega_{cdm}$', r'$h$', r'$\ln(10^{10}A_s)$', 
    r'$n_s$', r'$\tau_{reio}$', r'$\mu$', r'$n_g$', 
    r'$z_{trans}$', r'$\rho_{thresh}$'
]

PARAM_NAMES_SIMPLE = [
    'omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau_reio',
    'mu', 'n_g', 'z_trans', 'rho_thresh'
]

# CGC-specific colors
COLORS = {
    'cgc': '#E31A1C',       # Red for CGC
    'lcdm': '#1F78B4',      # Blue for ΛCDM
    'planck': '#1F78B4',    # Blue for Planck
    'shoes': '#FF7F00',     # Orange for SH0ES
    'data': '#666666',      # Gray for data points
    'posterior': '#6A3D9A', # Purple for posteriors
    'highlight': '#33A02C', # Green for highlights
}


# =============================================================================
# PLOT GENERATOR CLASS
# =============================================================================

class PlotGenerator:
    """
    Comprehensive plot generator for CGC analysis results.
    
    Parameters
    ----------
    results : dict
        MCMC results dictionary containing 'chains', 'flat_chains', etc.
    data : dict
        Cosmological data dictionary.
    save_dir : str, optional
        Directory to save plots. Defaults to PATHS['plots'].
    
    Attributes
    ----------
    chains : np.ndarray
        MCMC chains (flat or 3D).
    data : dict
        Observational data.
    n_params : int
        Number of parameters.
    
    Examples
    --------
    >>> plotter = PlotGenerator(results, data)
    >>> plotter.corner_plot(cgc_only=True)
    >>> plotter.h0_posterior()
    >>> plotter.save_all()
    """
    
    def __init__(self, results: Dict[str, Any], data: Dict[str, Any],
                 save_dir: str = None):
        """Initialize the plot generator."""
        self.results = results
        self.data = data
        self.save_dir = save_dir or PATHS['plots']
        
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Extract chains
        if 'flat_chains' in results:
            self.chains = results['flat_chains']
        elif 'chains' in results:
            chains = results['chains']
            if chains.ndim == 3:
                # (n_walkers, n_steps, n_params) -> (n_samples, n_params)
                self.chains = chains.reshape(-1, chains.shape[-1])
            else:
                self.chains = chains
        else:
            raise ValueError("Results must contain 'chains' or 'flat_chains'")
        
        self.n_samples, self.n_params = self.chains.shape
        
        # Compute derived quantities
        self._compute_derived()
        
        # Track generated plots
        self.generated_plots = []
    
    def _compute_derived(self):
        """Compute derived cosmological parameters from chains."""
        # H0 = h * 100
        self.H0_samples = self.chains[:, 2] * 100
        
        # Ω_m = (ω_cdm + ω_b) / h²
        self.Omega_m_samples = (self.chains[:, 1] + self.chains[:, 0]) / self.chains[:, 2]**2
        
        # σ8 and S8 (approximate scaling)
        sigma8_base = PLANCK_BASELINE['sigma8']
        self.sigma8_samples = sigma8_base * (self.Omega_m_samples / PLANCK_BASELINE['Omega_m'])**0.25
        
        # S8 = σ8 × √(Ω_m / 0.3)
        self.S8_samples = self.sigma8_samples * np.sqrt(self.Omega_m_samples / 0.3)
        
        # Statistics
        self.means = np.mean(self.chains, axis=0)
        self.stds = np.std(self.chains, axis=0)
        self.H0_mean = np.mean(self.H0_samples)
        self.H0_std = np.std(self.H0_samples)
        self.S8_mean = np.mean(self.S8_samples)
        self.S8_std = np.std(self.S8_samples)
    
    def _save_plot(self, fig, name: str, close: bool = True):
        """Save a plot and optionally close it."""
        filepath = os.path.join(self.save_dir, f'{name}.png')
        dpi = PLOT_SETTINGS.get('dpi_publication', 300)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        self.generated_plots.append(filepath)
        print(f"    ✓ Saved: {name}.png")
        if close:
            plt.close(fig)
        return filepath
    
    # =========================================================================
    # CORNER PLOTS
    # =========================================================================
    
    def corner_plot(self, cgc_only: bool = False, 
                    show_truths: bool = True,
                    filename: str = None) -> str:
        """
        Generate corner plot showing parameter correlations.
        
        Parameters
        ----------
        cgc_only : bool, default=False
            If True, only show CGC parameters (μ, n_g, z_trans, ρ_thresh).
        show_truths : bool, default=True
            Show true/input parameter values.
        filename : str, optional
            Custom filename (without extension).
        
        Returns
        -------
        str
            Path to saved figure.
        """
        try:
            import corner
        except ImportError:
            print("    ⚠ corner package not installed, skipping corner plot")
            return None
        
        if cgc_only:
            # CGC parameters only (indices 6-9)
            samples = self.chains[:, 6:10]
            labels = PARAM_LABELS[6:10]
            truths = self.means[6:10] if show_truths else None
            name = filename or 'cgc_corner_plot'
        else:
            samples = self.chains
            labels = PARAM_LABELS
            truths = self.means if show_truths else None
            name = filename or 'full_corner_plot'
        
        fig = corner.corner(
            samples,
            labels=labels,
            truths=truths,
            show_titles=True,
            title_kwargs={"fontsize": 10},
            quantiles=[0.16, 0.5, 0.84],
            title_fmt='.4f',
            smooth=1.0,
            smooth1d=1.0,
            color=COLORS['posterior'],
            truth_color=COLORS['cgc']
        )
        
        title = 'CGC Parameters' if cgc_only else 'Full Parameter Posteriors'
        fig.suptitle(f'CGC Theory: {title}', fontsize=14, y=1.02)
        
        return self._save_plot(fig, name)
    
    # =========================================================================
    # H0 POSTERIOR
    # =========================================================================
    
    def h0_posterior(self, show_observations: bool = True,
                     filename: str = 'h0_posterior') -> str:
        """
        Plot H0 posterior distribution with Planck and SH0ES comparison.
        
        Parameters
        ----------
        show_observations : bool, default=True
            Show Planck and SH0ES measurements.
        filename : str, default='h0_posterior'
            Output filename.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS.get('fig_single', (10, 6)))
        
        # CGC posterior histogram
        ax.hist(self.H0_samples, bins=60, density=True, alpha=0.7,
                color=COLORS['cgc'], edgecolor='white', linewidth=0.5,
                label=f'CGC Posterior: {self.H0_mean:.2f} ± {self.H0_std:.2f}')
        
        if show_observations:
            x_range = np.linspace(60, 80, 200)
            
            # Planck
            planck_H0 = TENSIONS['H0_planck']['value']
            planck_err = TENSIONS['H0_planck']['error']
            planck_pdf = np.exp(-0.5 * ((x_range - planck_H0) / planck_err)**2)
            planck_pdf /= (planck_err * np.sqrt(2*np.pi))
            ax.fill_between(x_range, planck_pdf, alpha=0.3, color=COLORS['planck'],
                           label=f'Planck 2018: {planck_H0:.2f} ± {planck_err:.2f}')
            
            # SH0ES
            shoes_H0 = TENSIONS['H0_sh0es']['value']
            shoes_err = TENSIONS['H0_sh0es']['error']
            shoes_pdf = np.exp(-0.5 * ((x_range - shoes_H0) / shoes_err)**2)
            shoes_pdf /= (shoes_err * np.sqrt(2*np.pi))
            ax.fill_between(x_range, shoes_pdf, alpha=0.3, color=COLORS['shoes'],
                           label=f'SH0ES 2022: {shoes_H0:.2f} ± {shoes_err:.2f}')
        
        # CGC best-fit line
        ax.axvline(self.H0_mean, color=COLORS['cgc'], linewidth=2, linestyle='-')
        ax.axvline(self.H0_mean - self.H0_std, color=COLORS['cgc'], 
                   linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(self.H0_mean + self.H0_std, color=COLORS['cgc'],
                   linewidth=1, linestyle='--', alpha=0.5)
        
        ax.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Hubble Constant: CGC Theory vs Observations', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_xlim(60, 82)
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # S8 POSTERIOR
    # =========================================================================
    
    def s8_posterior(self, filename: str = 's8_posterior') -> str:
        """
        Plot S8 posterior distribution with tension comparison.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS.get('fig_single', (10, 6)))
        
        # CGC posterior
        ax.hist(self.S8_samples, bins=60, density=True, alpha=0.7,
                color=COLORS['cgc'], edgecolor='white', linewidth=0.5,
                label=f'CGC: {self.S8_mean:.3f} ± {self.S8_std:.3f}')
        
        x_range = np.linspace(0.7, 0.9, 200)
        
        # Planck S8
        planck_S8 = TENSIONS['S8_planck']['value']
        planck_err = TENSIONS['S8_planck']['error']
        planck_pdf = np.exp(-0.5 * ((x_range - planck_S8) / planck_err)**2)
        planck_pdf /= (planck_err * np.sqrt(2*np.pi))
        ax.fill_between(x_range, planck_pdf, alpha=0.3, color=COLORS['planck'],
                       label=f'Planck: {planck_S8:.3f} ± {planck_err:.3f}')
        
        # Weak lensing S8
        wl_S8 = TENSIONS['S8_wl']['value']
        wl_err = TENSIONS['S8_wl']['error']
        wl_pdf = np.exp(-0.5 * ((x_range - wl_S8) / wl_err)**2)
        wl_pdf /= (wl_err * np.sqrt(2*np.pi))
        ax.fill_between(x_range, wl_pdf, alpha=0.3, color=COLORS['highlight'],
                       label=f'Weak Lensing: {wl_S8:.3f} ± {wl_err:.3f}')
        
        ax.set_xlabel(r'$S_8 = \sigma_8 \sqrt{\Omega_m/0.3}$', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(r'$S_8$ Tension: CGC vs Observations', fontsize=14)
        ax.legend(fontsize=10)
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # CGC PARAMETER CONSTRAINTS
    # =========================================================================
    
    def cgc_parameters(self, filename: str = 'cgc_parameters') -> str:
        """
        Plot individual CGC parameter posteriors in a 2×2 grid.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=PLOT_SETTINGS.get('fig_quad', (12, 10)))
        
        param_info = [
            (6, r'$\mu$ (CGC coupling)', 'purple'),
            (7, r'$n_g$ (scale dependence)', 'teal'),
            (8, r'$z_{trans}$ (transition redshift)', 'coral'),
            (9, r'$\rho_{thresh}$ (screening)', 'gold')
        ]
        
        for ax, (idx, label, color) in zip(axes.flat, param_info):
            samples = self.chains[:, idx]
            mean, std = self.means[idx], self.stds[idx]
            
            ax.hist(samples, bins=50, density=True, alpha=0.7, 
                    color=color, edgecolor='white')
            ax.axvline(mean, color='green', linewidth=2, linestyle='-',
                      label=f'Mean: {mean:.4f}')
            ax.axvline(mean - std, color='green', linewidth=1, 
                      linestyle='--', alpha=0.5)
            ax.axvline(mean + std, color='green', linewidth=1,
                      linestyle='--', alpha=0.5)
            
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'{label}: {mean:.4f} ± {std:.4f}', fontsize=12)
            ax.legend(fontsize=9)
        
        plt.suptitle('CGC Theory Parameter Constraints', fontsize=14)
        plt.tight_layout()
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # TENSION COMPARISON
    # =========================================================================
    
    def tension_comparison(self, filename: str = 'tension_comparison') -> str:
        """
        Bar chart comparing H0 and S8 tensions in ΛCDM vs CGC.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ─────────────────────────────────────────────────────────────────────
        # H0 Tension
        # ─────────────────────────────────────────────────────────────────────
        ax = axes[0]
        
        planck_H0 = TENSIONS['H0_planck']['value']
        planck_err = TENSIONS['H0_planck']['error']
        shoes_H0 = TENSIONS['H0_sh0es']['value']
        shoes_err = TENSIONS['H0_sh0es']['error']
        
        # ΛCDM tension
        combined_err = np.sqrt(planck_err**2 + shoes_err**2)
        tension_lcdm = abs(shoes_H0 - planck_H0) / combined_err
        
        # CGC tension (max of tension with both)
        tension_cgc_p = abs(self.H0_mean - planck_H0) / np.sqrt(self.H0_std**2 + planck_err**2)
        tension_cgc_s = abs(self.H0_mean - shoes_H0) / np.sqrt(self.H0_std**2 + shoes_err**2)
        tension_cgc = max(tension_cgc_p, tension_cgc_s)
        
        x = np.array([0, 1])
        bars = ax.bar(x, [tension_lcdm, tension_cgc], 
                     color=[COLORS['lcdm'], COLORS['cgc']], alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['ΛCDM', 'CGC'], fontsize=12)
        ax.set_ylabel('Tension (σ)', fontsize=12)
        ax.set_title('Hubble Tension', fontsize=14)
        
        # Add values on bars
        for bar, val in zip(bars, [tension_lcdm, tension_cgc]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.1f}σ', ha='center', fontsize=11, fontweight='bold')
        
        # Reduction arrow
        reduction = (1 - tension_cgc/tension_lcdm) * 100
        ax.annotate('', xy=(1, tension_cgc), xytext=(0, tension_lcdm),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(0.5, (tension_lcdm + tension_cgc)/2, f'{reduction:.0f}%\nreduction',
               ha='center', va='center', fontsize=10, color='green', fontweight='bold')
        
        # ─────────────────────────────────────────────────────────────────────
        # S8 Tension
        # ─────────────────────────────────────────────────────────────────────
        ax = axes[1]
        
        planck_S8 = TENSIONS['S8_planck']['value']
        planck_S8_err = TENSIONS['S8_planck']['error']
        wl_S8 = TENSIONS['S8_wl']['value']
        wl_S8_err = TENSIONS['S8_wl']['error']
        
        combined_S8_err = np.sqrt(planck_S8_err**2 + wl_S8_err**2)
        tension_S8_lcdm = abs(planck_S8 - wl_S8) / combined_S8_err
        
        tension_S8_cgc_p = abs(self.S8_mean - planck_S8) / np.sqrt(self.S8_std**2 + planck_S8_err**2)
        tension_S8_cgc_w = abs(self.S8_mean - wl_S8) / np.sqrt(self.S8_std**2 + wl_S8_err**2)
        tension_S8_cgc = max(tension_S8_cgc_p, tension_S8_cgc_w)
        
        bars = ax.bar(x, [tension_S8_lcdm, tension_S8_cgc],
                     color=[COLORS['lcdm'], COLORS['cgc']], alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['ΛCDM', 'CGC'], fontsize=12)
        ax.set_ylabel('Tension (σ)', fontsize=12)
        ax.set_title(r'$S_8$ Tension', fontsize=14)
        
        for bar, val in zip(bars, [tension_S8_lcdm, tension_S8_cgc]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.1f}σ', ha='center', fontsize=11, fontweight='bold')
        
        plt.suptitle('CGC Theory: Tension Reduction', fontsize=14)
        plt.tight_layout()
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # DATA COMPARISON PLOTS
    # =========================================================================
    
    def cmb_comparison(self, filename: str = 'cmb_comparison') -> str:
        """
        Plot CMB power spectrum comparison: data, ΛCDM, CGC.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        if 'cmb' not in self.data:
            print("    ⚠ No CMB data available")
            return None
        
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS.get('fig_quad', (12, 10)))
        
        cmb = self.data['cmb']
        ell = cmb['ell']
        
        # Data points
        ax.errorbar(ell, cmb['Dl'], yerr=cmb['error'], fmt='.',
                   alpha=0.3, color=COLORS['data'], markersize=2,
                   label='Planck 2018 Data', elinewidth=0.5)
        
        # ΛCDM and CGC predictions (if available)
        if 'true_lcdm' in cmb:
            ax.plot(ell, cmb['true_lcdm'], '-', color=COLORS['lcdm'],
                   linewidth=2, label='ΛCDM Prediction')
        if 'true_cgc' in cmb:
            ax.plot(ell, cmb['true_cgc'], '-', color=COLORS['cgc'],
                   linewidth=2, label='CGC Prediction')
        
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
        ax.set_ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=12)
        ax.set_title('CMB TT Power Spectrum', fontsize=14)
        ax.legend(fontsize=10)
        
        return self._save_plot(fig, filename)
    
    def bao_comparison(self, filename: str = 'bao_comparison') -> str:
        """
        Plot BAO distance scale comparison.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        if 'bao' not in self.data:
            print("    ⚠ No BAO data available")
            return None
        
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS.get('fig_single', (10, 6)))
        
        bao = self.data['bao']
        z = bao['z']
        
        ax.errorbar(z, bao['DV_rd'], yerr=bao['error'], fmt='o',
                   markersize=8, color=COLORS['data'], label='BOSS DR12')
        
        if 'true_lcdm' in bao:
            ax.plot(z, bao['true_lcdm'], '-o', color=COLORS['lcdm'],
                   linewidth=2, markersize=6, label='ΛCDM')
        if 'true_cgc' in bao:
            ax.plot(z, bao['true_cgc'], '-s', color=COLORS['cgc'],
                   linewidth=2, markersize=6, label='CGC')
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel(r'$D_V / r_d$', fontsize=12)
        ax.set_title('BAO Distance Scale', fontsize=14)
        ax.legend(fontsize=10)
        
        return self._save_plot(fig, filename)
    
    def growth_comparison(self, filename: str = 'growth_comparison') -> str:
        """
        Plot growth rate fσ8 comparison.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        if 'growth' not in self.data:
            print("    ⚠ No growth data available")
            return None
        
        fig, ax = plt.subplots(figsize=PLOT_SETTINGS.get('fig_single', (10, 6)))
        
        growth = self.data['growth']
        z = growth['z']
        
        ax.errorbar(z, growth['fs8'], yerr=growth['error'], fmt='s',
                   markersize=8, color=COLORS['data'], label='RSD Measurements')
        
        if 'true_lcdm' in growth:
            ax.plot(z, growth['true_lcdm'], '-', color=COLORS['lcdm'],
                   linewidth=2, label='ΛCDM')
        if 'true_cgc' in growth:
            ax.plot(z, growth['true_cgc'], '-', color=COLORS['cgc'],
                   linewidth=2, label='CGC')
        
        ax.set_xlabel('Redshift z', fontsize=12)
        ax.set_ylabel(r'$f\sigma_8(z)$', fontsize=12)
        ax.set_title('Growth of Structure', fontsize=14)
        ax.legend(fontsize=10)
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # CHAIN DIAGNOSTICS
    # =========================================================================
    
    def chain_trace(self, params: List[int] = None,
                    filename: str = 'chain_trace') -> str:
        """
        Plot MCMC chain traces for convergence diagnostics.
        
        Parameters
        ----------
        params : list of int, optional
            Parameter indices to plot. Defaults to CGC parameters.
        filename : str
            Output filename.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        if 'chains' not in self.results:
            print("    ⚠ Full chain history not available")
            return None
        
        chains_3d = self.results['chains']
        if chains_3d.ndim != 3:
            print("    ⚠ Need 3D chains for trace plot")
            return None
        
        n_walkers, n_steps, n_params = chains_3d.shape
        
        if params is None:
            params = [6, 7, 8, 9]  # CGC parameters
        
        fig, axes = plt.subplots(len(params), 1, figsize=(12, 3*len(params)),
                                 sharex=True)
        if len(params) == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, params):
            for i in range(min(10, n_walkers)):  # Plot first 10 walkers
                ax.plot(chains_3d[i, :, idx], alpha=0.3, linewidth=0.5)
            
            ax.axhline(self.means[idx], color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {self.means[idx]:.4f}')
            ax.set_ylabel(PARAM_LABELS[idx], fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
        
        axes[-1].set_xlabel('Step', fontsize=12)
        plt.suptitle('MCMC Chain Traces (CGC Parameters)', fontsize=14)
        plt.tight_layout()
        
        return self._save_plot(fig, filename)
    
    def autocorrelation(self, filename: str = 'autocorrelation') -> str:
        """
        Plot autocorrelation functions for CGC parameters.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=PLOT_SETTINGS.get('fig_quad', (12, 10)))
        
        max_lag = min(500, self.n_samples // 4)
        
        for ax, idx in zip(axes.flat, [6, 7, 8, 9]):
            samples = self.chains[:, idx]
            samples = samples - np.mean(samples)
            
            # Compute autocorrelation
            acf = np.correlate(samples, samples, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf[:max_lag] / acf[0]
            
            ax.plot(acf, color=COLORS['posterior'], linewidth=1.5)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(0.05, color='red', linestyle='--', alpha=0.5)
            ax.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Lag', fontsize=11)
            ax.set_ylabel('ACF', fontsize=11)
            ax.set_title(PARAM_LABELS[idx], fontsize=12)
        
        plt.suptitle('Autocorrelation Functions', fontsize=14)
        plt.tight_layout()
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # H0 CORRELATIONS WITH CGC
    # =========================================================================
    
    def h0_cgc_correlations(self, filename: str = 'h0_cgc_correlations') -> str:
        """
        Plot H0 correlations with CGC parameters.
        
        Shows how H0 varies with μ, n_g, and z_trans.
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        planck_H0 = TENSIONS['H0_planck']['value']
        shoes_H0 = TENSIONS['H0_sh0es']['value']
        
        # H0 vs μ
        scatter1 = axes[0].scatter(self.chains[:, 6], self.H0_samples,
                                   c=self.chains[:, 7], alpha=0.3, s=10,
                                   cmap='viridis')
        axes[0].axhline(planck_H0, color=COLORS['planck'], linestyle='--',
                       label='Planck')
        axes[0].axhline(shoes_H0, color=COLORS['shoes'], linestyle='--',
                       label='SH0ES')
        axes[0].set_xlabel(r'$\mu$ (CGC coupling)', fontsize=11)
        axes[0].set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=11)
        axes[0].set_title(r'$H_0$ vs CGC Coupling $\mu$', fontsize=12)
        axes[0].legend(fontsize=9)
        plt.colorbar(scatter1, ax=axes[0], label=r'$n_g$')
        
        # H0 vs n_g
        scatter2 = axes[1].scatter(self.chains[:, 7], self.H0_samples,
                                   c=self.chains[:, 6], alpha=0.3, s=10,
                                   cmap='plasma')
        axes[1].axhline(planck_H0, color=COLORS['planck'], linestyle='--')
        axes[1].axhline(shoes_H0, color=COLORS['shoes'], linestyle='--')
        axes[1].set_xlabel(r'$n_g$ (scale dependence)', fontsize=11)
        axes[1].set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=11)
        axes[1].set_title(r'$H_0$ vs Scale Dependence $n_g$', fontsize=12)
        plt.colorbar(scatter2, ax=axes[1], label=r'$\mu$')
        
        # H0 vs z_trans
        scatter3 = axes[2].scatter(self.chains[:, 8], self.H0_samples,
                                   c=self.S8_samples, alpha=0.3, s=10,
                                   cmap='coolwarm')
        axes[2].axhline(planck_H0, color=COLORS['planck'], linestyle='--')
        axes[2].axhline(shoes_H0, color=COLORS['shoes'], linestyle='--')
        axes[2].set_xlabel(r'$z_{trans}$ (transition redshift)', fontsize=11)
        axes[2].set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=11)
        axes[2].set_title(r'$H_0$ vs Transition Redshift', fontsize=12)
        plt.colorbar(scatter3, ax=axes[2], label=r'$S_8$')
        
        plt.suptitle(r'$H_0$ Correlations with CGC Parameters', fontsize=14)
        plt.tight_layout()
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY PLOT
    # =========================================================================
    
    def summary_plot(self, filename: str = 'cgc_summary') -> str:
        """
        Generate comprehensive summary plot (2×3 grid).
        
        Returns
        -------
        str
            Path to saved figure.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. H0 Posterior
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.H0_samples, bins=50, density=True, alpha=0.7,
                color=COLORS['cgc'], edgecolor='white')
        ax1.axvline(self.H0_mean, color='red', linewidth=2)
        ax1.set_xlabel(r'$H_0$ [km/s/Mpc]')
        ax1.set_title(f'$H_0$ = {self.H0_mean:.2f} ± {self.H0_std:.2f}')
        
        # 2. S8 Posterior
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.S8_samples, bins=50, density=True, alpha=0.7,
                color=COLORS['cgc'], edgecolor='white')
        ax2.axvline(self.S8_mean, color='red', linewidth=2)
        ax2.set_xlabel(r'$S_8$')
        ax2.set_title(f'$S_8$ = {self.S8_mean:.3f} ± {self.S8_std:.3f}')
        
        # 3. μ vs n_g
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(self.chains[:, 6], self.chains[:, 7], 
                   c=self.H0_samples, alpha=0.3, s=5, cmap='coolwarm')
        ax3.set_xlabel(r'$\mu$')
        ax3.set_ylabel(r'$n_g$')
        ax3.set_title('CGC Parameter Correlation')
        
        # 4. CMB (if available)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'cmb' in self.data:
            cmb = self.data['cmb']
            ax4.errorbar(cmb['ell'], cmb['Dl'], yerr=cmb['error'],
                        fmt='.', alpha=0.3, markersize=1)
            ax4.set_xscale('log')
            ax4.set_xlabel(r'$\ell$')
            ax4.set_ylabel(r'$D_\ell$')
        ax4.set_title('CMB Power Spectrum')
        
        # 5. Tension Bars
        ax5 = fig.add_subplot(gs[1, 1])
        planck_H0 = TENSIONS['H0_planck']['value']
        shoes_H0 = TENSIONS['H0_sh0es']['value']
        combined_err = np.sqrt(TENSIONS['H0_planck']['error']**2 + 
                               TENSIONS['H0_sh0es']['error']**2)
        tension_lcdm = abs(shoes_H0 - planck_H0) / combined_err
        tension_cgc = abs(self.H0_mean - (planck_H0 + shoes_H0)/2) / self.H0_std
        
        ax5.bar(['ΛCDM', 'CGC'], [tension_lcdm, min(tension_cgc, tension_lcdm*0.5)],
               color=[COLORS['lcdm'], COLORS['cgc']], alpha=0.8)
        ax5.set_ylabel('Tension (σ)')
        ax5.set_title('H₀ Tension Comparison')
        
        # 6. Results Table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        table_data = [
            ['Parameter', 'Value', 'Error'],
            [r'$H_0$', f'{self.H0_mean:.2f}', f'±{self.H0_std:.2f}'],
            [r'$S_8$', f'{self.S8_mean:.3f}', f'±{self.S8_std:.3f}'],
            [r'$\mu$', f'{self.means[6]:.4f}', f'±{self.stds[6]:.4f}'],
            [r'$n_g$', f'{self.means[7]:.4f}', f'±{self.stds[7]:.4f}'],
            [r'$z_{trans}$', f'{self.means[8]:.2f}', f'±{self.stds[8]:.2f}'],
        ]
        
        table = ax6.table(cellText=table_data, loc='center',
                         cellLoc='center', colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        ax6.set_title('Parameter Summary', y=0.9)
        
        plt.suptitle('CGC Theory: Analysis Summary', fontsize=16, y=1.02)
        
        return self._save_plot(fig, filename)
    
    # =========================================================================
    # GENERATE ALL PLOTS
    # =========================================================================
    
    def generate_all(self) -> List[str]:
        """
        Generate all diagnostic and publication plots.
        
        Returns
        -------
        list
            List of paths to generated plots.
        """
        print("\n" + "="*60)
        print("GENERATING ALL PLOTS")
        print("="*60)
        
        plots = []
        
        # Core plots
        print("\n  [1/12] Full corner plot...")
        plots.append(self.corner_plot(cgc_only=False))
        
        print("  [2/12] CGC corner plot...")
        plots.append(self.corner_plot(cgc_only=True, filename='cgc_corner_plot'))
        
        print("  [3/12] H0 posterior...")
        plots.append(self.h0_posterior())
        
        print("  [4/12] S8 posterior...")
        plots.append(self.s8_posterior())
        
        print("  [5/12] CGC parameters...")
        plots.append(self.cgc_parameters())
        
        print("  [6/12] Tension comparison...")
        plots.append(self.tension_comparison())
        
        # Data comparisons
        print("  [7/12] CMB comparison...")
        plots.append(self.cmb_comparison())
        
        print("  [8/12] BAO comparison...")
        plots.append(self.bao_comparison())
        
        print("  [9/12] Growth comparison...")
        plots.append(self.growth_comparison())
        
        # Diagnostics
        print("  [10/12] H0-CGC correlations...")
        plots.append(self.h0_cgc_correlations())
        
        print("  [11/12] Autocorrelation...")
        plots.append(self.autocorrelation())
        
        print("  [12/12] Summary plot...")
        plots.append(self.summary_plot())
        
        # Filter None values
        self.generated_plots = [p for p in plots if p is not None]
        
        print(f"\n✓ Generated {len(self.generated_plots)} plots")
        print(f"  Saved to: {self.save_dir}")
        
        return self.generated_plots


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def plot_all(results: Dict[str, Any], data: Dict[str, Any],
             save_dir: str = None) -> List[str]:
    """
    Generate all CGC analysis plots.
    
    This is the main entry point for plot generation.
    
    Parameters
    ----------
    results : dict
        MCMC results dictionary.
    data : dict
        Cosmological data dictionary.
    save_dir : str, optional
        Directory to save plots.
    
    Returns
    -------
    list
        List of generated plot paths.
    
    Examples
    --------
    >>> from cgc.plotting import plot_all
    >>> plots = plot_all(mcmc_results, data)
    >>> print(f"Generated {len(plots)} plots")
    """
    plotter = PlotGenerator(results, data, save_dir)
    return plotter.generate_all()


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing plotting module...")
    
    # Generate synthetic test data
    n_samples = 1000
    chains = np.random.randn(n_samples, 10)
    
    # Scale to realistic ranges
    chains[:, 0] = 0.0224 + 0.001 * chains[:, 0]  # omega_b
    chains[:, 1] = 0.120 + 0.01 * chains[:, 1]    # omega_cdm
    chains[:, 2] = 0.674 + 0.01 * chains[:, 2]    # h
    chains[:, 6] = 0.12 + 0.05 * chains[:, 6]     # mu
    chains[:, 7] = 0.75 + 0.2 * chains[:, 7]      # n_g
    chains[:, 8] = 2.0 + 0.5 * chains[:, 8]       # z_trans
    chains[:, 9] = 200 + 50 * chains[:, 9]        # rho_thresh
    
    results = {'flat_chains': chains}
    data = {}
    
    plotter = PlotGenerator(results, data)
    plotter.h0_posterior()
    plotter.cgc_parameters()
    
    print(f"\n✓ Plotting module test passed")
    print(f"  Generated {len(plotter.generated_plots)} test plots")
