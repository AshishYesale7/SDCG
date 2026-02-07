#!/usr/bin/env python3
"""
Publication-quality plots for advanced CGC theory features.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'font.family': 'serif'
})

# Import CGC theory
from cgc_advanced_theory import AdvancedCGCTheory

def main():
    cgc = AdvancedCGCTheory(mu=0.149, n_g=0.8, z_trans=1.64)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # =======================================================================
    # Panel 1: Scale-dependent G(k)
    # =======================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    k = np.logspace(-2, 1, 100)  # h/Mpc
    
    for z, color, ls in [(0.0, 'blue', '-'), (0.5, 'green', '--'), 
                          (1.0, 'orange', '-.'), (2.0, 'red', ':')]:
        G_ratio = cgc.G_eff(k, z)
        ax1.semilogx(k, (G_ratio - 1) * 100, color=color, ls=ls, lw=2, label=f'z = {z}')
    
    ax1.axhline(0, color='gray', ls='--', alpha=0.5)
    ax1.axvline(cgc.k_cgc, color='purple', ls=':', lw=1.5, alpha=0.7, label=f'$k_{{CGC}}$ = {cgc.k_cgc:.2f}')
    
    ax1.set_xlabel('$k$ [h/Mpc]')
    ax1.set_ylabel(r'$\Delta G/G_N$ [%]')
    ax1.set_title(r'(a) Scale-Dependent Gravitational Coupling $G_{eff}(k,z)$')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_xlim(0.01, 10)
    ax1.set_ylim(-5, 80)
    
    # Annotation
    ax1.annotate(r'$G_{eff} = G_N[1 + \mu (k/k_{CGC})^{n_g} W(z)]$',
                xy=(0.02, 60), fontsize=10, style='italic')
    ax1.annotate(f'$n_g$ = {cgc.n_g}', xy=(0.02, 50), fontsize=10)
    
    # =======================================================================
    # Panel 2: Chameleon Screening
    # =======================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    rho = np.logspace(-30, 5, 200)  # kg/m^3
    S = np.array([cgc.chameleon_screening(r) for r in rho])
    
    ax2.semilogx(rho, S, 'k-', lw=2.5)
    
    # Mark regions
    ax2.axvspan(1e-30, 1e-20, alpha=0.2, color='blue', label='Cosmological')
    ax2.axvspan(1e-5, 1e5, alpha=0.2, color='red', label='Laboratory/Solar System')
    
    ax2.axvline(cgc.rho_thresh, color='green', ls='--', lw=2, label=r'$\rho_{thresh}$')
    
    ax2.set_xlabel(r'$\rho$ [kg/m³]')
    ax2.set_ylabel('Screening factor $S(\\rho)$')
    ax2.set_title('(b) Chameleon Screening Function')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_xlim(1e-30, 1e5)
    ax2.set_ylim(-0.05, 1.1)
    
    # Annotations
    ax2.annotate('CGC active', xy=(1e-28, 0.9), fontsize=10, color='blue', weight='bold')
    ax2.annotate('CGC screened', xy=(1e0, 0.1), fontsize=10, color='red', weight='bold')
    
    # =======================================================================
    # Panel 3: RG Running of N_eff
    # =======================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    T = np.logspace(-4, 7, 200)  # eV
    N_eff = np.array([cgc.N_eff_running(t) for t in T])
    
    ax3.semilogx(T, N_eff, 'k-', lw=2.5)
    ax3.axhline(3.044, color='gray', ls='--', lw=1, label='SM: $N_{eff}^{SM}$ = 3.044')
    
    # Mark epochs
    T_BBN = 1e6
    T_CMB = 0.26
    ax3.axvline(T_BBN, color='orange', ls=':', lw=2, alpha=0.7)
    ax3.axvline(T_CMB, color='purple', ls=':', lw=2, alpha=0.7)
    
    ax3.annotate('BBN', xy=(T_BBN*1.5, 3.13), fontsize=10, color='orange')
    ax3.annotate('CMB', xy=(T_CMB*0.2, 3.13), fontsize=10, color='purple')
    
    # Constraint bands
    ax3.fill_between([T.min(), T.max()], [2.99-0.17]*2, [2.99+0.17]*2, 
                     alpha=0.15, color='green', label='Planck 2018: $2.99 \\pm 0.17$')
    
    ax3.set_xlabel('Temperature $T$ [eV]')
    ax3.set_ylabel('$N_{eff}(T)$')
    ax3.set_title('(c) RG Running of Effective Relativistic Species')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_xlim(1e-4, 1e7)
    ax3.set_ylim(2.8, 3.2)
    
    # =======================================================================
    # Panel 4: Scale-dependent Growth Rate
    # =======================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    k = np.logspace(-2, 0.5, 50)
    
    for z, color, marker in [(0.3, 'blue', 'o'), (0.5, 'green', 's'), 
                              (0.7, 'orange', '^'), (1.0, 'red', 'd')]:
        f_CGC = np.array([cgc.growth_rate(kk, z) for kk in k])
        
        # GR value (scale-independent)
        Omega_m_z = 0.315 * (1+z)**3 / (0.315*(1+z)**3 + 0.685)
        f_GR = Omega_m_z**0.55
        
        deviation = (f_CGC / f_GR - 1) * 100
        ax4.semilogx(k, deviation, color=color, lw=2, label=f'z = {z}')
    
    ax4.axhline(0, color='gray', ls='--', alpha=0.5)
    
    # Typical RSD measurement uncertainties
    ax4.fill_between([0.01, 3], [-5, -5], [5, 5], alpha=0.1, color='gray',
                     label='Typical RSD uncertainty')
    
    ax4.set_xlabel('$k$ [h/Mpc]')
    ax4.set_ylabel(r'$\Delta f / f_{GR}$ [%]')
    ax4.set_title(f'(d) Scale-Dependent Growth Rate ($n_g$ = {cgc.n_g})')
    ax4.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax4.set_xlim(0.01, 3)
    ax4.set_ylim(-10, 5)
    
    # =======================================================================
    # Overall title
    # =======================================================================
    fig.suptitle('CGC Theory: Advanced Physical Features\n' +
                 r'$\mu = 0.149$, $n_g = 0.8$, $z_{trans} = 1.64$',
                 fontsize=14, weight='bold', y=0.98)
    
    plt.savefig('/Users/ashishyesale/Videos/EDU/Hypothesis/MCMC_cgc/plots/cgc_advanced_features.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig('/Users/ashishyesale/Videos/EDU/Hypothesis/MCMC_cgc/plots/cgc_advanced_features.pdf',
                bbox_inches='tight')
    print("✓ Saved: plots/cgc_advanced_features.png")
    print("✓ Saved: plots/cgc_advanced_features.pdf")
    
    # =======================================================================
    # Figure 2: Hydrodynamical simulation predictions
    # =======================================================================
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Matter power spectrum enhancement
    ax = axes[0]
    k = np.logspace(-2, 1, 100)
    
    for z, color in [(0.0, 'blue'), (1.0, 'orange'), (2.0, 'red'), (3.0, 'purple')]:
        G = cgc.G_eff(k, z)
        # P(k) ∝ G^2 in linear regime
        P_ratio = G**2
        ax.semilogx(k, P_ratio, color=color, lw=2, label=f'z = {z}')
    
    ax.axhline(1, color='gray', ls='--')
    ax.set_xlabel('$k$ [h/Mpc]')
    ax.set_ylabel(r'$P_{CGC}(k)/P_{\Lambda CDM}(k)$')
    ax.set_title('Matter Power Spectrum Enhancement')
    ax.legend(loc='upper left')
    ax.set_xlim(0.01, 10)
    ax.set_ylim(0.95, 2.5)
    
    # Right: Required simulation parameters
    ax = axes[1]
    ax.axis('off')
    
    text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║     HYDRODYNAMICAL SIMULATION REQUIREMENTS FOR CGC           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Box Size:     ≥ 500 Mpc/h                                   ║
    ║  Resolution:   ≤ 100 kpc/h (baryons)                         ║
    ║  Particles:    ≥ 2 × 2048³                                   ║
    ║                                                              ║
    ║  Modified Physics:                                           ║
    ║  ──────────────────────────────────────────────────────────  ║
    ║  • Scale-dependent Poisson: ∇²Φ = 4πG_eff(k,z,ρ) ρ_m         ║
    ║  • Chameleon screening: S(ρ) = exp(-ρ/ρ_thresh)              ║
    ║  • Modified growth: D(k,z) with n_g ≈ 0.8                    ║
    ║                                                              ║
    ║  Key Observables to Measure:                                 ║
    ║  ──────────────────────────────────────────────────────────  ║
    ║  1. Matter P(k) enhancement at k > 0.1 h/Mpc                 ║
    ║  2. Halo mass function shift at M > 10¹³ M☉                  ║
    ║  3. Lyman-α flux P_F(k) at z = 2-4                           ║
    ║  4. Void profiles and abundances                             ║
    ║  5. Galaxy clustering fσ8(k)                                 ║
    ║                                                              ║
    ║  Recommended Codes:                                          ║
    ║  ──────────────────────────────────────────────────────────  ║
    ║  • AREPO (MHD + MG module)                                   ║
    ║  • GADGET-4 (CGC extension)                                  ║
    ║  • MP-GADGET (Lyman-α)                                       ║
    ║  • Nyx (IGM physics)                                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/ashishyesale/Videos/EDU/Hypothesis/MCMC_cgc/plots/cgc_hydro_requirements.png',
                dpi=200, bbox_inches='tight', facecolor='white')
    print("✓ Saved: plots/cgc_hydro_requirements.png")
    
    print("\n" + "="*60)
    print("ADVANCED CGC PLOTS GENERATED")
    print("="*60)


if __name__ == "__main__":
    main()
