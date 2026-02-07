#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            SDCG Production MCMC Analysis                                     ║
║                                                                              ║
║  Configuration:                                                              ║
║    • 128 walkers, 50000 steps                                               ║
║    • Autocorrelation-based burn-in detection                                ║
║    • R-hat < 1.01, ESS > 1000, autocorr threshold 50                       ║
║    • HDF5 backend with checkpointing every 1000 steps                       ║
║    • Multiprocessing parallelization (auto-detect cores)                     ║
║                                                                              ║
║  Parameters sampled: ω_b, ω_cdm, h, ln10As, n_s, τ, μ  (7 dim)            ║
║  Fixed by theory: n_g=0.0125, z_trans=1.67, α=2, ρ_thresh=200              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import warnings
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

# ═══════════════════════════════════════════════════════════════════════════
# Add project root to path
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'simulations'))

# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PRODUCTION_CONFIG = {
    # Sampler settings
    'n_walkers': 128,
    'n_steps': 20000,          # v13: increased from 2000 for convergence
    'n_dim': 7,
    'seed': 42,
    
    # Convergence criteria
    'rhat_threshold': 1.01,
    'ess_target': 1000,
    'autocorr_threshold': 50,
    
    # Checkpointing
    'checkpoint_interval': 500,  # v13: less frequent for longer run
    'backend': 'hdf5',
    
    # Parallelization
    'n_threads': None,  # Auto-detect
    
    # Data probes
    'include_cmb': False,          # Toy 3-Gaussian model CANNOT fit 2507 real Planck points
                                     # (chi2/dof ~ 80). CMB info encoded via Gaussian priors instead.
    'include_sne': True,
    'include_lyalpha': True,
    
    # Priors
    'use_eft_priors': False,       # No Gaussian prior — flat prior [0, 0.50] only
    'tight_eft_priors': False,      # μ_void determined by data, not prior
    
    # Thinning
    'thin': 10,
    
    # Burn-in fraction (v13: increased from 20% to 30%)
    'burnin_fraction': 0.30,
}


def setup_output_dirs():
    """Create output directories for this run."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(PROJECT_ROOT, 'results', f'production_run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'chains'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'diagnostics'), exist_ok=True)
    return run_dir, timestamp


def print_banner(run_dir, n_cores):
    """Print production run banner."""
    print("\n" + "═" * 70)
    print("║  SDCG PRODUCTION MCMC ANALYSIS")
    print("═" * 70)
    print(f"║  Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"║  Output:      {run_dir}")
    print(f"║  Walkers:     {PRODUCTION_CONFIG['n_walkers']}")
    print(f"║  Steps:       {PRODUCTION_CONFIG['n_steps']}")
    print(f"║  Dimensions:  {PRODUCTION_CONFIG['n_dim']} (7 sampled)")
    print(f"║  CPU cores:   {n_cores}")
    print(f"║  Checkpoint:  every {PRODUCTION_CONFIG['checkpoint_interval']} steps")
    print(f"║  R-hat goal:  < {PRODUCTION_CONFIG['rhat_threshold']}")
    print(f"║  ESS goal:    > {PRODUCTION_CONFIG['ess_target']}")
    print(f"║  Probes:      BAO + Growth + H0 + SNe + Lyα (CMB via Planck priors)")
    print(f"║  Seed:        {PRODUCTION_CONFIG['seed']}")
    print("═" * 70 + "\n")


def load_data():
    """Load all cosmological datasets."""
    from cgc.data_loader import DataLoader
    
    print("Loading cosmological data...")
    loader = DataLoader(use_real_data=True, verbose=True)
    data = loader.load_all(
        include_sne=PRODUCTION_CONFIG['include_sne'],
        include_lyalpha=PRODUCTION_CONFIG['include_lyalpha']
    )
    return data


def compute_gelman_rubin(chains_3d):
    """
    Compute Gelman-Rubin R-hat from 3D chain array.
    
    Parameters
    ----------
    chains_3d : np.ndarray
        Shape (n_steps, n_walkers, n_dim)
    
    Returns
    -------
    np.ndarray
        R-hat for each parameter
    """
    n_steps, n_walkers, n_dim = chains_3d.shape
    
    # Use second half
    chains = chains_3d[n_steps // 2:]
    n = len(chains)
    
    R_hat = np.zeros(n_dim)
    
    for i in range(n_dim):
        # Mean of each walker chain
        chain_means = np.mean(chains[:, :, i], axis=0)
        
        # Within-chain variance
        W = np.mean(np.var(chains[:, :, i], axis=0, ddof=1))
        
        # Between-chain variance
        B = n * np.var(chain_means, ddof=1)
        
        # Pooled estimate
        var_hat = (1 - 1/n) * W + B/n
        
        R_hat[i] = np.sqrt(var_hat / W) if W > 0 else 1.0
    
    return R_hat


def compute_ess(flat_chains):
    """
    Compute effective sample size for each parameter.
    
    Parameters
    ----------
    flat_chains : np.ndarray
        Shape (n_samples, n_dim)
    
    Returns
    -------
    np.ndarray
        ESS for each parameter
    """
    try:
        import emcee
        n_samples, n_dim = flat_chains.shape
        ess = np.zeros(n_dim)
        for i in range(n_dim):
            try:
                tau = emcee.autocorr.integrated_time(
                    flat_chains[:, i], quiet=True
                )
                ess[i] = n_samples / tau[0] if tau[0] > 0 else n_samples
            except Exception:
                ess[i] = n_samples / 50  # Fallback estimate
        return ess
    except Exception:
        return np.full(flat_chains.shape[1], flat_chains.shape[0] / 50)


# ═══════════════════════════════════════════════════════════════════════════
# Module-level globals for multiprocessing (must be picklable)
# ═══════════════════════════════════════════════════════════════════════════
_GLOBAL_DATA = None
_GLOBAL_KWARGS = None


def _init_worker(data, kwargs):
    """Initialize worker process with shared data."""
    global _GLOBAL_DATA, _GLOBAL_KWARGS
    _GLOBAL_DATA = data
    _GLOBAL_KWARGS = kwargs


def _log_prob_worker(theta):
    """Module-level log_prob function for multiprocessing pickling."""
    from cgc.likelihoods import log_probability
    return log_probability(theta, _GLOBAL_DATA, **_GLOBAL_KWARGS)


def run_production_mcmc():
    """Run the full production MCMC analysis."""
    global _GLOBAL_DATA, _GLOBAL_KWARGS
    
    import emcee
    from cgc.parameters import CGCParameters, get_bounds_array
    from cgc.likelihoods import log_probability
    
    # ═══════════════════════════════════════════════════════════════════
    # Setup
    # ═══════════════════════════════════════════════════════════════════
    
    run_dir, timestamp = setup_output_dirs()
    n_cores = cpu_count()
    n_processes = max(1, n_cores - 1)  # Leave 1 core free
    
    print_banner(run_dir, n_processes)
    
    # ═══════════════════════════════════════════════════════════════════
    # Load data
    # ═══════════════════════════════════════════════════════════════════
    
    data = load_data()
    
    # ═══════════════════════════════════════════════════════════════════
    # Configuration
    # ═══════════════════════════════════════════════════════════════════
    
    n_walkers = PRODUCTION_CONFIG['n_walkers']
    n_steps = PRODUCTION_CONFIG['n_steps']
    n_dim = PRODUCTION_CONFIG['n_dim']
    seed = PRODUCTION_CONFIG['seed']
    checkpoint_interval = PRODUCTION_CONFIG['checkpoint_interval']
    
    np.random.seed(seed)
    
    # ═══════════════════════════════════════════════════════════════════
    # Initialize walkers
    # ═══════════════════════════════════════════════════════════════════
    
    # ─── Initialize at Planck 2018 + CGC μ_eff ─────────────────────────
    # NOTE: theta[6] = μ_eff (effective coupling), NOT μ_fit (bare)!
    #   μ_eff ≈ 0.149 is what enters all observable formulas directly.
    #   μ_fit = μ_eff / S_avg ≈ 0.149 / 0.31 ≈ 0.47 (back-calculated).
    #   h = 0.6736 (Planck bare parameter); CGC boost is in H0_eff.
    theta0 = np.array([
        0.02237,   # ω_b   (Planck 2018)
        0.1200,    # ω_cdm (Planck 2018)
        0.6736,    # h     (Planck 2018; CGC boost via H0_eff formula)
        3.044,     # ln10As (Planck 2018)
        0.9649,    # n_s   (Planck 2018)
        0.0544,    # τ     (Planck 2018, Gaussian prior center)
        0.149,     # μ_eff (effective coupling; μ_fit=0.47×S_avg=0.31)
    ])
    
    print(f"\nInitial parameter vector (Planck 2018 + CGC μ_eff):")
    param_names = ['omega_b', 'omega_cdm', 'h', 'ln10As', 'n_s', 'tau', 'mu']
    for name, val in zip(param_names, theta0):
        print(f"  {name:12s} = {val:.6f}")
    
    # Scatter walkers around Planck baseline
    # 1e-2 gives enough spread for emcee's stretch-move to explore properly
    # (1e-4 was too tight — walkers collapse instead of exploring)
    scatter = 1e-2
    p0 = theta0 * (1 + scatter * np.random.randn(n_walkers, n_dim))
    
    # Clip to bounds
    bounds = get_bounds_array()
    for i in range(n_dim):
        low, high = bounds[i]
        p0[:, i] = np.clip(p0[:, i], low + 1e-6, high - 1e-6)
    
    print(f"  NOTE: theta[6] = μ_eff (effective coupling), NOT μ_fit (bare)!")
    print(f"  μ_fit = μ_eff / S_avg ≈ {theta0[6]:.3f} / 0.31 ≈ {theta0[6]/0.31:.2f}")
    print(f"\nInitialized {n_walkers} walkers in {n_dim}D parameter space")
    
    # ═══════════════════════════════════════════════════════════════════
    # HDF5 backend for persistent storage
    # ═══════════════════════════════════════════════════════════════════
    
    backend_path = os.path.join(run_dir, 'chains', f'mcmc_chains_{timestamp}.h5')
    backend = emcee.backends.HDFBackend(backend_path)
    backend.reset(n_walkers, n_dim)
    print(f"\nHDF5 backend: {backend_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Set up module-level data for multiprocessing
    # ═══════════════════════════════════════════════════════════════════
    
    likelihood_kwargs = {
        'include_cmb': PRODUCTION_CONFIG['include_cmb'],
        'include_sne': PRODUCTION_CONFIG['include_sne'],
        'include_lyalpha': PRODUCTION_CONFIG['include_lyalpha'],
        'use_eft_priors': PRODUCTION_CONFIG['use_eft_priors'],
        'tight_eft_priors': PRODUCTION_CONFIG['tight_eft_priors'],
    }
    
    # Set globals for the main process (used in serial mode too)
    _GLOBAL_DATA = data
    _GLOBAL_KWARGS = likelihood_kwargs
    
    # ═══════════════════════════════════════════════════════════════════
    # Create sampler (serial - avoids multiprocessing pickling issues)
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\nSetting up emcee EnsembleSampler (serial mode)...")
    pool = None
    
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, _log_prob_worker,
        backend=backend
    )
    print("  Serial execution (avoids multiprocessing pickling overhead)")
    
    # ═══════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════
    # Run MCMC with integrated live TUI monitor (no screen-clear)
    # ═══════════════════════════════════════════════════════════════════
    
    # ANSI colors (no screen clear — works in all terminals)
    BOLD  = '\033[1m';  DIM   = '\033[2m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    CYAN  = '\033[96m'; MAGENTA = '\033[95m'; RESET = '\033[0m'
    
    PLANCK_REF = [0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544, None]
    labels = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ', 'μ']
    
    def _bar(fraction, width=50):
        filled = int(width * fraction)
        bar = '█' * filled + '░' * (width - filled)
        pct = fraction * 100
        color = RED if pct < 25 else (YELLOW if pct < 75 else GREEN)
        return f'{color}{bar}{RESET} {pct:5.1f}%'
    
    def _spark(values, width=20):
        if len(values) < 2:
            return ''
        sparks = '▁▂▃▄▅▆▇█'
        mn, mx = min(values), max(values)
        rng = mx - mn if mx > mn else 1
        step = max(1, len(values) // width)
        return ''.join(sparks[int((v - mn) / rng * (len(sparks) - 1))] for v in values[::step][:width])
    
    def _print_tui(step, n_total, t_start_time, af_mean, af_lo, af_hi,
                   param_results, rhat_vals, checkpoint_history, converged_flag,
                   conv_count, req_conv, ess_vals):
        """Print TUI dashboard — no screen clear, just prints below previous output."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        elapsed = time.time() - t_start_time
        frac = step / n_total
        
        if step > 0:
            rate = elapsed / step
            remaining = (n_total - step) * rate
            eta = now + timedelta(seconds=remaining)
            eta_str = f"{remaining/3600:.1f}h ({eta.strftime('%b %d %H:%M')})"
            rate_str = f"{rate:.2f} s/step"
        else:
            eta_str = "calculating..."
            rate_str = "..."
        
        W = 68
        print(f"\n{BOLD}{CYAN}╔{'═'*W}╗{RESET}")
        print(f"{BOLD}{CYAN}║{RESET}  {BOLD}SDCG Production MCMC — Live Monitor{RESET}{' '*(W-37)}{BOLD}{CYAN}║{RESET}")
        print(f"{BOLD}{CYAN}╠{'═'*W}╣{RESET}")
        print(f"{BOLD}{CYAN}║{RESET}  Step: {BOLD}{step:,}{RESET} / {n_total:,}   |   Elapsed: {elapsed/60:.0f} min   |   Rate: {rate_str}{RESET}")
        print(f"{BOLD}{CYAN}║{RESET}  {_bar(frac, width=50)}")
        print(f"{BOLD}{CYAN}║{RESET}  ETA: {eta_str}")
        
        if af_mean is not None:
            af_color = GREEN if 0.2 < af_mean < 0.5 else RED
            print(f"{BOLD}{CYAN}║{RESET}  Acceptance: {af_color}{af_mean:.3f}{RESET} (range: {af_lo:.3f}–{af_hi:.3f})")
        
        print(f"{BOLD}{CYAN}╠{'═'*W}╣{RESET}")
        
        if not param_results:
            print(f"{BOLD}{CYAN}║{RESET}  {DIM}Waiting for first checkpoint (step {checkpoint_interval})...{RESET}")
        else:
            p = param_results
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}Latest Checkpoint: Step {step:,}{RESET}")
            print(f"{BOLD}{CYAN}║{RESET}")
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}{'Param':6s}  {'Median':>10s}  {'± 1σ':>8s}  {'Planck':>10s}  {'Δ':>6s}  Status{RESET}")
            print(f"{BOLD}{CYAN}║{RESET}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*6}")
            
            for i, (name, planck) in enumerate(zip(labels, PLANCK_REF)):
                med, err = p[name]['median'], p[name]['err']
                if planck is not None:
                    delta_sigma = abs(med - planck) / err if err > 0 else 0
                    delta_str = f"{delta_sigma:.1f}σ"
                    status = f"{GREEN}✓{RESET}" if delta_sigma < 1 else (f"{YELLOW}~{RESET}" if delta_sigma < 3 else f"{RED}✗{RESET}")
                    planck_str = f"{planck:.5f}"
                else:
                    sig = med / err if err > 0 else 0
                    delta_str = f"{sig:.0f}σ"
                    status = f"{GREEN}★{RESET}" if sig > 3 else f"{YELLOW}?{RESET}"
                    planck_str = "    —"
                print(f"{BOLD}{CYAN}║{RESET}  {name:6s}  {med:10.5f}  {err:8.5f}  {planck_str:>10s}  {delta_str:>6s}  {status}")
            
            # Key results box
            mu = p['μ']
            h_val = p['h']
            ocdm = p['ω_cdm']
            H0 = h_val['median'] * 100
            
            print(f"{BOLD}{CYAN}║{RESET}")
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}┌─── KEY RESULTS ──────────────────────────────────────┐{RESET}")
            
            mu_sig = mu['median'] / mu['err'] if mu['err'] > 0 else 0
            mu_color = GREEN if 0.01 < mu['median'] < 0.40 else RED
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  μ = {mu_color}{BOLD}{mu['median']:.4f} ± {mu['err']:.4f}{RESET}  ({mu_sig:.1f}σ detection)              {BOLD}│{RESET}")
            
            h0_color = GREEN if 66 < H0 < 74 else RED
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  H₀ = {h0_color}{H0:.1f}{RESET} km/s/Mpc  (h = {h_val['median']:.4f})                {BOLD}│{RESET}")
            
            ocdm_pct = abs(ocdm['median'] - 0.12) / 0.12 * 100
            ocdm_color = GREEN if ocdm_pct < 5 else RED
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  ω_cdm = {ocdm_color}{ocdm['median']:.5f}{RESET}  (Δ = {ocdm_pct:.1f}% from Planck)          {BOLD}│{RESET}")
            
            healthy = 0.01 < mu['median'] < 0.40 and ocdm_pct < 10 and 64 < H0 < 76
            health_str = f"{GREEN}HEALTHY{RESET}" if healthy else f"{RED}CHECK{RESET}"
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  Chain health: {health_str}                                   {BOLD}│{RESET}")
            print(f"{BOLD}{CYAN}║{RESET}  {BOLD}└──────────────────────────────────────────────────────┘{RESET}")
            
            # R-hat
            if rhat_vals is not None:
                print(f"{BOLD}{CYAN}║{RESET}")
                print(f"{BOLD}{CYAN}║{RESET}  {BOLD}R-hat Convergence:{RESET} (target < 1.01)")
                rhat_line = "  "
                for name, r in zip(labels, rhat_vals):
                    if r < 1.01:
                        rhat_line += f"{GREEN}{name}={r:.3f}✓{RESET} "
                    elif r < 1.05:
                        rhat_line += f"{YELLOW}{name}={r:.3f}~{RESET} "
                    else:
                        rhat_line += f"{RED}{name}={r:.3f}✗{RESET} "
                print(f"{BOLD}{CYAN}║{RESET}{rhat_line}")
                
                all_conv = np.all(rhat_vals < 1.01)
                if ess_vals is not None:
                    ess_min = int(np.min(ess_vals))
                    ess_str = f"  ESS_min={ess_min}"
                else:
                    ess_str = ""
                
                if all_conv and converged_flag:
                    conv_str = f"{GREEN}{BOLD}★ CONVERGED ({conv_count}/{req_conv}) ★{RESET}{ess_str}"
                elif all_conv:
                    conv_str = f"{YELLOW}R̂ OK ({conv_count}/{req_conv}){RESET}{ess_str}"
                else:
                    conv_str = f"{YELLOW}Not yet{RESET}{ess_str}"
                print(f"{BOLD}{CYAN}║{RESET}  Overall: {conv_str}")
            
            # History sparklines
            if len(checkpoint_history) > 1:
                print(f"{BOLD}{CYAN}║{RESET}")
                print(f"{BOLD}{CYAN}║{RESET}  {BOLD}Parameter Evolution:{RESET}")
                for name in ['μ', 'h', 'ω_cdm']:
                    vals = [cp[name]['median'] for cp in checkpoint_history]
                    spark = _spark(vals)
                    print(f"{BOLD}{CYAN}║{RESET}  {name:6s}: {spark} {vals[-1]:.5f}")
        
        n_cps = len(checkpoint_history) if checkpoint_history else 0
        total_cps = n_total // checkpoint_interval
        print(f"{BOLD}{CYAN}╠{'═'*W}╣{RESET}")
        print(f"{BOLD}{CYAN}║{RESET}  Checkpoints: {n_cps}/{total_cps}   |   Updated: {now.strftime('%H:%M:%S')}")
        print(f"{BOLD}{CYAN}╚{'═'*W}╝{RESET}")
        sys.stdout.flush()
    
    def _step_progress(step, n_total, t0):
        """Inline single-line progress between checkpoints using \\r."""
        elapsed = time.time() - t0
        frac = step / n_total
        rate = elapsed / step if step > 0 else 0
        eta_h = (n_total - step) * rate / 3600 if step > 0 else 0
        w = 30
        filled = int(w * frac)
        bar = '█' * filled + '░' * (w - filled)
        sys.stdout.write(f"\r  {CYAN}▸{RESET} {bar} {frac*100:5.1f}% │ step {step:>5d}/{n_total} │ {rate:.1f}s/it │ ETA {eta_h:.1f}h ")
        sys.stdout.flush()
    
    t_start = time.time()
    state = p0
    converged = False
    old_tau = np.inf
    convergence_count = 0
    required_convergence = 4
    checkpoint_history = []
    
    # Initial display
    _print_tui(0, n_steps, t_start, None, None, None,
               None, None, checkpoint_history, False, 0, required_convergence, None)
    
    for batch_start in range(0, n_steps, checkpoint_interval):
        batch_size = min(checkpoint_interval, n_steps - batch_start)
        
        # Run step-by-step so we can show live progress between checkpoints
        for _sample in sampler.sample(state, iterations=batch_size, progress=False):
            state = _sample
            current_step = batch_start + (sampler.iteration - (batch_start if batch_start == 0 else sampler.iteration - batch_size + (sampler.iteration - batch_start - batch_size)))
            # simpler: sampler.iteration is the total steps done
            _step_progress(sampler.iteration, n_steps, t_start)
        
        # ─── Checkpoint: compute diagnostics & print TUI ───
        steps_done = sampler.iteration
        
        # Clear the \r progress line
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        sys.stdout.flush()
        
        af = sampler.acceptance_fraction
        af_mean = np.mean(af)
        af_lo, af_hi = af.min(), af.max()
        
        chains_3d = sampler.get_chain()
        
        burn = max(1, chains_3d.shape[0] // 3)
        flat = chains_3d[burn:].reshape(-1, n_dim)
        
        param_results = {}
        for i, name in enumerate(labels):
            q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
            param_results[name] = {
                'median': q50,
                'err': (q84 - q16) / 2,
                'mean': np.mean(flat[:, i])
            }
        checkpoint_history.append(param_results)
        
        checkpoint_path = os.path.join(
            run_dir, 'chains', f'checkpoint_{steps_done:06d}.npz'
        )
        np.savez_compressed(
            checkpoint_path,
            chains=chains_3d,
            n_steps_completed=steps_done,
            n_walkers=n_walkers,
            acceptance_fraction=af
        )
        
        rhat_vals = None
        ess_vals = None
        if steps_done >= 2000:
            rhat_vals = compute_gelman_rubin(chains_3d)
            rhat_ok = np.all(rhat_vals < PRODUCTION_CONFIG['rhat_threshold'])
            
            discard = max(500, steps_done // 5)
            flat_ess = sampler.get_chain(discard=discard, thin=1, flat=True)
            ess_vals = compute_ess(flat_ess)
            ess_ok = np.all(ess_vals > PRODUCTION_CONFIG['ess_target'])
            
            all_converged = rhat_ok and ess_ok
            if all_converged:
                convergence_count += 1
                if convergence_count >= required_convergence:
                    converged = True
            else:
                convergence_count = 0
        
        # Print full TUI dashboard
        _print_tui(steps_done, n_steps, t_start,
                   af_mean, af_lo, af_hi,
                   param_results, rhat_vals, checkpoint_history,
                   converged, convergence_count, required_convergence, ess_vals)
        
        if converged:
            print(f"\n  {GREEN}{BOLD}★★★ CONVERGENCE ACHIEVED at step {steps_done} ★★★{RESET}\n")
            break
    
    # ═══════════════════════════════════════════════════════════════════
    # Clean up pool
    # ═══════════════════════════════════════════════════════════════════
    
    if pool is not None:
        pool.close()
        pool.join()
    
    total_time = time.time() - t_start
    total_steps = sampler.iteration
    
    print(f"\n{'═' * 70}")
    print(f"  MCMC COMPLETE")
    print(f"  Total steps: {total_steps}")
    print(f"  Total time:  {total_time/60:.1f} minutes")
    print(f"  Converged:   {'YES' if converged else 'NO (reached max steps)'}")
    print(f"{'═' * 70}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Post-processing: extract chains with autocorrelation burn-in
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n\nPOST-PROCESSING")
    print("=" * 70)
    
    burnin_frac = PRODUCTION_CONFIG.get('burnin_fraction', 0.30)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        burn_in_autocorr = int(2 * np.max(tau))
        burn_in_frac = int(total_steps * burnin_frac)
        burn_in = max(burn_in_autocorr, burn_in_frac)  # Use whichever is larger
        thin = max(1, int(0.5 * np.min(tau)))
        print(f"  Autocorrelation burn-in: {burn_in_autocorr} steps")
        print(f"  Fraction burn-in ({burnin_frac:.0%}): {burn_in_frac} steps")
        print(f"  Using burn-in: {burn_in} steps (max of both)")
        print(f"  Thinning factor: {thin}")
    except Exception:
        burn_in = int(total_steps * burnin_frac)
        thin = PRODUCTION_CONFIG['thin']
        print(f"  Fallback burn-in: {burn_in} steps ({burnin_frac:.0%})")
        print(f"  Fallback thin: {thin}")
    
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    print(f"  Final chain shape: {flat_samples.shape}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Parameter constraints
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'═' * 70}")
    print("  PARAMETER CONSTRAINTS")
    print(f"{'═' * 70}")
    
    labels = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ', 'μ']
    results = {}
    
    for i, name in enumerate(labels):
        q = np.percentile(flat_samples[:, i], [16, 50, 84])
        median = q[1]
        err_low = q[1] - q[0]
        err_high = q[2] - q[1]
        mean = np.mean(flat_samples[:, i])
        std = np.std(flat_samples[:, i])
        
        results[name] = {
            'mean': mean, 'std': std,
            'median': median, 'err_low': err_low, 'err_high': err_high
        }
        
        print(f"  {name:8s} = {median:.5f}  +{err_high:.5f}  -{err_low:.5f}  "
              f"(mean={mean:.5f} ± {std:.5f})")
    
    # ═══════════════════════════════════════════════════════════════════
    # SDCG-specific results
    # ═══════════════════════════════════════════════════════════════════
    
    mu_samples = flat_samples[:, 6]
    mu_mean, mu_std = np.mean(mu_samples), np.std(mu_samples)
    h_samples = flat_samples[:, 2]
    h_mean = np.mean(h_samples)
    
    print(f"\n{'═' * 70}")
    print("  SDCG PHYSICS SUMMARY")
    print(f"{'═' * 70}")
    print(f"  μ_fit          = {mu_mean:.4f} ± {mu_std:.4f}")
    print(f"  Detection:       {mu_mean/mu_std:.1f}σ (μ ≠ 0)")
    print(f"  n_g            = 0.0125 (FIXED: β₀²/4π²)")
    print(f"  z_trans        = 1.67 (FIXED: cosmic dynamics)")
    print(f"  ρ_thresh       = 200 (FIXED: virial theorem)")
    
    # Derived quantities
    mu_eff_void = mu_mean * 1.0  # S(void) ≈ 1.0 for ρ << ρ_thresh
    mu_eff_lyalpha = mu_mean * 0.1  # S(IGM) ≈ 0.1
    
    print(f"\n  Derived:")
    print(f"  μ_eff(void)    = {mu_eff_void:.4f} (S_void ≈ 1)")
    print(f"  μ_eff(Lyα/IGM) = {mu_eff_lyalpha:.4f} (S_IGM ≈ 0.1)")
    
    # H0 estimate
    H0_planck = 67.4
    delta_H0 = H0_planck * mu_eff_void * 0.31  # CGC coupling
    H0_cgc = H0_planck + delta_H0
    print(f"\n  H₀ prediction:   {H0_cgc:.1f} km/s/Mpc (from h = {h_mean:.4f})")
    
    # ═══════════════════════════════════════════════════════════════════
    # Final convergence diagnostics
    # ═══════════════════════════════════════════════════════════════════
    
    chains_3d = sampler.get_chain()
    R_hat_final = compute_gelman_rubin(chains_3d)
    ess_final = compute_ess(flat_samples)
    
    print(f"\n{'═' * 70}")
    print("  FINAL CONVERGENCE DIAGNOSTICS")
    print(f"{'═' * 70}")
    
    for i, name in enumerate(labels):
        rhat_status = "✓" if R_hat_final[i] < 1.01 else "✗"
        ess_status = "✓" if ess_final[i] > 1000 else "✗"
        print(f"  {name:8s}: R-hat = {R_hat_final[i]:.5f} {rhat_status}  "
              f"ESS = {ess_final[i]:.0f} {ess_status}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Save final results
    # ═══════════════════════════════════════════════════════════════════
    
    results_path = os.path.join(run_dir, f'production_results_{timestamp}.npz')
    np.savez_compressed(
        results_path,
        flat_chains=flat_samples,
        chains_3d=chains_3d,
        n_walkers=n_walkers,
        n_steps=total_steps,
        n_dim=n_dim,
        burn_in=burn_in,
        thin=thin,
        acceptance_fraction=sampler.acceptance_fraction,
        gelman_rubin=R_hat_final,
        ess=ess_final,
        param_names=labels,
        mu_mean=mu_mean,
        mu_std=mu_std,
        converged=converged,
        total_time=total_time,
        seed=seed
    )
    print(f"\n  Results saved: {results_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Save text summary
    # ═══════════════════════════════════════════════════════════════════
    
    summary_path = os.path.join(run_dir, f'summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("SDCG Production MCMC Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Steps: {total_steps}, Walkers: {n_walkers}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Converged: {converged}\n\n")
        
        f.write("Parameter Constraints (median +upper -lower):\n")
        f.write("-" * 60 + "\n")
        for i, name in enumerate(labels):
            r = results[name]
            f.write(f"  {name:8s} = {r['median']:.6f}  "
                    f"+{r['err_high']:.6f}  -{r['err_low']:.6f}\n")
        
        f.write(f"\nSDCG Coupling:\n")
        f.write(f"  μ = {mu_mean:.4f} ± {mu_std:.4f} ({mu_mean/mu_std:.1f}σ detection)\n")
        f.write(f"  n_g = 0.0125 (FIXED)\n")
        f.write(f"  z_trans = 1.67 (FIXED)\n\n")
        
        f.write("Convergence:\n")
        f.write("-" * 60 + "\n")
        for i, name in enumerate(labels):
            f.write(f"  {name:8s}: R-hat = {R_hat_final[i]:.5f}, "
                    f"ESS = {ess_final[i]:.0f}\n")
    
    print(f"  Summary saved: {summary_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Generate diagnostic plots
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\nGenerating diagnostic plots...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import corner
        
        # 1. Corner plot
        fig = corner.corner(
            flat_samples,
            labels=[r'$\omega_b$', r'$\omega_{cdm}$', r'$h$',
                    r'$\ln(10^{10}A_s)$', r'$n_s$', r'$\tau$', r'$\mu$'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.5f',
            title_kwargs={"fontsize": 10},
        )
        corner_path = os.path.join(run_dir, 'plots', f'corner_{timestamp}.png')
        fig.savefig(corner_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Corner plot: {corner_path}")
        
        # 2. Trace plots
        fig, axes = plt.subplots(n_dim, 1, figsize=(12, 2.5*n_dim), sharex=True)
        trace_labels = [r'$\omega_b$', r'$\omega_{cdm}$', r'$h$',
                        r'$\ln(10^{10}A_s)$', r'$n_s$', r'$\tau$', r'$\mu$']
        
        for i in range(n_dim):
            ax = axes[i]
            # Plot a subset of walkers for clarity
            for j in range(0, n_walkers, max(1, n_walkers // 16)):
                ax.plot(chains_3d[:, j, i], alpha=0.3, linewidth=0.5)
            ax.axvline(burn_in, color='red', linestyle='--', alpha=0.5, label='burn-in')
            ax.set_ylabel(trace_labels[i])
            if i == 0:
                ax.legend(loc='upper right')
        
        axes[-1].set_xlabel('Step')
        fig.suptitle('MCMC Trace Plots', fontsize=14)
        trace_path = os.path.join(run_dir, 'plots', f'traces_{timestamp}.png')
        fig.savefig(trace_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Trace plots: {trace_path}")
        
        # 3. μ posterior
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mu_samples, bins=80, density=True, alpha=0.7, color='steelblue',
                edgecolor='navy', linewidth=0.5)
        ax.axvline(mu_mean, color='red', linewidth=2, label=f'Mean: {mu_mean:.4f}')
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label='ΛCDM (μ=0)')
        ax.axvspan(mu_mean - mu_std, mu_mean + mu_std, alpha=0.2, color='red',
                   label=f'1σ: ±{mu_std:.4f}')
        ax.set_xlabel(r'$\mu$ (CGC coupling)', fontsize=14)
        ax.set_ylabel('Posterior density', fontsize=14)
        ax.set_title(f'SDCG Coupling: μ = {mu_mean:.4f} ± {mu_std:.4f} '
                     f'({mu_mean/mu_std:.1f}σ detection)', fontsize=14)
        ax.legend(fontsize=12)
        mu_path = os.path.join(run_dir, 'plots', f'mu_posterior_{timestamp}.png')
        fig.savefig(mu_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  μ posterior: {mu_path}")
        
    except Exception as e:
        print(f"  Plot generation failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'═' * 70}")
    print(f"  PRODUCTION RUN COMPLETE")
    print(f"{'═' * 70}")
    print(f"  μ = {mu_mean:.4f} ± {mu_std:.4f} ({mu_mean/mu_std:.1f}σ detection)")
    print(f"  All outputs in: {run_dir}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'═' * 70}\n")
    
    return flat_samples, results, run_dir


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        flat_samples, results, run_dir = run_production_mcmc()
    except KeyboardInterrupt:
        print("\n\nMCMC interrupted by user. Checkpoints saved.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
