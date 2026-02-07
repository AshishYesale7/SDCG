#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║           SDCG MCMC LIVE PROGRESS MONITOR                           ║
║  Auto-refreshes every 30s, shows visual bars & checkpoint results   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, glob, re
import numpy as np
from datetime import datetime, timedelta

PROJECT = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT, 'mcmc_run.log')
RESULTS_BASE = os.path.join(PROJECT, 'results')
TOTAL_STEPS = 20000
CP_INTERVAL = 500
LABELS = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ', 'μ']
PLANCK = [0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544, None]

CLEAR = '\033[2J\033[H'
BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BG_GREEN = '\033[42m'
BG_YELLOW = '\033[43m'
BG_BLUE = '\033[44m'


def progress_bar(fraction, width=40, label=''):
    filled = int(width * fraction)
    bar = '█' * filled + '░' * (width - filled)
    pct = fraction * 100
    if pct < 25:
        color = RED
    elif pct < 75:
        color = YELLOW
    else:
        color = GREEN
    return f'{color}{bar}{RESET} {pct:5.1f}% {label}'


def spark_line(values, width=20):
    """Tiny sparkline for a series of values."""
    if len(values) < 2:
        return ''
    sparks = '▁▂▃▄▅▆▇█'
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    line = ''
    step = max(1, len(values) // width)
    for v in values[::step][:width]:
        idx = int((v - mn) / rng * (len(sparks) - 1))
        line += sparks[idx]
    return line


def get_current_step():
    """Parse current step from log file."""
    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()
        # Find last progress bar line like "12%|█▏ | 58/500 [04:21..."
        matches = re.findall(r'(\d+)/(\d+)\s*\[', content)
        if matches:
            current_in_batch, batch_size = int(matches[-1][0]), int(matches[-1][1])
            # Find completed checkpoints
            cp_matches = re.findall(r'Checkpoint (\d+)/\d+', content)
            completed_steps = int(cp_matches[-1]) if cp_matches else 0
            return completed_steps + current_in_batch
        return 0
    except:
        return 0


def get_acceptance():
    """Parse latest acceptance from log."""
    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()
        matches = re.findall(r'Acceptance: ([\d.]+) \(range: ([\d.]+)-([\d.]+)\)', content)
        if matches:
            return float(matches[-1][0]), float(matches[-1][1]), float(matches[-1][2])
    except:
        pass
    return None, None, None


def get_run_dir():
    run_dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, 'production_run_*')))
    return run_dirs[-1] if run_dirs else None


def compute_rhat(chains_3d):
    n_steps, n_walkers, n_dim = chains_3d.shape
    chains = chains_3d[n_steps // 2:]
    n = len(chains)
    R = np.zeros(n_dim)
    for i in range(n_dim):
        cm = np.mean(chains[:, :, i], axis=0)
        W = np.mean(np.var(chains[:, :, i], axis=0, ddof=1))
        B = n * np.var(cm, ddof=1)
        v = (1 - 1/n) * W + B/n
        R[i] = np.sqrt(v / W) if W > 0 else 1.0
    return R


def analyze_checkpoint(cp_file):
    """Analyze a single checkpoint file."""
    data = np.load(cp_file, allow_pickle=True)
    chains_3d = data['chains']
    n_steps_done = int(data['n_steps_completed'])
    af = data['acceptance_fraction']
    n_steps, n_walkers, n_dim = chains_3d.shape
    
    burn = max(1, n_steps // 3)
    flat = chains_3d[burn:].reshape(-1, n_dim)
    
    results = {}
    for i, name in enumerate(LABELS):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        results[name] = {'median': q50, 'err': (q84 - q16)/2, 'mean': np.mean(flat[:, i])}
    
    rhat = compute_rhat(chains_3d) if n_steps >= 200 else None
    
    return {
        'step': n_steps_done,
        'params': results,
        'acceptance': np.mean(af),
        'af_range': (af.min(), af.max()),
        'rhat': rhat,
        'n_samples': flat.shape[0],
    }


def render_screen(step, start_time, checkpoints_data):
    """Render the full terminal display."""
    now = datetime.now()
    elapsed = (now - start_time).total_seconds()
    fraction = step / TOTAL_STEPS
    
    # ETA
    if step > 0:
        rate = elapsed / step
        remaining = (TOTAL_STEPS - step) * rate
        eta = now + timedelta(seconds=remaining)
        eta_str = f"{remaining/3600:.1f}h ({eta.strftime('%b %d %H:%M')})"
        rate_str = f"{rate:.2f} s/step"
    else:
        eta_str = "calculating..."
        rate_str = "..."
    
    lines = []
    lines.append(f"{CLEAR}")
    lines.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}SDCG Production MCMC — Live Monitor{RESET}                           {BOLD}{CYAN}║{RESET}")
    lines.append(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════╣{RESET}")
    lines.append(f"{BOLD}{CYAN}║{RESET}  Step: {BOLD}{step:,}{RESET} / {TOTAL_STEPS:,}   |   Elapsed: {elapsed/60:.0f} min   |   Rate: {rate_str}  {BOLD}{CYAN}║{RESET}")
    lines.append(f"{BOLD}{CYAN}║{RESET}  {progress_bar(fraction, width=50)}     {BOLD}{CYAN}║{RESET}")
    lines.append(f"{BOLD}{CYAN}║{RESET}  ETA: {eta_str:<54s}{BOLD}{CYAN}║{RESET}")
    
    # Acceptance
    af_mean, af_lo, af_hi = get_acceptance()
    if af_mean:
        af_color = GREEN if 0.2 < af_mean < 0.5 else RED
        lines.append(f"{BOLD}{CYAN}║{RESET}  Acceptance: {af_color}{af_mean:.3f}{RESET} (range: {af_lo:.3f}–{af_hi:.3f})                     {BOLD}{CYAN}║{RESET}")
    
    lines.append(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════╣{RESET}")
    
    if not checkpoints_data:
        lines.append(f"{BOLD}{CYAN}║{RESET}  {DIM}Waiting for first checkpoint (step {CP_INTERVAL})...{RESET}                  {BOLD}{CYAN}║{RESET}")
        lines.append(f"{BOLD}{CYAN}║{RESET}  {DIM}Progress bar below from emcee tqdm in log file.{RESET}                {BOLD}{CYAN}║{RESET}")
    else:
        latest = checkpoints_data[-1]
        p = latest['params']
        
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}Latest Checkpoint: Step {latest['step']:,}{RESET} ({latest['n_samples']:,} post-burn samples)    {BOLD}{CYAN}║{RESET}")
        lines.append(f"{BOLD}{CYAN}║{RESET}                                                                  {BOLD}{CYAN}║{RESET}")
        
        # Parameter table header
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}{'Param':6s}  {'Median':>10s}  {'± 1σ':>8s}  {'Planck':>10s}  {'Δ':>6s}  Status{RESET}  {BOLD}{CYAN}║{RESET}")
        lines.append(f"{BOLD}{CYAN}║{RESET}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*6}  {BOLD}{CYAN}║{RESET}")
        
        for i, (name, planck) in enumerate(zip(LABELS, PLANCK)):
            r = p[name]
            med, err = r['median'], r['err']
            
            if planck is not None:
                delta_sigma = abs(med - planck) / err if err > 0 else 0
                delta_str = f"{delta_sigma:.1f}σ"
                if delta_sigma < 1:
                    status = f"{GREEN}✓{RESET}"
                elif delta_sigma < 3:
                    status = f"{YELLOW}~{RESET}"
                else:
                    status = f"{RED}✗{RESET}"
                planck_str = f"{planck:.5f}"
            else:
                # μ parameter
                sig = med / err if err > 0 else 0
                delta_str = f"{sig:.0f}σ"
                status = f"{GREEN}★{RESET}" if sig > 3 else f"{YELLOW}?{RESET}"
                planck_str = "    —"
            
            lines.append(f"{BOLD}{CYAN}║{RESET}  {name:6s}  {med:10.5f}  {err:8.5f}  {planck_str:>10s}  {delta_str:>6s}  {status}      {BOLD}{CYAN}║{RESET}")
        
        # Key results box
        mu = p['μ']
        h_val = p['h']
        ocdm = p['ω_cdm']
        H0 = h_val['median'] * 100
        
        lines.append(f"{BOLD}{CYAN}║{RESET}                                                                  {BOLD}{CYAN}║{RESET}")
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}┌─── KEY RESULTS ──────────────────────────────────────┐{RESET}     {BOLD}{CYAN}║{RESET}")
        
        mu_sig = mu['median'] / mu['err'] if mu['err'] > 0 else 0
        mu_color = GREEN if 0.05 < mu['median'] < 0.20 else RED
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  μ = {mu_color}{BOLD}{mu['median']:.4f} ± {mu['err']:.4f}{RESET}  ({mu_sig:.1f}σ detection)              {BOLD}│{RESET}     {BOLD}{CYAN}║{RESET}")
        
        h0_color = GREEN if 66 < H0 < 74 else RED
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  H₀ = {h0_color}{H0:.1f}{RESET} km/s/Mpc  (h = {h_val['median']:.4f})                {BOLD}│{RESET}     {BOLD}{CYAN}║{RESET}")
        
        ocdm_pct = abs(ocdm['median'] - 0.12) / 0.12 * 100
        ocdm_color = GREEN if ocdm_pct < 5 else RED
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  ω_cdm = {ocdm_color}{ocdm['median']:.5f}{RESET}  (Δ = {ocdm_pct:.1f}% from Planck)          {BOLD}│{RESET}     {BOLD}{CYAN}║{RESET}")
        
        # Health
        healthy = 0.05 < mu['median'] < 0.20 and ocdm_pct < 10 and 64 < H0 < 76
        health_str = f"{GREEN}HEALTHY{RESET}" if healthy else f"{RED}CHECK{RESET}"
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}│{RESET}  Chain health: {health_str}                                   {BOLD}│{RESET}     {BOLD}{CYAN}║{RESET}")
        lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}└──────────────────────────────────────────────────────┘{RESET}     {BOLD}{CYAN}║{RESET}")
        
        # R-hat convergence
        if latest['rhat'] is not None:
            lines.append(f"{BOLD}{CYAN}║{RESET}                                                                  {BOLD}{CYAN}║{RESET}")
            lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}R-hat Convergence:{RESET} (target < 1.01)                            {BOLD}{CYAN}║{RESET}")
            rhat_line = "  "
            for name, r in zip(LABELS, latest['rhat']):
                if r < 1.01:
                    rhat_line += f"{GREEN}{name}={r:.3f}✓{RESET} "
                elif r < 1.05:
                    rhat_line += f"{YELLOW}{name}={r:.3f}~{RESET} "
                else:
                    rhat_line += f"{RED}{name}={r:.3f}✗{RESET} "
            lines.append(f"{BOLD}{CYAN}║{RESET}{rhat_line}  {BOLD}{CYAN}║{RESET}")
            
            converged = np.all(latest['rhat'] < 1.01)
            conv_str = f"{GREEN}{BOLD}★ CONVERGED ★{RESET}" if converged else f"{YELLOW}Not yet{RESET}"
            lines.append(f"{BOLD}{CYAN}║{RESET}  Overall: {conv_str}                                              {BOLD}{CYAN}║{RESET}")
        
        # History sparklines (if multiple checkpoints)
        if len(checkpoints_data) > 1:
            lines.append(f"{BOLD}{CYAN}║{RESET}                                                                  {BOLD}{CYAN}║{RESET}")
            lines.append(f"{BOLD}{CYAN}║{RESET}  {BOLD}Parameter Evolution:{RESET}                                          {BOLD}{CYAN}║{RESET}")
            for name in ['μ', 'h', 'ω_cdm']:
                vals = [cp['params'][name]['median'] for cp in checkpoints_data]
                spark = spark_line(vals)
                lines.append(f"{BOLD}{CYAN}║{RESET}  {name:6s}: {spark} {vals[-1]:.5f}                              {BOLD}{CYAN}║{RESET}")
    
    lines.append(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════╣{RESET}")
    n_cps = len(checkpoints_data)
    total_cps = TOTAL_STEPS // CP_INTERVAL
    lines.append(f"{BOLD}{CYAN}║{RESET}  Checkpoints: {n_cps}/{total_cps}   |   Updated: {now.strftime('%H:%M:%S')}   |   Ctrl+C to stop  {BOLD}{CYAN}║{RESET}")
    lines.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    
    print('\n'.join(lines))


def main():
    run_dir = get_run_dir()
    if not run_dir:
        print("No production run found.")
        return
    
    start_time = datetime.fromtimestamp(os.path.getctime(run_dir))
    
    print(f"{BOLD}Starting live monitor...{RESET} (refreshes every 30s)")
    
    try:
        while True:
            step = get_current_step()
            
            # Load all checkpoints
            cp_files = sorted(glob.glob(os.path.join(run_dir, 'chains', 'checkpoint_*.npz')))
            checkpoints_data = []
            for cp in cp_files:
                try:
                    checkpoints_data.append(analyze_checkpoint(cp))
                except:
                    pass
            
            render_screen(step, start_time, checkpoints_data)
            
            if step >= TOTAL_STEPS:
                print(f"\n{GREEN}{BOLD}  MCMC COMPLETE!{RESET}\n")
                break
            
            time.sleep(30)
    
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped. MCMC continues in background.{RESET}")
        print(f"Resume: python3 monitor_live.py")


if __name__ == '__main__':
    main()
