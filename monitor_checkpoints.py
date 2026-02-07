#!/usr/bin/env python3
"""
Checkpoint Monitor for SDCG Production MCMC
============================================
Reads all available checkpoint .npz files and reports key diagnostics:
  - Parameter medians ± 1σ
  - μ value and detection significance
  - H₀ derived estimate
  - Acceptance fraction
  - R-hat (if enough steps)
"""

import os
import sys
import glob
import numpy as np

RESULTS_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results'
)

# Find the latest production run directory
run_dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, 'production_run_*')))
if not run_dirs:
    print("No production run directories found.")
    sys.exit(1)

run_dir = run_dirs[-1]
print(f"Monitoring: {os.path.basename(run_dir)}")
print("=" * 80)

# Find all checkpoint files
checkpoint_files = sorted(glob.glob(os.path.join(run_dir, 'chains', 'checkpoint_*.npz')))

if not checkpoint_files:
    print("No checkpoints saved yet. Wait for the first 500 steps.")
    sys.exit(0)

LABELS = ['ω_b', 'ω_cdm', 'h', 'ln10As', 'n_s', 'τ', 'μ']


def compute_gelman_rubin(chains_3d):
    """R-hat from 3D chain (steps, walkers, dims)."""
    n_steps, n_walkers, n_dim = chains_3d.shape
    chains = chains_3d[n_steps // 2:]
    n = len(chains)
    R_hat = np.zeros(n_dim)
    for i in range(n_dim):
        chain_means = np.mean(chains[:, :, i], axis=0)
        W = np.mean(np.var(chains[:, :, i], axis=0, ddof=1))
        B = n * np.var(chain_means, ddof=1)
        var_hat = (1 - 1/n) * W + B/n
        R_hat[i] = np.sqrt(var_hat / W) if W > 0 else 1.0
    return R_hat


for cp_file in checkpoint_files:
    cp_name = os.path.basename(cp_file)
    step_num = int(cp_name.replace('checkpoint_', '').replace('.npz', ''))
    
    data = np.load(cp_file, allow_pickle=True)
    chains_3d = data['chains']  # (steps, walkers, dims)
    n_steps_done = int(data['n_steps_completed'])
    af = data['acceptance_fraction']
    
    n_steps, n_walkers, n_dim = chains_3d.shape
    
    # Use second half as "post-burn-in" samples
    burn = max(1, n_steps // 3)
    flat = chains_3d[burn:].reshape(-1, n_dim)
    
    print(f"\n{'━' * 80}")
    print(f"  CHECKPOINT {step_num:,} / 20,000  ({100*step_num/20000:.1f}%)")
    print(f"  Chain shape: {chains_3d.shape}  |  Post-burn samples: {flat.shape[0]:,}")
    print(f"  Acceptance: {np.mean(af):.3f} (range: {af.min():.3f} – {af.max():.3f})")
    print(f"{'━' * 80}")
    
    # Parameter constraints
    print(f"\n  {'Param':8s}  {'Median':>10s}  {'± 1σ':>10s}  {'Mean':>10s}  {'Planck':>10s}  {'Status'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    
    planck_vals = [0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544, None]
    
    for i, (name, planck) in enumerate(zip(LABELS, planck_vals)):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        mean = np.mean(flat[:, i])
        std = np.std(flat[:, i])
        err = (q84 - q16) / 2
        
        if planck is not None:
            delta = abs(q50 - planck)
            status = "✓" if delta < 3 * err else "⚠"
        else:
            status = "★" if q50 > 2 * std else "?"
        
        print(f"  {name:8s}  {q50:10.5f}  {err:10.5f}  {mean:10.5f}  "
              f"{'N/A' if planck is None else f'{planck:.5f}':>10s}  {status}")
    
    # Key derived quantities
    mu_samples = flat[:, 6]
    h_samples = flat[:, 2]
    mu_med = np.median(mu_samples)
    mu_std = np.std(mu_samples)
    h_med = np.median(h_samples)
    H0_direct = h_med * 100
    
    print(f"\n  ┌─── KEY RESULTS ───────────────────────────────────────────┐")
    print(f"  │  μ = {mu_med:.4f} ± {mu_std:.4f}  "
          f"({mu_med/mu_std:.1f}σ detection of μ ≠ 0)" if mu_std > 0 else "")
    print(f"  │  H₀ = {H0_direct:.2f} km/s/Mpc  (direct from h = {h_med:.4f})")
    print(f"  │  ω_cdm = {np.median(flat[:,1]):.5f}  "
          f"(Planck: 0.12000, Δ = {abs(np.median(flat[:,1])-0.12)/0.12*100:.1f}%)")
    
    # Health check
    healthy = True
    issues = []
    if mu_med > 0.20:
        healthy = False
        issues.append(f"μ={mu_med:.3f} > 0.20 (hitting bound)")
    if abs(np.median(flat[:, 1]) - 0.12) > 0.01:
        healthy = False
        issues.append(f"ω_cdm drifting from Planck")
    if np.mean(af) < 0.15 or np.mean(af) > 0.50:
        healthy = False
        issues.append(f"Acceptance rate abnormal: {np.mean(af):.3f}")
    
    if healthy:
        print(f"  │  ✅ HEALTH: Chain looks healthy")
    else:
        print(f"  │  ⚠️  HEALTH: {'; '.join(issues)}")
    print(f"  └─────────────────────────────────────────────────────────┘")
    
    # R-hat (only meaningful with enough steps)
    if n_steps >= 500:
        R_hat = compute_gelman_rubin(chains_3d)
        rhat_ok = np.all(R_hat < 1.01)
        print(f"\n  R-hat: ", end="")
        for name, r in zip(LABELS, R_hat):
            flag = "✓" if r < 1.01 else ("~" if r < 1.05 else "✗")
            print(f"{name}={r:.3f}{flag}  ", end="")
        print(f"\n  Overall: {'CONVERGED ★' if rhat_ok else 'Not yet converged'}")

print(f"\n{'=' * 80}")
print(f"  Checkpoints analyzed: {len(checkpoint_files)}")

# Time estimate
if len(checkpoint_files) >= 2:
    import datetime
    first_cp = os.path.getmtime(checkpoint_files[0])
    last_cp = os.path.getmtime(checkpoint_files[-1])
    n_cps = len(checkpoint_files)
    if n_cps > 1:
        time_per_cp = (last_cp - first_cp) / (n_cps - 1)
        remaining_cps = 40 - n_cps  # 20000/500 = 40 total
        eta_hours = remaining_cps * time_per_cp / 3600
        print(f"  Estimated time remaining: {eta_hours:.1f} hours")

print(f"{'=' * 80}\n")
