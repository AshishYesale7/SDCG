#!/usr/bin/env python3
"""
Diagnose MCMC: compare likelihood components + verify fixes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from simulations.cgc.likelihoods import (
    log_prior, log_prior_gaussian,
    log_likelihood_cmb, log_likelihood_bao, log_likelihood_sne,
    log_likelihood_h0, log_likelihood_growth, log_likelihood_lyalpha,
    log_likelihood, log_probability
)
from run_production_mcmc import load_data, PRODUCTION_CONFIG

data = load_data()

# Two parameter sets to compare
theta_thesis = np.array([
    0.02237, 0.1200, 0.7000, 3.044, 0.9649, 0.0544, 0.47
])

theta_collapsed = np.array([
    0.02260, 0.17879, 0.65031, 3.29843, 0.99899, 0.02064, 0.35934
])

theta_planck_lcdm = np.array([
    0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544, 0.0
])

print("="*75)
print("MCMC FIX VERIFICATION (CMB disabled, BAO fixed, Planck priors)")
print("="*75)

for label, theta in [("Thesis CGC (h=0.70, μ=0.47)", theta_thesis),
                     ("Collapsed chain (@500)", theta_collapsed),
                     ("Planck ΛCDM (μ=0)", theta_planck_lcdm)]:
    print(f"\n{'─'*75}")
    print(f"  {label}")
    print(f"{'─'*75}")
    
    lp = log_prior(theta)
    
    ll_bao = log_likelihood_bao(theta, data['bao']) if 'bao' in data else 0
    ll_sne = log_likelihood_sne(theta, data['sne']) if 'sne' in data else 0
    ll_h0 = log_likelihood_h0(theta, data['H0']) if 'H0' in data else 0
    ll_growth = log_likelihood_growth(theta, data['growth']) if 'growth' in data else 0
    ll_lya = log_likelihood_lyalpha(theta, data['lyalpha']) if 'lyalpha' in data else 0
    
    ll_total = ll_bao + ll_sne + ll_h0 + ll_growth + ll_lya
    posterior = lp + ll_total
    
    # Full log_probability as MCMC sees it
    full_post = log_probability(theta, data, **{
        'include_cmb': PRODUCTION_CONFIG['include_cmb'],
        'include_sne': PRODUCTION_CONFIG['include_sne'],
        'include_lyalpha': PRODUCTION_CONFIG['include_lyalpha'],
        'use_eft_priors': PRODUCTION_CONFIG['use_eft_priors'],
        'tight_eft_priors': PRODUCTION_CONFIG['tight_eft_priors'],
    })
    
    print(f"  Prior (w/ Gaussians):  {lp:>10.2f}")
    print(f"  BAO:                   {ll_bao:>10.2f}")
    print(f"  SNe:                   {ll_sne:>10.2f}")
    print(f"  H₀:                   {ll_h0:>10.2f}")
    print(f"  Growth:                {ll_growth:>10.2f}")
    print(f"  Lyα:                   {ll_lya:>10.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  Total LL:              {ll_total:>10.2f}")
    print(f"  Posterior (manual):     {posterior:>10.2f}")
    print(f"  Posterior (MCMC):       {full_post:>10.2f}")

print(f"\n{'='*75}")
print("EXPECTED: Thesis CGC should have BEST posterior (highest value)")
print("If collapsed is still better, there's still a problem.")
print("="*75)
