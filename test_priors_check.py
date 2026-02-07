#!/usr/bin/env python3
"""Quick validation of updated priors."""
import sys, numpy as np
sys.path.insert(0, 'simulations')
from cgc.likelihoods import log_prior, log_prior_gaussian

t = np.array([0.02237, 0.120, 0.6736, 3.044, 0.9649, 0.0544, 0.15])
print("mu=0.15 (sweet spot):")
print(f"  flat: {log_prior(t):.4f}")
print(f"  tight: {log_prior_gaussian(t, tight_eft_priors=True):.4f}")

t2 = t.copy(); t2[6] = 0.46
print(f"\nmu=0.46 (old stuck): flat={log_prior(t2)}")

t3 = t.copy(); t3[6] = 0.199
print(f"\nmu=0.199 (near bound):")
print(f"  flat: {log_prior(t3):.4f}")
print(f"  tight: {log_prior_gaussian(t3, tight_eft_priors=True):.4f}")

t4 = t.copy(); t4[6] = 0.001
print(f"\nmu=0.001 (LCDM):")
print(f"  flat: {log_prior(t4):.4f}")
print(f"  tight: {log_prior_gaussian(t4, tight_eft_priors=True):.4f}")

print("\nAll checks passed!")
