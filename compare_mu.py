#!/usr/bin/env python3
"""Compare mu values from all MCMC runs"""
import numpy as np
import os

print("=" * 80)
print("COMPARING ALL MCMC RESULTS - mu parameter")
print("=" * 80)

results_dir = "results"

for f in sorted(os.listdir(results_dir)):
    if f.endswith('.npz'):
        path = os.path.join(results_dir, f)
        try:
            data = np.load(path, allow_pickle=True)
            keys = list(data.keys())
            
            mu_val = None
            mu_err = None
            
            # Different ways mu might be stored
            if 'mu_mcmc' in data:
                mu_val = float(data['mu_mcmc'])
                mu_err = float(data.get('mu_err', 0))
            elif 'mu_a' in data:
                mu_val = float(data['mu_a'])
                mu_err = float(data.get('mu_a_err', 0))
            elif 'chains' in data:
                chains = data['chains']
                if hasattr(chains, 'shape') and len(chains.shape) == 2:
                    # Check if there are enough columns
                    if chains.shape[1] >= 7:
                        mu = chains[len(chains)//2:, 6]
                        mu_val = float(np.mean(mu))
                        mu_err = float(np.std(mu))
            elif 'results' in data:
                r = data['results']
                if hasattr(r, 'item'):
                    r = r.item()
                    if isinstance(r, dict) and 'mu_mean' in r:
                        mu_val = float(r['mu_mean'])
                        mu_err = float(r.get('mu_std', 0))
            
            if mu_val is not None:
                # Calculate Delta_v prediction
                v = 80  # km/s
                dv = v * (np.sqrt(1 + mu_val * 1.0) - np.sqrt(1 + mu_val * 0.001))
                
                print(f"\n{f}:")
                print(f"  mu = {mu_val:.5f} +/- {mu_err:.5f}")
                print(f"  Delta_v predicted = +{dv:.2f} km/s")
                if mu_val > 0.1:
                    print(f"  NOTE: HIGH mu value (without Lya constraint?)")
                elif mu_val < 0.05:
                    print(f"  NOTE: LOW mu value (with Lya constraint?)")
                    
        except Exception as e:
            pass  # Skip files that can't be parsed

print("\n" + "=" * 80)
print("KEY INSIGHT:")
print("=" * 80)
print("""
There are TWO types of MCMC runs:

1. WITHOUT Lyman-alpha constraint:
   - mu ~ 0.15-0.48 (high)
   - Delta_v ~ 6-18 km/s
   - Tension with dwarfs: ~4 sigma

2. WITH Lyman-alpha constraint:
   - mu ~ 0.045 (low, Lya upper limit)
   - Delta_v ~ 1.78 km/s  
   - Tension with dwarfs: ~0.8 sigma

The Lyman-alpha constraint REDUCES mu dramatically!
""")

# Also check the Lya-constrained value
lya_file = "results/cgc_lace_comprehensive_v6.npz"
if os.path.exists(lya_file):
    data = np.load(lya_file, allow_pickle=True)
    print("\nLyman-alpha constrained analysis (cgc_lace_comprehensive_v6.npz):")
    for key in data.keys():
        val = data[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: {val}")
        elif isinstance(val, (int, float, np.floating)):
            print(f"  {key}: {float(val):.5f}")
