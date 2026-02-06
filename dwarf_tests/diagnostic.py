#!/usr/bin/env python3
"""
Diagnostic analysis to understand why cluster dwarfs rotate faster.
"""

import pandas as pd
import numpy as np

# Load data
void = pd.read_csv('filtered/void_dwarfs.csv')
cluster = pd.read_csv('filtered/cluster_dwarfs.csv')

print('DIAGNOSTIC: Understanding the unexpected result')
print('='*60)
print()

# Check W50 distributions
print('W50 (line width) distributions:')
print(f'  Void:    mean={void["W50"].mean():.1f}, median={void["W50"].median():.1f}')
print(f'  Cluster: mean={cluster["W50"].mean():.1f}, median={cluster["W50"].median():.1f}')
print()

# Check distance distributions 
print('Distance distributions (Mpc):')
print(f'  Void:    mean={void["Dist"].mean():.1f}, median={void["Dist"].median():.1f}')
print(f'  Cluster: mean={cluster["Dist"].mean():.1f}, median={cluster["Dist"].median():.1f}')
print()

# Check local density values
print('Local density (k=10 nearest neighbors):')
print(f'  Void:    mean={void["local_density"].mean():.4f}')
print(f'  Cluster: mean={cluster["local_density"].mean():.4f}')
print()

# Compare at very tight mass matching
print('Rotation velocity at matched HI masses:')
print('-'*50)
for logM in [8.5, 9.0, 9.2, 9.4]:
    v_m = void[(void['logMHI'] >= logM-0.1) & (void['logMHI'] < logM+0.1)]
    c_m = cluster[(cluster['logMHI'] >= logM-0.1) & (cluster['logMHI'] < logM+0.1)]
    if len(v_m) > 5 and len(c_m) > 5:
        delta = v_m['v_rot'].mean() - c_m['v_rot'].mean()
        print(f'  logMHI={logM:.1f}: void={v_m["v_rot"].mean():.1f}, cluster={c_m["v_rot"].mean():.1f}, Delta={delta:+.1f} km/s')
        print(f'            N_void={len(v_m)}, N_cluster={len(c_m)}')
    else:
        print(f'  logMHI={logM:.1f}: insufficient data (void={len(v_m)}, cluster={len(c_m)})')

print()
print('INTERPRETATION:')
print('='*60)
print()
print('Cluster dwarfs show HIGHER W50 at the same HI mass.')
print()
print('Possible physical explanations (not SDCG):')
print('  1. Tidal interactions in dense environments broaden HI profiles')
print('  2. Ram pressure from ICM affects gas kinematics')
print('  3. Galaxy harassment increases velocity dispersion')
print('  4. Beam confusion in clusters adds non-rotational motion')
print()
print('These are KNOWN astrophysical effects independent of gravity theory.')
print()
print('CONCLUSION FOR THESIS:')
print('='*60)
print()
print('This test CANNOT distinguish SDCG from LCDM because:')
print('  - Environmental effects dominate over any gravity modification')
print('  - W50 is not a clean rotation measurement in dense environments')
print('  - Need resolved rotation curves, not integrated line widths')
