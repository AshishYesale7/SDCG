#!/usr/bin/env python3
"""Check the atom interferometry screening values in thesis v13."""
import math

rho_crit = 9.47e-27  # kg/m^3
rho_thresh = 200.0 * rho_crit  # = 1.894e-24 kg/m^3
alpha = 2

rho_W = 19300.0   # kg/m^3 (Tungsten)
rho_Al = 2700.0   # kg/m^3 (Aluminum)
rho_Earth = 5500.0 # kg/m^3 (Earth average)

print("=" * 65)
print("SCREENING FUNCTION: S(rho) = 1 / (1 + (rho/rho_thresh)^alpha)")
print("=" * 65)
print(f"rho_crit   = {rho_crit:.2e} kg/m^3")
print(f"rho_thresh = {rho_thresh:.2e} kg/m^3")
print(f"alpha      = {alpha}")
print()

for name, rho in [("Tungsten (W)", rho_W),
                   ("Aluminum (Al)", rho_Al),
                   ("Earth avg", rho_Earth)]:
    ratio = rho / rho_thresh
    S = 1.0 / (1.0 + ratio**alpha)
    print(f"{name}:")
    print(f"  rho            = {rho:.0f} kg/m^3")
    print(f"  rho/rho_thresh = {ratio:.2e}")
    print(f"  S(rho)         = {S:.2e}")
    print()

S_Al = 1.0 / (1.0 + (rho_Al / rho_thresh)**2)
S_W  = 1.0 / (1.0 + (rho_W  / rho_thresh)**2)
dS = S_Al - S_W

print("=" * 65)
print("THESIS CLAIMS vs CORRECT VALUES")
print("=" * 65)
print(f"Thesis S_Al = 0.15     Correct S_Al = {S_Al:.2e}")
print(f"Thesis S_W  = 0.92     Correct S_W  = {S_W:.2e}")
print(f"Thesis dS   = -0.77    Correct dS   = {dS:.2e}")
print()

print("ORDER CHECK:")
print(f"  Al density ({rho_Al:.0f}) < W density ({rho_W:.0f})")
print(f"  => S_Al should be > S_W (less density = less screening)")
print(f"  Thesis says S_Al=0.15 < S_W=0.92 => REVERSED!")
print()

# What densities WOULD give those S values?
rho_015 = rho_thresh * math.sqrt(1.0/0.15 - 1.0)
rho_092 = rho_thresh * math.sqrt(1.0/0.92 - 1.0)
print(f"For S=0.15: need rho = {rho_015:.2e} kg/m^3 = {rho_015/rho_crit:.0f} rho_crit")
print(f"For S=0.92: need rho = {rho_092:.2e} kg/m^3 = {rho_092/rho_crit:.0f} rho_crit")
print("  (These are IGM/filament densities, NOT lab material densities!)")
print()

# Signal check
mu = 0.47
signal_thesis = 3.7e-9
signal_correct = mu * abs(dS)
print("=" * 65)
print("SIGNAL CHECK: delta_g/g")
print("=" * 65)
print(f"Thesis claims:  {signal_thesis:.1e}")
print(f"Correct (mu*|dS|): {mu} * {abs(dS):.2e} = {signal_correct:.2e}")
if signal_correct > 0:
    print(f"Overestimate:   {signal_thesis/signal_correct:.0e} x")
print()

# Solar system consistency check
print("=" * 65)
print("SOLAR SYSTEM CONSISTENCY CHECK")
print("=" * 65)
rho_moon = 3344.0  # kg/m^3
rho_sun_avg = 1408.0  # kg/m^3
for name, rho in [("Moon", rho_moon), ("Sun (avg)", rho_sun_avg), ("Earth", rho_Earth)]:
    S = 1.0 / (1.0 + (rho / rho_thresh)**2)
    mu_eff = mu * S
    print(f"{name}: rho={rho:.0f}, S={S:.2e}, mu_eff={mu_eff:.2e}")
print()
print("If S were O(0.1-1) at lab densities, it would be O(0.1-1)")
print("at Earth/Moon densities too => 10-50% gravity deviation")
print("=> violates lunar laser ranging (delta_g/g < 1e-13)")
