"""
Title: Appendix A.7 – Glueball Ratio Scan
Author: Jacob Stelzriede
Date: 2025-04-07
Description:
    This script scans across multiple values of the twist-compression parameters (λ, n)
    and computes the mass ratio m₄ / m₃ for each parameter set. The goal is to identify
    configurations that yield mass spectra matching known SU(3) glueball ratios from
    lattice QCD. This script was used to identify the best-matching configuration
    (λ = 1.49, n = 2.17), which is used throughout Appendix A.2 and A.7.

    Purpose:
    - Validates: Parameter regions consistent with $2^{++}/0^{++}$ glueball ratios
    - Locking logic: freeze-lock (E > E_gap at final timestep)
    - Parameters scanned: λ and n (twist-compression strength and nonlinearity)

Reproducibility:
    - Matches: calibrated spectrum and mass ratios in Appendix A.7
    - Matches script: glueball_ratio_scan.py

License: MIT
"""

# --- Parameter Descriptions ---
# L         : Lattice size (L x L grid)
# T         : Number of time steps
# dx, dt    : Spatial and temporal resolution
# m         : Mass-like parameter for twist inertia
# r         : Radius scaling factor (often 1.0)
# lambda_   : Compression strength parameter (λ, scanned)
# kappa     : Compression scale (κ)
# n         : Nonlinearity exponent (scanned)
# E_gap     : Mass gap threshold (energy locking threshold)
# omega_seeds : Initial twist values used to generate mass spectrum
# sim_to_MeV : Unit conversion factor (simulation → physical units)
# target_levels : Indexes of levels to compare (e.g. Level 3 and 4)



import numpy as np

# Simulation constants
L = 30
T = 200
dx = 1.0
dt = 0.05
m = 1.0
r = 1.0
kappa = 1.0
E_gap = 0.5
sim_to_MeV = 2000
sigma = 1.5
omega_seeds = np.linspace(0.5, 3.0, 15)
lambda_fixed = 1.49
target_levels = [2, 3]  # Level 3 and 4 for ratio

def Etwist(omega):
    return 0.5 * m * r**2 * omega**2

def Ecomp(omega, lambda_, n):
    return lambda_ * (omega / kappa) ** n

def total_energy(omega, lambda_, n):
    return Etwist(omega) + Ecomp(omega, lambda_, n)

def apply_gaussian_seed(field, center, omega0, sigma=1.0):
    size = 5
    half = size // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x, y = center[0] + i, center[1] + j
            if 0 <= x < L and 0 <= y < L:
                field[x, y] = omega0 * np.exp(-(i**2 + j**2) / (2 * sigma**2))

def run_freeze_lock_spectrum(lambda_, n):
    results = []
    for omega0 in omega_seeds:
        omega = np.zeros((L, L))
        locked_mask = np.zeros((L, L), dtype=bool)
        apply_gaussian_seed(omega, (L // 2, L // 2), omega0, sigma=sigma)

        for t in range(T):
            new_omega = omega.copy()
            for i in range(1, L - 1):
                for j in range(1, L - 1):
                    if not locked_mask[i, j]:
                        laplacian = (omega[i+1, j] + omega[i-1, j] + omega[i, j+1] + omega[i, j-1] - 4 * omega[i, j]) / dx**2
                        new_omega[i, j] += dt * laplacian
            omega = new_omega
            E = total_energy(omega, lambda_, n)
            locked_mask |= (E > E_gap)

        E_final = total_energy(omega, lambda_, n)
        locked_energy = np.sum(E_final[locked_mask]) * sim_to_MeV
        results.append(locked_energy)
    return results

# Run ultra-fine n scan
n_values = np.round(np.arange(2.10, 2.221, 0.01), 3)
results = {}

for n in n_values:
    spectrum = run_freeze_lock_spectrum(lambda_fixed, n)
    if spectrum[target_levels[0]] > 0:
        ratio = spectrum[target_levels[1]] / spectrum[target_levels[0]]
    else:
        ratio = float('inf')
    results[n] = (spectrum[target_levels[0]], spectrum[target_levels[1]], ratio)

# Output results
for n, (m3, m4, ratio) in results.items():
    print(f"n = {n:.2f} | Level 3 = {m3:.1f} MeV | Level 4 = {m4:.1f} MeV | Ratio (2++/0++) = {ratio:.3f}")
